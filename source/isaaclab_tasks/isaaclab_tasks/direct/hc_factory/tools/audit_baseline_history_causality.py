#!/usr/bin/env python3
"""Measure historical-hot lookahead and frozen-model sensitivity on train/validation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from factory_baselines.dataset import FactoryBaselineTensorDataset, load_shared_dataset
from factory_baselines.evaluation import event_rule_kwargs
from factory_baselines.torch_trainer import _manifest_hash, _model_inputs, _resolve_device, load_checkpoint
from factory_bn_shared.remain import station_report_metrics
from reevaluate_main_validation import compare_reports, file_hash, load_standalone_metrics


def ongoing_group(will, start, historical_hot, eligible):
    return ((historical_hot > .5) | (start == 0)) & (will > .5) & (eligible > .5)


def prefix_flag(reference, features, t, window_size, min_hot, gap):
    if not 0 < t <= len(features):
        raise ValueError("Historical flag requires a nonempty observed prefix")
    return reference.ops_hot_mask(features[:t], window_size_s=window_size,
                                  min_hot_windows=min_hot, gap_windows=gap)[-1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset_dir", "main_bundle", "pdformer_root", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(4)
    device = _resolve_device(args.device)
    payload, manifest = load_shared_dataset(args.dataset_dir)
    manifest_sha = _manifest_hash(args.dataset_dir / "dataset_manifest.json")
    split_path = args.dataset_dir / "episode_split_audit.json"
    input_path = args.dataset_dir / "validation_contract_audit.json"
    split = json.loads(split_path.read_text())
    audit = json.loads(input_path.read_text())
    if not split["episode_split_match"] or not audit["comparison_match"]:
        raise ValueError("Resolve existing alignment audits before history sensitivity")
    if (audit["split_audit_sha256"] != file_hash(split_path)
        or split["provenance"]["baseline_manifest"]["sha256"] != manifest_sha
        or split["provenance"]["main_bundle"]["sha256"] != file_hash(args.main_bundle / "episodes.npz")):
        raise ValueError("Dataset/bundle/split no longer matches the input audit")
    reference_path = args.pdformer_root / "factory_bn/remain.py"
    if file_hash(reference_path) != audit["main_remain_source_sha256"]:
        raise ValueError("Main operational-hot logic changed after input alignment")
    reference = load_standalone_metrics(reference_path)
    # These are the already-audited operational event contract, not tuning knobs.
    window_size, min_hot, gap = 60., 8, 1
    if int(manifest["event_min_windows"]) != min_hot:
        raise ValueError("Unexpected event contract")
    with (args.dataset_dir / "model_sample_index.csv").open(newline="", encoding="utf-8") as stream:
        rows = [row for row in csv.DictReader(stream) if row["split"] in {"train", "validation"}]
    row_by_index = {int(row["sample_index"]): row for row in rows}
    if len(row_by_index) != len(rows):
        raise ValueError("Duplicate sample indices")
    expected_indices = set(payload["split_indices"]["train"].tolist()) | set(payload["split_indices"]["validation"].tolist())
    if set(row_by_index) != expected_indices:
        raise ValueError("CSV/tensor train and validation indices differ")
    inverse = {group: name for name, group in split["main_episode_mapping"].items()}
    if len(inverse) != len(split["main_episode_mapping"]):
        raise ValueError("Episode mapping is not one-to-one")
    groups = {}
    for row in rows:
        groups.setdefault(row["group_id"], []).append(row)
    samples = FactoryBaselineTensorDataset(payload)
    flags = {}
    counts = {name: {"samples": 0, "eligible_node_windows": 0, "changed_history": 0,
                    "changed_positive_history": 0, "changed_ongoing_loss_group": 0,
                    "positive_events": 0, "upcoming_events": 0}
              for name in ("train", "validation")}
    with np.load(args.main_bundle / "episodes.npz", allow_pickle=False) as bundle:
        main_nodes = bundle["resource_ids"].tolist()
        if set(main_nodes) != set(manifest["node_ids"]):
            raise ValueError("Main/baseline node catalog mismatch")
        order = [main_nodes.index(node) for node in manifest["node_ids"]]
        for group, group_rows in groups.items():
            features = bundle[inverse[group] + "_features"][:, order]
            full_hot = reference.ops_hot_mask(features, window_size_s=window_size,
                                              min_hot_windows=min_hot, gap_windows=gap)
            for row in group_rows:
                index = int(row["sample_index"])
                sample = samples[index]
                t = int(sample["target_start_position"])
                original = sample["hist_last_hot"].numpy()
                if not np.array_equal(original, full_hot[t - 1]):
                    raise ValueError("Frozen bundle does not reproduce historical flags")
                causal = prefix_flag(reference, features, t, window_size, min_hot, gap)
                flags[index] = causal
                eligible = sample["occ_node_mask"].numpy() > .5
                will, start = sample["event_will"].numpy(), sample["event_start"].numpy()
                changed = (original != causal) & eligible
                old_group = ongoing_group(will, start, original, eligible)
                new_group = ongoing_group(will, start, causal, eligible)
                count = counts[row["split"]]
                count["samples"] += 1
                count["eligible_node_windows"] += int(eligible.sum())
                count["changed_history"] += int(changed.sum())
                count["changed_positive_history"] += int((changed & (will > .5)).sum())
                count["changed_ongoing_loss_group"] += int((old_group != new_group).sum())
                count["positive_events"] += int(((will > .5) & eligible).sum())
                count["upcoming_events"] += int(((will > .5) & (start > 0) & eligible).sum())
    print("Historical flags and loss groups:", json.dumps(counts), flush=True)
    results = []
    loader = DataLoader(FactoryBaselineTensorDataset(payload, payload["split_indices"]["validation"].tolist()),
                        batch_size=32, shuffle=False)
    for path in args.checkpoints:
        model, checkpoint = load_checkpoint(path, device)
        if checkpoint["metadata"]["dataset_manifest_sha256"] != manifest_sha:
            raise ValueError("Checkpoint manifest differs")
        if checkpoint["train_config"]["evaluate_test"]:
            raise ValueError("Use validation-only development checkpoints in this audit")
        collected = {}
        model.eval()
        with torch.no_grad():
            for batch in loader:
                causal = np.stack([flags[int(index)] for index in batch["sample_index"]])
                batch = {key: value.to(device) for key, value in batch.items()}
                output = model(**_model_inputs(batch))
                values = {key: batch[key] for key in ("y_hot", "remain_mask", "occ_node_mask", "hist_last_hot")}
                values.update(probability=output["event_will_logit"].sigmoid(),
                              start=output["event_start_logit"].argmax(-1), duration=output["event_duration"])
                for key, value in values.items():
                    collected.setdefault(key, []).append(value.detach().cpu().numpy())
                collected.setdefault("prefix_hot", []).append(causal)
        arrays = {key: np.concatenate(value) for key, value in collected.items()}
        threshold = float(checkpoint["metadata"]["event_report_threshold"])
        scores = {}
        for policy, key in (("full_episode", "hist_last_hot"), ("observed_prefix", "prefix_hot")):
            scores[policy] = station_report_metrics(
                arrays["y_hot"], arrays["probability"], arrays["start"], arrays["duration"],
                arrays["remain_mask"], arrays["occ_node_mask"], threshold=threshold,
                **event_rule_kwargs(min_hot), start_tol_windows=3,
                hist_last_hot=arrays[key], target_hist_last_hot=arrays["hist_last_hot"],
                force_ongoing_will=False,
            )
        saved_path = path.parent / "metrics.json"
        saved = json.loads(saved_path.read_text())["validation"]["station_report"]
        difference = compare_reports(scores["full_episode"], saved,
                     ["report_precision", "report_recall", "report_f1", "n_pred_who", "n_true_who", "start_mae", "dur_mae"])
        if difference:
            raise ValueError(f"Saved validation scores did not reproduce: {difference}")
        record = {"checkpoint": str(path.resolve()), "checkpoint_sha256": file_hash(path),
                  "epoch": int(checkpoint["epoch"]), "threshold": threshold, "scores": scores,
                  "delta_prefix_minus_full": {key: scores["observed_prefix"][key] - scores["full_episode"][key]
                                              for key in scores["full_episode"]}}
        results.append(record)
        print(path.parent, "P/R/F1 full->prefix:",
              [(round(scores[p]["report_precision"], 6), round(scores[p]["report_recall"], 6),
                round(scores[p]["report_f1"], 6)) for p in ("full_episode", "observed_prefix")], flush=True)
        del model
    report = {"status": "completed", "test_evaluated": False, "training_performed": False,
              "dataset_manifest_sha256": manifest_sha, "main_bundle_sha256": file_hash(args.main_bundle / "episodes.npz"),
              "source_reference_sha256": file_hash(reference_path), "counts": counts, "checkpoints": results,
              "scope": "Historical hot flag only. Identical saved weights, X, labels, masks and saved thresholds. "
                       "Prefix flags are diagnostic, not a replacement of the shared experiment contract. "
                       "Even unchanged loss groups do not prove unchanged checkpoint/threshold selection in a new training run."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("Output:", args.output, flush=True)


if __name__ == "__main__":
    main()
