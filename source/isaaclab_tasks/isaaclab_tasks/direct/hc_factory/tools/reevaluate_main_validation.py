#!/usr/bin/env python3
"""Read-only replay of the frozen main checkpoint on audited validation episodes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_standalone_metrics(path: Path):
    spec = importlib.util.spec_from_file_location("baseline_replay_remain", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validation_names(rows: list[dict], mapping: dict[str, str]) -> set[str]:
    if any(row["split"] != "validation" for row in rows):
        raise ValueError("This replay must not score train or test")
    inverse = {group: name for name, group in mapping.items()}
    if len(inverse) != len(mapping):
        raise ValueError("Episode mapping is not one-to-one")
    groups = {row["group_id"] for row in rows}
    if not groups or not groups.issubset(inverse):
        raise ValueError("Validation groups are missing from the audited mapping")
    return {inverse[group] for group in groups}


def compare_reports(actual: dict, expected: dict, keys: list[str]) -> dict:
    return {
        key: {"actual": float(actual[key]), "expected": float(expected[key])}
        for key in keys
        if not np.isclose(actual[key], expected[key], atol=1e-6, rtol=1e-6)
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset_dir", "main_bundle", "main_checkpoint", "pdformer_root", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--main_source_commit", required=True)
    parser.add_argument("--thresholds", type=float, nargs="+", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not args.thresholds or any(not 0 < value < 1 for value in args.thresholds):
        raise ValueError("Provide probability thresholds strictly between zero and one")
    torch.set_num_threads(4)
    root = args.pdformer_root.resolve()
    source_commit = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    if source_commit != args.main_source_commit:
        raise ValueError("Main source changed: do not replay with unverified code")
    subprocess.run(["git", "-C", str(root), "diff", "--exit-code", "HEAD", "--", "factory_bn"], check=True)
    split_path = args.dataset_dir / "episode_split_audit.json"
    input_path = args.dataset_dir / "validation_contract_audit.json"
    split = json.loads(split_path.read_text(encoding="utf-8"))
    audited = json.loads(input_path.read_text(encoding="utf-8"))
    if not split["episode_split_match"] or not audited["comparison_match"]:
        raise ValueError("Resolve data alignment before scoring the main checkpoint")
    if audited["split_audit_sha256"] != file_hash(split_path):
        raise ValueError("Input audit is not bound to this episode split audit")
    if audited["main_remain_source_sha256"] != file_hash(root / "factory_bn/remain.py"):
        raise ValueError("Main target construction changed after the alignment audit")
    for name, path in (
        ("baseline_manifest", args.dataset_dir / "dataset_manifest.json"),
        ("main_bundle", args.main_bundle / "episodes.npz"),
        ("main_checkpoint", args.main_checkpoint),
    ):
        if file_hash(path) != split["provenance"][name]["sha256"]:
            raise ValueError(f"{name} changed after the episode split audit")
    manifest = json.loads((args.dataset_dir / "dataset_manifest.json").read_text())
    if manifest["dataset_version"] != "factory_baseline_dataset_v5":
        raise ValueError("This replay requires the aligned v5 dataset")
    with (args.dataset_dir / "model_sample_index.csv").open(newline="", encoding="utf-8") as stream:
        rows = [row for row in csv.DictReader(stream) if row["split"] == "validation"]
    names = validation_names(rows, split["main_episode_mapping"])
    if len(rows) != audited["sample_count"] or len(names) != audited["episode_count"]:
        raise ValueError("Validation cohort changed after the input audit")

    # Keep supplied main imports separate from this repository's copied factory_bn.
    if any(name == "factory_bn" or name.startswith("factory_bn.") for name in sys.modules):
        raise RuntimeError("Run this replay as a fresh process to isolate main imports")
    sys.path.insert(0, str(root))
    from factory_bn import dataset as reference_dataset
    from factory_bn import remain as reference_metrics
    from factory_bn.infer import _data_feature_from_ckpt
    from factory_bn.model import BNPDFormer

    if Path(reference_dataset.__file__).resolve().parent != root / "factory_bn":
        raise RuntimeError("Imported the wrong main experiment implementation")
    checkpoint = torch.load(args.main_checkpoint, map_location="cpu", weights_only=False)
    config, meta = dict(checkpoint["config"]), checkpoint["data_meta"]
    if config["train_mode"] != "unsupervised" or not config["remain_to_jobs_done"] or config["use_stgnpp"]:
        raise ValueError("Expected the audited operational-occupancy checkpoint without STGNPP")
    if int(config["event_min_windows"]) != int(manifest["event_min_windows"]) or int(config["start_tol_windows"]) != 3:
        raise ValueError("Main/baseline event matching settings differ")
    bundle = reference_dataset.load_factory_bn_bundle(args.main_bundle)
    if bundle["resource_ids"] != list(meta["resource_ids"]) or bundle["resource_types"] != list(meta["resource_types"]):
        raise ValueError("Checkpoint node order/types differ from the frozen bundle")
    episodes = [episode for episode in bundle["episodes"] if episode["name"] in names]
    if {episode["name"] for episode in episodes} != names:
        raise ValueError("Missing validation episodes in main bundle")
    by_group = {}
    for row in rows:
        by_group.setdefault(row["group_id"], []).append(row)
    samples, history_difference, history_cells = [], 0, 0
    for episode in episodes:
        features = episode["features"]
        positions = [t for t in range(int(config["input_window"]), min(len(features),
                     reference_metrics.first_done_index(episode["jobs_remaining"])))
                     if episode["jobs_remaining"][t - 1] > 0]
        cohort = by_group[split["main_episode_mapping"][episode["name"]]]
        by_anchor = {float(row["anchor_time_s"]): row for row in cohort}
        if len(by_anchor) != len(cohort) or len(positions) != len(cohort):
            raise ValueError("Main/baseline validation window counts differ")
        part = reference_dataset._build_samples(
            [episode], input_window=int(config["input_window"]),
            output_window=int(config["output_window"]),
            horizon_windows=int(round(config["horizon_s"] / config["window_size_s"])),
            max_hist_events=int(config["max_hist_events"]), window_size_s=float(config["window_size_s"]),
            horizon_s=float(config["horizon_s"]), remain_to_jobs_done=True,
            max_remain_windows=int(config["max_remain_windows"]),
            occupancy_horizon_windows=int(config["occupancy_horizon_windows"]),
            hot_min_windows=int(config["hot_min_windows"]), hot_gap_windows=int(config["hot_gap_windows"]),
            train_mode="unsupervised",
        )
        if len(part) != len(positions):
            raise ValueError("Main sample builder produced unexpected positions")
        for t, sample in zip(positions, part):
            row = by_anchor[float(episode["window_start_s"][t - 1])]
            if (float(row["first_future_start_s"]) != float(episode["window_start_s"][t])
                or json.loads(row["input_window_indices"]) != episode["windows"][t-int(config["input_window"]):t].tolist()):
                raise ValueError("Main/baseline sample anchors differ")
            if sample["x"].shape[-1] != int(meta["feature_dim"]):
                raise ValueError("Feature count differs: no padding/trimming allowed")
            prefix_hot = reference_metrics.ops_hot_mask(
                features[:t], window_size_s=float(config["window_size_s"]),
                min_hot_windows=int(config["hot_min_windows"]), gap_windows=int(config["hot_gap_windows"]),
            )[-1]
            valid = sample["occ_node_mask"] > .5
            history_difference += int(((prefix_hot != sample["hist_last_hot"]) & valid).sum())
            history_cells += int(valid.sum())
        samples.extend(part)
    if len(samples) != len(rows):
        raise ValueError("Validation sample totals differ")
    scalers = []
    for kind, dimension in (("feature", int(meta["feature_dim"])), ("score", 1)):
        mean = np.asarray(meta[f"{kind}_scaler_mean"], dtype=np.float32)
        std = np.asarray(meta[f"{kind}_scaler_std"], dtype=np.float32)
        if mean.shape != (dimension,) or std.shape != (dimension,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any():
            raise ValueError("Saved scaler shape/values invalid; do not refit")
        scalers.append(reference_dataset.Scaler(mean=mean, std=std))
    config["device"] = torch.device(args.device)
    model = BNPDFormer(config, _data_feature_from_ckpt(meta)).to(config["device"])
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    collected = {}
    loader = DataLoader(reference_dataset.FactoryBNWindowDataset(samples, *scalers),
                        batch_size=args.batch_size, shuffle=False)
    with torch.no_grad():
        for batch in loader:
            inputs = {key: batch[key].to(config["device"]) for key in (
                "X", "jobs_remaining", "jobs_total", "hist_last_hot", "hist_cluster", "hist_cluster_prev", "hist_tpm",
            )}
            predicted = model.predict(inputs)
            values = {key: batch[key] for key in ("y_hot", "remain_mask", "occ_node_mask", "hist_last_hot")}
            values.update(raw_probability=predicted["event_will_logit"].sigmoid(),
                          raw_start=predicted["event_start_logit"].argmax(-1),
                          original_probability=predicted["event_will_prob"],
                          original_start=predicted["event_start_idx"], duration=predicted["event_dur"])
            for key, value in values.items():
                collected.setdefault(key, []).append(value.detach().cpu().numpy())
    arrays = {key: np.concatenate(parts) for key, parts in collected.items()}
    for key in ("raw_probability", "raw_start", "original_probability", "original_start", "duration", "hist_last_hot"):
        if arrays[key].shape != arrays["occ_node_mask"].shape or not np.isfinite(arrays[key]).all():
            raise ValueError(f"Unexpected prediction shape/non-finite values: {key}")
    baseline_path = Path(__file__).resolve().parents[1] / "PDFormer/factory_bn/remain.py"
    baseline_metrics = load_standalone_metrics(baseline_path)

    def score(module, probability, start, threshold, force):
        return module.station_report_metrics(
            arrays["y_hot"], arrays[probability], arrays[start], arrays["duration"],
            arrays["remain_mask"], arrays["occ_node_mask"], threshold=threshold,
            min_windows=int(config["event_min_windows"]), start_tol_windows=int(config["start_tol_windows"]),
            hist_last_hot=arrays["hist_last_hot"], force_ongoing_will=force,
            will_floor=float(config["ongoing_will_floor"]),
        )

    original = score(reference_metrics, "original_probability", "original_start",
                     float(config["event_report_threshold"]), bool(config["force_ongoing_will"]))
    keys = ["report_precision", "report_recall", "report_f1", "report_recall_ongoing",
            "report_recall_upcoming", "n_true_who", "n_pred_who", "start_mae", "dur_mae"]
    original_diff = compare_reports(original, checkpoint["val_metrics"], keys)
    common = []
    for threshold in args.thresholds:
        row = score(baseline_metrics, "raw_probability", "raw_start", threshold, False)
        reference = score(reference_metrics, "raw_probability", "raw_start", threshold, False)
        if compare_reports(row, reference, list(reference)):
            raise ValueError("Main and baseline scoring disagree on identical predictions")
        common.append({"threshold": threshold, **row})
    result = {
        "scope": "frozen main checkpoint validation replay; no retraining or threshold selection",
        "test_evaluated": False, "sample_count": len(samples), "episode_names": sorted(names),
        "checkpoint_epoch": int(checkpoint["epoch"]), "checkpoint_sha256": file_hash(args.main_checkpoint),
        "source_commit": source_commit, "main_factory_bn_source_hashes": {
            str(path.relative_to(root)): file_hash(path) for path in sorted((root / "factory_bn").rglob("*.py"))
        }, "baseline_metrics_sha256": file_hash(baseline_path),
        "input_audit_sha256": file_hash(input_path), "split_audit_sha256": file_hash(split_path),
        "original_decode": original, "saved_metrics_reproduced": not original_diff,
        "original_policy": {key: config[key] for key in (
            "event_report_threshold", "event_min_windows", "start_tol_windows", "force_ongoing_will", "ongoing_will_floor",
        )},
        "saved_metrics_differences": original_diff, "common_decode": common,
        "common_metric_implementations_agree": True,
        "observed_history_prefix_diagnostic": {"different_node_windows": history_difference,
                                               "eligible_node_windows": history_cells},
        "limitations": "Both replay policies retain the existing full-episode-smoothed hist_last_hot. "
                       "Prefix differences are diagnostic only; no causal-metric claim is made. "
                       "Current source and bundle do not prove historical training provenance. "
                       "No future target tensors are passed to model.predict.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("original reproduced:", not original_diff, "differences:", original_diff)
    print("original P/R/F1:", *[original[key] for key in keys[:3]])
    for row in common:
        print("common", row["threshold"], *[round(row[key], 6) for key in keys[:3]],
              "upcoming", round(row["report_recall_upcoming"], 6))
    print("history prefix diagnostic:", result["observed_history_prefix_diagnostic"])
    print("Output:", args.output)


if __name__ == "__main__":
    main()
