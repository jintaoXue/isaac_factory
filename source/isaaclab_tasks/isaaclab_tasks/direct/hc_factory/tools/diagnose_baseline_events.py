#!/usr/bin/env python3
"""Diagnose train/validation event errors without changing scoring or selecting models."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from factory_baselines.dataset import FactoryBaselineTensorDataset, load_shared_dataset
from factory_baselines.metrics import _binary_metrics
from factory_baselines.torch_trainer import (
    _manifest_hash,
    _model_inputs,
    _resolve_device,
    load_checkpoint,
)
from factory_bn_shared.remain import station_report_metrics


def attach_node_catalog(report: dict, catalog_path: Path, manifest: dict) -> None:
    with catalog_path.open(newline="", encoding="utf-8") as stream:
        catalog = list(csv.DictReader(stream))
    by_index = {int(node["node_index"]): node for node in catalog}
    if len(by_index) != len(catalog) or set(by_index) != set(range(len(manifest["node_ids"]))):
        raise ValueError("Node catalog indices do not match the dataset manifest")
    for index, node_id in enumerate(manifest["node_ids"]):
        node = by_index[index]
        if (node["resource_id"] != node_id or node["resource_type"] !=
                manifest["resource_types"][int(node["resource_type_index"])]):
            raise ValueError("Node catalog identities/types differ from the dataset manifest")
    for row in report["thresholds"]:
        for node in row["per_node"]:
            source = by_index[node["node_index"]]
            node.update(resource_id=source["resource_id"], resource_type=source["resource_type"])


def summarize_events(arrays: dict[str, np.ndarray], thresholds: list[float]) -> dict:
    valid = arrays["occ_node_mask"] > 0.5
    positive = (arrays["event_will"] > 0.5) & valid
    start = arrays["event_start"]
    groups = {
        "ongoing": positive & (start == 0),
        "upcoming": positive & (start > 0),
        "negative": valid & ~positive,
    }
    probability = arrays["will_probability"]
    decoded_start = np.where(
        arrays["hist_last_hot"] > 0.5, 0, arrays["predicted_start"]
    )
    future_observed = arrays["remain_mask"] > 0.5
    short_horizon = np.broadcast_to(future_observed.sum(axis=1)[:, None] < 8, valid.shape)
    any_future_hot = ((arrays["y_hot"] > 0.5) & future_observed[:, :, None]).any(axis=1)
    negative_kinds = {
        "short_observed_horizon": groups["negative"] & short_horizon,
        "hot_without_qualifying_event": groups["negative"] & ~short_horizon & any_future_hot,
        "no_future_hot": groups["negative"] & ~short_horizon & ~any_future_hot,
    }
    output = {"groups": {}, "thresholds": [], "ranking": {}}
    ranking_groups = {
        "all_events": (positive, valid),
        "ongoing_vs_negative": (groups["ongoing"], groups["ongoing"] | groups["negative"]),
        "upcoming_vs_negative": (groups["upcoming"], groups["upcoming"] | groups["negative"]),
        "events_vs_short_hot_negative": (
            positive, positive | negative_kinds["hot_without_qualifying_event"]
        ),
    }
    for name, (target, mask) in ranking_groups.items():
        metrics = _binary_metrics(target[mask].astype(np.int64), probability[mask])
        count = int(mask.sum())
        prevalence = float(target[mask].mean()) if count else None
        output["ranking"][name] = {
            "sample_count": count,
            "positive_count": metrics["positive_count"],
            "negative_count": metrics["negative_count"],
            "positive_rate": prevalence,
            "tie_aware_average_precision": metrics["pr_auc"],
            "roc_auc": metrics["roc_auc"],
            "ap_over_prevalence": (
                metrics["pr_auc"] / prevalence if prevalence else None
            ),
        }
    for name, mask in groups.items():
        values = probability[mask]
        output["groups"][name] = {
            "count": int(mask.sum()),
            "will_q10_q50_q90": (
                np.quantile(values, [0.1, 0.5, 0.9]).tolist() if values.size else None
            ),
            "hist_hot_count": int((mask & (arrays["hist_last_hot"] > 0.5)).sum()),
        }
        if name != "negative":
            errors = np.abs(decoded_start[mask] - start[mask])
            output["groups"][name]["start_within_tolerance_rate"] = (
                float((errors <= 3).mean()) if errors.size else None
            )
    for threshold in thresholds:
        report = station_report_metrics(
            arrays["y_hot"], probability, arrays["predicted_start"],
            arrays["predicted_duration"], arrays["remain_mask"],
            arrays["occ_node_mask"], threshold=threshold,
            min_windows=8, start_tol_windows=3,
            hist_last_hot=arrays["hist_last_hot"], force_ongoing_will=False,
        )
        row = {"threshold": threshold, **report}
        predicted = probability >= threshold
        for name in ("ongoing", "upcoming"):
            mask = groups[name]
            probability_miss = mask & ~predicted
            timing_miss = mask & predicted & (np.abs(decoded_start - start) > 3)
            row[f"{name}_probability_misses"] = int(probability_miss.sum())
            row[f"{name}_timing_misses"] = int(timing_miss.sum())
        row["false_positive_stations"] = int((predicted & groups["negative"]).sum())
        timing_error = positive & predicted & (np.abs(decoded_start - start) > 3)
        hits = positive & predicted & ~timing_error
        false_alarm = (predicted & groups["negative"]) | timing_error
        counts = {"true_event_wrong_start": int(timing_error.sum())}
        historical_hot = arrays["hist_last_hot"] > 0.5
        for kind, mask in negative_kinds.items():
            for state, hist_mask in (("historically_hot", historical_hot),
                                     ("historically_cold", ~historical_hot)):
                counts[f"{kind}_{state}"] = int((predicted & mask & hist_mask).sum())
        row["report_false_alarm_breakdown"] = counts
        row["report_false_alarm_count"] = int(false_alarm.sum())
        row["per_node"] = [
            {
                "node_index": index,
                "predicted": int((predicted[:, index] & valid[:, index]).sum()),
                "true": int(positive[:, index].sum()),
                "report_hits": int(hits[:, index].sum()),
                "false_alarms": int(false_alarm[:, index].sum()),
                "wrong_start": int(timing_error[:, index].sum()),
                "negative_subgroups": {
                    name: int((predicted[:, index] & mask[:, index]).sum())
                    for name, mask in negative_kinds.items()
                },
            }
            for index in range(valid.shape[1]) if bool(valid[:, index].any())
        ]
        row["predicted_duration_q25_q50_q75"] = (
            np.quantile(arrays["predicted_duration"][predicted & valid], [.25, .5, .75]).tolist()
            if bool((predicted & valid).any()) else None
        )
        if sum(counts.values()) != int(false_alarm.sum()):
            raise AssertionError("False-alarm diagnostic groups must be disjoint and exhaustive")
        expected_precision = float(hits.sum()) / max(int((predicted & valid).sum()), 1)
        if not np.isclose(expected_precision, row["report_precision"]):
            raise ValueError("Event diagnostic targets differ from canonical report metrics")
        output["thresholds"].append(row)
    return output


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation"), default="validation")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[
        0.20, 0.30, 0.40, 0.50, 0.55, 0.60, 0.68, 0.75, 0.80, 0.85, 0.90, 0.95,
    ])
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(args.threads)
    device = _resolve_device(args.device)
    payload, manifest = load_shared_dataset(args.dataset_dir)
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    if checkpoint["metadata"]["dataset_manifest_sha256"] != _manifest_hash(
        args.dataset_dir / "dataset_manifest.json"
    ):
        raise ValueError("Checkpoint and dataset manifest hashes do not match")
    if int(manifest["event_min_windows"]) != 8:
        raise ValueError("This diagnostic requires the canonical 8-window event contract")
    loader = DataLoader(
        FactoryBaselineTensorDataset(payload, payload["split_indices"][args.split]),
        batch_size=args.batch_size, shuffle=False,
    )
    collected: dict[str, list[np.ndarray]] = {}
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            result = model(**_model_inputs(batch))
            values = {key: batch[key] for key in (
                "sample_index", "y_hot", "remain_mask", "occ_node_mask",
                "hist_last_hot", "event_will", "event_start",
            )}
            values.update(
                will_probability=result["event_will_logit"].sigmoid(),
                predicted_start=result["event_start_logit"].argmax(-1),
                predicted_duration=result["event_duration"],
            )
            for key, value in values.items():
                collected.setdefault(key, []).append(value.cpu().numpy())
    arrays = {key: np.concatenate(values) for key, values in collected.items()}
    chosen_threshold = float(checkpoint["metadata"]["event_report_threshold"])
    thresholds = sorted(set([*args.thresholds, chosen_threshold]))
    report = summarize_events(arrays, thresholds)
    attach_node_catalog(report, args.dataset_dir / "node_catalog.csv", manifest)
    report.update(
        split=args.split, test_evaluated=False,
        checkpoint=str(args.checkpoint.resolve()), epoch=checkpoint["epoch"],
        dataset_manifest_sha256=checkpoint["metadata"]["dataset_manifest_sha256"],
        sample_count=len(arrays["sample_index"]),
        saved_report_threshold=chosen_threshold,
        diagnostic_scope="Future labels and horizon length are retrospective diagnostic strata, "
                         "not deployable features or proposed alarm filters. No threshold is selected here.",
        ranking_scope="One existing will score is ranked in each retrospective subgroup. "
                      "Ongoing/upcoming are not separate predicted heads. Tie-aware AP is diagnostic, "
                      "not a replacement for the canonical checkpoint-selection metric.",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["groups"], indent=2), flush=True)
    print(json.dumps(report["ranking"], indent=2), flush=True)
    print("threshold P R F1 upcoming_R upcoming_probability_misses upcoming_timing_misses")
    for row in report["thresholds"]:
        print(row["threshold"], *[round(row[k], 4) for k in (
            "report_precision", "report_recall", "report_f1", "report_recall_upcoming",
        )], row["upcoming_probability_misses"], row["upcoming_timing_misses"], flush=True)
    print(f"Output: {args.output}", flush=True)


if __name__ == "__main__":
    main()
