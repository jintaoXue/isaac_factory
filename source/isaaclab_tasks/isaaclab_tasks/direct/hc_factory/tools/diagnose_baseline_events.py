#!/usr/bin/env python3
"""Decompose validation event misses without selecting on or reading test targets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from factory_baselines.dataset import FactoryBaselineTensorDataset, load_shared_dataset
from factory_baselines.torch_trainer import (
    _manifest_hash,
    _model_inputs,
    _resolve_device,
    load_checkpoint,
)
from factory_bn_shared.remain import station_report_metrics


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
    output = {"groups": {}, "thresholds": []}
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
        output["thresholds"].append(row)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[
        0.20, 0.30, 0.40, 0.50, 0.55, 0.60, 0.68, 0.75, 0.80, 0.85, 0.90, 0.95,
    ])
    args = parser.parse_args()
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
        FactoryBaselineTensorDataset(payload, payload["split_indices"]["validation"]),
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
    report = summarize_events(arrays, args.thresholds)
    report.update(
        split="validation", test_evaluated=False,
        checkpoint=str(args.checkpoint.resolve()), epoch=checkpoint["epoch"],
        dataset_manifest_sha256=checkpoint["metadata"]["dataset_manifest_sha256"],
        sample_count=len(arrays["sample_index"]),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["groups"], indent=2), flush=True)
    print("threshold P R F1 upcoming_R upcoming_probability_misses upcoming_timing_misses")
    for row in report["thresholds"]:
        print(row["threshold"], *[round(row[k], 4) for k in (
            "report_precision", "report_recall", "report_f1", "report_recall_upcoming",
        )], row["upcoming_probability_misses"], row["upcoming_timing_misses"], flush=True)
    print(f"Output: {args.output}", flush=True)


if __name__ == "__main__":
    main()
