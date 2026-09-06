#!/usr/bin/env python3
"""Compare validation inputs and targets with the main bundle, without scoring models."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from audit_baseline_episode_split import file_hash
from factory_baselines.dataset import FactoryBaselineTensorDataset, load_shared_dataset


def record_difference(stats, key, actual, expected, sample_index, atol=1e-4):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape:
        raise ValueError(f"{key}: shape mismatch {actual.shape} != {expected.shape}")
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError(f"{key}: non-finite comparison values")
    different = ~np.isclose(actual, expected, rtol=1e-5, atol=atol, equal_nan=False)
    row = stats.setdefault(key, {"mismatched_samples": 0, "mismatched_cells": 0,
                                 "max_absolute_difference": 0.0, "example_samples": []})
    if different.any():
        row["mismatched_samples"] += 1
        row["mismatched_cells"] += int(different.sum())
        delta = np.abs(actual.astype(float) - expected.astype(float))
        row["max_absolute_difference"] = max(row["max_absolute_difference"], float(delta.max()))
        if len(row["example_samples"]) < 8:
            row["example_samples"].append(int(sample_index))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_dir", type=Path, required=True)
    parser.add_argument("--main_bundle", type=Path, required=True)
    parser.add_argument("--main_checkpoint", type=Path, required=True)
    parser.add_argument("--pdformer_root", type=Path, required=True)
    parser.add_argument("--split_audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(4)
    split_audit = json.loads(args.split_audit.read_text(encoding="utf-8"))
    if not split_audit["episode_split_match"]:
        raise ValueError("Resolve the episode split mismatch before comparing samples")
    for name, path in (
        ("baseline_manifest", args.dataset_dir / "dataset_manifest.json"),
        ("main_bundle", args.main_bundle / "episodes.npz"),
        ("main_checkpoint", args.main_checkpoint),
    ):
        if file_hash(path) != split_audit["provenance"][name]["sha256"]:
            raise ValueError(f"{name} changed since the episode split audit")
    payload, manifest = load_shared_dataset(args.dataset_dir)
    cause_source = manifest["cause_label_source"]
    if cause_source["kind"] != "frozen_main_bundle":
        raise ValueError("Expected explicitly frozen main cause labels")
    for name, filename in (("meta", "meta.json"), ("episodes", "episodes.npz")):
        if file_hash(args.main_bundle / filename) != cause_source["files"][name]["sha256"]:
            raise ValueError("Cause-label source differs from the main bundle under audit")
    # Load the supplied main implementation separately from baseline imports.
    reference_path = args.pdformer_root / "factory_bn/remain.py"
    spec = importlib.util.spec_from_file_location("main_remain_reference", reference_path)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    config = torch.load(args.main_checkpoint, map_location="cpu", weights_only=False)["config"]
    if config["train_mode"] != "unsupervised" or not config["remain_to_jobs_done"]:
        raise ValueError("Expected main operational-occupancy, remaining-jobs experiment")
    history = int(config["input_window"])
    horizon = int(config["max_remain_windows"])
    k_occ = int(config["occupancy_horizon_windows"])
    min_hot = int(config["hot_min_windows"])
    gap_hot = int(config["hot_gap_windows"])
    event_min = int(config["event_min_windows"])
    window_size = float(config["window_size_s"])
    normalization = json.loads((args.dataset_dir / "normalization.json").read_text())
    mean = np.asarray(normalization["feature_mean"], dtype=np.float32)
    std = np.asarray(normalization["feature_std"], dtype=np.float32)
    with (args.dataset_dir / "model_sample_index.csv").open(newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row["split"] == "validation"]
    if (len(rows) != len(payload["split_indices"]["validation"])
        or {int(row["sample_index"]) for row in rows} != set(payload["split_indices"]["validation"].tolist())):
        raise ValueError("CSV/tensor validation sample indices differ")
    by_group = defaultdict(list)
    for row in rows:
        by_group[row["group_id"]].append(row)
    names = {group: name for name, group in split_audit["main_episode_mapping"].items()}
    dataset = FactoryBaselineTensorDataset(payload)
    stats, position_mismatches = {}, []
    with np.load(args.main_bundle / "episodes.npz", allow_pickle=False) as bundle:
        main_nodes = bundle["resource_ids"].tolist()
        if set(main_nodes) != set(manifest["node_ids"]):
            raise ValueError("Node catalogs differ")
        node_order = [main_nodes.index(node) for node in manifest["node_ids"]]
        for group, group_rows in by_group.items():
            name = names[group]
            features = reference.ensure_labor_saturated_feature(bundle[name + "_features"])
            features = features[:, node_order]
            scores = bundle[name + "_scores"][:, node_order]
            starts, windows = bundle[name + "_window_start_s"], bundle[name + "_windows"]
            jobs, total = bundle[name + "_jobs_remaining"], float(bundle[name + "_jobs_total"][0])
            done = reference.first_done_index(jobs)
            hot = reference.ops_hot_mask(features, window_size_s=window_size,
                                         min_hot_windows=min_hot, gap_windows=gap_hot)
            occ_mask = reference.occupancy_node_mask(features)
            main_positions = {t for t in range(history, min(len(features), done)) if jobs[t - 1] > 0}
            positions = {int(payload["target_start_position"][int(row["sample_index"])]) for row in group_rows}
            if positions != main_positions:
                position_mismatches.append({"group_id": group,
                    "baseline_only": sorted(positions - main_positions),
                    "main_only": sorted(main_positions - positions)})
            for row in group_rows:
                index = int(row["sample_index"])
                sample = dataset[index]
                t = int(sample["target_start_position"])
                target_score, target_hot, remain_mask, remain_len = reference.pack_remain_target(
                    scores, hot, t=t, done_ti=done, max_remain_windows=horizon,
                    occupancy_horizon_windows=k_occ,
                )
                will, start, duration = reference.node_event_targets(
                    target_hot, min_windows=event_min, remain_mask=remain_mask, occ_node_mask=occ_mask,
                )
                expected = dict(y_score=target_score, y_hot=target_hot,
                                y_cause=bundle[name + "_cause"][t-1],
                                remain_mask=remain_mask, target_remain_len=remain_len,
                                occ_node_mask=occ_mask, hist_last_hot=hot[t-1], event_will=will,
                                event_start=start, event_duration=duration, jobs_remaining=jobs[t-1], jobs_total=total)
                for key, value in expected.items():
                    record_difference(stats, key, sample[key].numpy(), value, index)
                raw_x = sample["x"].numpy().copy()
                raw_x[..., :len(mean)] = raw_x[..., :len(mean)] * std + mean
                observed = sample["observation_mask"].numpy().astype(bool)
                record_difference(stats, "raw_x_observed", raw_x[observed], features[t-history:t][observed], index)
                record_difference(stats, "anchor_time_s", float(row["anchor_time_s"]), starts[t-1], index)
                record_difference(stats, "first_future_start_s", float(row["first_future_start_s"]), starts[t], index)
                record_difference(stats, "input_window_indices", json.loads(row["input_window_indices"]), windows[t-history:t], index)
    result = {
        "scope": "validation inputs, sample anchors, masks and event targets; no test scoring",
        "test_evaluated": False, "sample_count": len(rows), "episode_count": len(by_group),
        "comparison_match": not position_mismatches and not any(row["mismatched_samples"] for row in stats.values()),
        "fields": stats, "position_mismatches": position_mismatches,
        "split_audit_sha256": file_hash(args.split_audit),
        "main_remain_source_sha256": file_hash(reference_path),
        "input_tolerance": {"absolute": 1e-4, "relative": 1e-5},
        "input_scope": "Raw X compared on observed history nodes after undoing baseline continuous-feature scaling. "
        "This does not compare architecture-specific scaling, graph construction or padded-node encodings.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("validation samples:", len(rows), "position mismatches:", len(position_mismatches))
    for key, row in stats.items():
        print(key, row["mismatched_samples"], row["mismatched_cells"], row["max_absolute_difference"])
    print("comparison_match:", result["comparison_match"], "output:", args.output)
    if not result["comparison_match"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
