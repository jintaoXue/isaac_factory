#!/usr/bin/env python3
"""Audit historical information availability without training or changing v6."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
import subprocess
import types

import numpy as np


REFERENCE = "20c40e230aedee6aef2429d352413fbcf0fa571a"
MAIN_PACKAGE = "source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/PDFormer/factory_bn"
SPLITS = ("train", "validation")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_reference(repo: Path) -> tuple[types.ModuleType, dict]:
    sources = {}
    for name in ("remain.py", "dataset.py", "model.py"):
        sources[name] = subprocess.check_output(
            ["git", "show", f"{REFERENCE}:{MAIN_PACKAGE}/{name}"], cwd=repo
        )
    reference = types.ModuleType("audited_precursor_reference")
    exec(compile(sources["remain.py"], f"{REFERENCE}:remain.py", "exec"), reference.__dict__)
    if (reference.PRECURSOR_LOOKBACK, reference.PRECURSOR_FAR_WINDOWS,
            reference.PRECURSOR_DIM) != (5, 30, 23):
        raise ValueError("Reference precursor contract differs from the audited source")
    return reference, {
        "commit": REFERENCE,
        "files": {f"{MAIN_PACKAGE}/{name}": hashlib.sha256(data).hexdigest()
                  for name, data in sources.items()},
    }


def history_summary(reference, features: np.ndarray, t: int, *, include_far: bool) -> np.ndarray:
    if features.ndim != 3 or features.shape[-1] != 27 or not 30 <= t <= len(features):
        raise ValueError("Expected 27-channel features and a complete 30-window history")
    hist_start = t - 30
    far = features[max(0, hist_start - 30):hist_start] if include_far and hist_start else None
    return reference.pack_precursor_features(features[hist_start:t], far)


def audit(dataset_dir: Path, repo: Path) -> dict:
    reference, provenance = load_reference(repo)
    manifest = json.loads((dataset_dir / "dataset_manifest.json").read_text())
    split = json.loads((dataset_dir / "split_manifest.json").read_text())
    if manifest["dataset_version"] != "factory_baseline_dataset_v6":
        raise ValueError("This audit is registered for the unchanged dense v6 bundle")
    if manifest["input_windows"] != 30 or manifest["window_size_s"] != 60:
        raise ValueError("Expected 30 one-minute encoder windows")
    if manifest["shared_bundle_alignment"]["status"] != "passed":
        raise ValueError("Baseline/common export alignment must already have passed")
    names = {row["group_id"]: row["main_episode_name"] for row in manifest["source_episodes"]}
    groups = defaultdict(list)
    with (dataset_dir / "model_sample_index.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row["split"] in SPLITS:
                groups[row["group_id"]].append(row)
    selected_indices = {name: [] for name in SPLITS}
    selected_groups = {name: set() for name in SPLITS}
    stats = {name: {"samples": 0, "far_windows_histogram": Counter(),
                    "samples_with_nonzero_far_summary": 0, "node_samples": 0,
                    "far_feature_nonzero_cells": np.zeros(5, dtype=np.int64),
                    "far_feature_sums": np.zeros(5, dtype=np.float64)} for name in SPLITS}
    with np.load(dataset_dir / "episodes.npz", allow_pickle=False) as bundle:
        source_nodes = bundle["resource_ids"].tolist()
        if set(source_nodes) != set(manifest["node_ids"]):
            raise ValueError("Node inventories differ")
        order = [source_nodes.index(node) for node in manifest["node_ids"]]
        for group, rows in groups.items():
            name = names[group]
            # Only train/validation feature arrays are loaded. No targets or test arrays.
            features = bundle[name + "_features"][:, order]
            windows = bundle[name + "_windows"]
            starts = bundle[name + "_window_start_s"]
            positions = {int(window): i for i, window in enumerate(windows)}
            for row in rows:
                split_name = row["split"]
                if group not in split[split_name]["group_ids"]:
                    raise ValueError("Sample/episode split identity mismatch")
                t = positions[int(row["anchor_window_index"])] + 1
                history_windows = json.loads(row["input_window_indices"])
                np.testing.assert_array_equal(history_windows, windows[t-30:t])
                np.testing.assert_allclose(starts[t-30:t], starts[t-1] - np.arange(29, -1, -1) * 60)
                np.testing.assert_allclose(
                    [float(row["anchor_time_s"]), float(row["first_future_start_s"])],
                    [starts[t-1], starts[t-1] + 60],
                )
                far_start = max(0, t - 60)
                if t > 30:
                    np.testing.assert_allclose(np.diff(starts[far_start:t]), 60)
                near_only = history_summary(reference, features, t, include_far=False)
                extended = history_summary(reference, features, t, include_far=True)
                if not np.isfinite(extended).all():
                    raise ValueError("Non-finite precursor input")
                np.testing.assert_array_equal(extended[:, :18], near_only[:, :18])
                np.testing.assert_array_equal(near_only[:, 18:], 0)
                # Removing every future observation must preserve the entire precursor.
                np.testing.assert_array_equal(
                    extended, history_summary(reference, features[:t], t, include_far=True)
                )
                far = extended[:, 18:]
                record = stats[split_name]
                record["samples"] += 1
                record["far_windows_histogram"][min(30, t - 30)] += 1
                record["samples_with_nonzero_far_summary"] += int(np.any(far != 0))
                record["node_samples"] += len(far)
                record["far_feature_nonzero_cells"] += (far != 0).sum(axis=0)
                record["far_feature_sums"] += far.sum(axis=0, dtype=np.float64)
                selected_indices[split_name].append(int(row["sample_index"]))
                selected_groups[split_name].add(group)
    for name, record in stats.items():
        if sorted(selected_indices[name]) != sorted(split[name]["sample_indices"]):
            raise ValueError(f"Incomplete/duplicate {name} samples")
        if selected_groups[name] != set(split[name]["group_ids"]):
            raise ValueError(f"Incomplete {name} episode coverage")
        if record["samples"] != manifest["sample_counts"][name]:
            raise ValueError("Manifest sample count mismatch")
        record["episodes"] = len(selected_groups[name])
        record["far_windows_histogram"] = dict(sorted(record["far_windows_histogram"].items()))
        record["far_feature_nonzero_cells"] = record["far_feature_nonzero_cells"].tolist()
        record["far_feature_mean_all_nodes"] = (record.pop("far_feature_sums") / record["node_samples"]).tolist()
    return {
        "status": "passed_input_availability_audit_not_model_comparison",
        "source_reference": provenance,
        "dataset_files_sha256": {name: sha256(dataset_dir / name) for name in (
            "dataset_manifest.json", "split_manifest.json", "model_sample_index.csv", "episodes.npz")},
        "dataset_version": manifest["dataset_version"],
        "evaluation_contract": manifest["evaluation_contract"],
        "splits_examined": list(SPLITS), "test_arrays_loaded": False,
        "labels_used": False, "training_started": False, "dataset_modified": False,
        "observation_budget": {
            "anchor": "t is first future window; half-open array slices",
            "near_encoder": "features[t-30:t]",
            "near_summary": "last 5 encoder windows; 12 last, 5 mean, 1 queue delta",
            "far_summary": "features[max(0,t-60):t-30]; 4 means and 1 maximum",
            "far_feature_order": ["queue_mean", "blocked_mean", "inbound_wait_mean",
                                  "material_shortage_mean", "queue_max"],
            "far_feature_scales": [5, 60, 60, 1, 5],
            "far_mean_clipping": [-3, 3], "far_queue_max_clipping": [0, 3],
            "precursor_dim_before_main_neighbor_queue": 23,
            "far_summary_dim": 5, "missing_far_policy": "zero; no past exists before episode start",
        },
        "checks": ["sample/split/node/anchor identity", "contiguous historical windows",
                   "near-only and extended first 18 dimensions exactly equal",
                   "near-only last 5 dimensions exactly zero", "future truncation invariant"],
        "statistics": stats,
        "limitations": [
            "This is the baseline 208-episode bundle, not the unverified main 204-episode bundle.",
            "Availability/nonzero input does not establish predictive utility or checkpoint use.",
            "All-node summaries include inapplicable nodes; no event recall or AP is computed.",
            "Whole-episode smoothed historical hot remains a separate unresolved common issue.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip()).resolve()
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=repo, text=True).strip()
    if branch != "dev_xwt" or repo.name != "BSTAN_isaac_factory":
        raise ValueError("Run only in the existing BSTAN_isaac_factory/dev_xwt server repository")
    dataset_dir, output = args.dataset_dir.resolve(), args.output.resolve()
    if not dataset_dir.is_dir() or not dataset_dir.is_relative_to(repo):
        raise ValueError("Dataset must be an existing directory within BSTAN_isaac_factory")
    if output.parent != dataset_dir or output.exists():
        raise ValueError("Use a new result filename directly in the existing dataset directory")
    result = audit(dataset_dir, repo)
    result["audit_code_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    with output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"output": str(output), "status": result["status"],
                      "statistics": result["statistics"]}, indent=2))


if __name__ == "__main__":
    main()
