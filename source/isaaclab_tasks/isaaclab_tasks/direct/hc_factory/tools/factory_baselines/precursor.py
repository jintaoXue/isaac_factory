"""Explicit past-only onset summaries for controlled B4/B5 input ablations."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from factory_bn_shared.bundle import file_hash


PRECURSOR_MODES = ("none", "near", "near_far")
PRECURSOR_DIM = 23
_LAST = (0, 1, 2, 3, 6, 7, 11, 12, 13, 14, 15, 18)
_MEAN = (0, 6, 7, 14, 15)
_FAR = (0, 6, 14, 15)
_SCALE = {0: 5., 1: 60., 2: 1., 3: 2., 6: 60., 7: 60.,
          11: 1., 12: 1., 13: 60., 14: 60., 15: 1., 18: 1.}


def pack_history_precursor(features: np.ndarray, t: int, mode: str) -> np.ndarray:
    """t is the first future index; near mode has exactly zero far fields."""
    if mode not in {"near", "near_far"}:
        raise ValueError("A precursor requires near or near_far mode")
    if features.ndim != 3 or features.shape[-1] != 27 or not 30 <= t <= len(features):
        raise ValueError("Expected complete 30-window history with 27 channels")
    tail = features[t-5:t]
    last, mean = tail[-1], tail.mean(axis=0)
    col = lambda a, i: np.clip(a[:, i] / _SCALE[i], -3., 3.)
    values = [col(last, i) for i in _LAST] + [col(mean, i) for i in _MEAN]
    values.append(np.clip((last[:, 0] - tail[0, :, 0]) / 5., -3., 3.))
    if mode == "near_far" and t > 30:
        far = features[max(0, t-60):t-30]
        values.extend(col(far.mean(axis=0), i) for i in _FAR)
        values.append(np.clip(far[:, :, 0].max(axis=0) / 5., 0., 3.))
    else:
        values.extend(np.zeros(len(last), dtype=np.float32) for _ in range(5))
    result = np.stack(values, axis=-1).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Non-finite precursor input")
    return result


def attach_precursor(payload: dict, manifest: dict, dataset_dir: Path, mode: str,
                     splits: tuple[str, ...], expected_contract: dict | None = None) -> tuple[dict, dict | None]:
    """Build only requested splits in memory; never replace the v6 dataset files."""
    if mode == "none":
        if expected_contract is not None:
            raise ValueError("Checkpoint expects precursor features but mode is none")
        return payload, None
    if mode not in PRECURSOR_MODES or not splits or set(splits) - {"train", "validation", "test"}:
        raise ValueError("Invalid precursor mode/splits")
    if manifest["input_windows"] != 30 or manifest["window_size_s"] != 60:
        raise ValueError("Precursor contract requires 30 one-minute history windows")
    if payload["x"].shape[1] != 30 or payload["x"].shape[-1] != 27:
        raise ValueError("Unexpected baseline encoder tensor shape")
    bundle_hash = file_hash(dataset_dir / "episodes.npz")
    if manifest["shared_bundle_alignment"]["bundle_sha256"] != bundle_hash:
        raise ValueError("The common export changed after v6 alignment")
    contract = {
        "version": "factory_baseline_precursor_v1", "mode": mode,
        "dimension": PRECURSOR_DIM, "near_summary_windows": 5,
        "encoder_windows": 30, "extra_history_windows_max": 30 if mode == "near_far" else 0,
        "far_field_policy": "observed_mean_max" if mode == "near_far" else "five_zeros",
        "dataset_manifest_sha256": file_hash(dataset_dir / "dataset_manifest.json"),
        "bundle_sha256": bundle_hash,
        "sample_index_sha256": file_hash(dataset_dir / "model_sample_index.csv"),
        "split_manifest_sha256": file_hash(dataset_dir / "split_manifest.json"),
        "normalization_sha256": file_hash(dataset_dir / "normalization.json"),
        "near_source": "baseline_x_inverted_with_frozen_training_normalization",
        "feature_source_sha256": file_hash(Path(__file__)),
        "reference_feature_definition": "dev_tyx@20c40e2:remain.py pack_precursor_features; no decoder",
    }
    if expected_contract is not None and expected_contract != contract:
        raise ValueError("Checkpoint precursor contract differs from current input construction")
    normalization = json.loads((dataset_dir / "normalization.json").read_text())
    mean = np.asarray(normalization["feature_mean"], dtype=np.float32)
    std = np.asarray(normalization["feature_std"], dtype=np.float32)
    if mean.shape != (21,) or std.shape != (21,) or not np.isfinite(mean).all() or not (np.isfinite(std) & (std > 0)).all():
        raise ValueError("Expected the frozen 21-channel training normalization")
    names = {r["group_id"]: r["main_episode_name"] for r in manifest["source_episodes"]}
    expected = {int(i) for split in splits for i in payload["split_indices"][split]}
    groups = defaultdict(list)
    with (dataset_dir / "model_sample_index.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row["split"] in splits:
                groups[row["group_id"]].append(row)
    result = torch.zeros((len(payload["x"]), len(manifest["node_ids"]), PRECURSOR_DIM))
    valid = torch.zeros(len(result), dtype=torch.bool)
    with np.load(dataset_dir / "episodes.npz", allow_pickle=False) as bundle:
        nodes = bundle["resource_ids"].tolist()
        if set(nodes) != set(manifest["node_ids"]):
            raise ValueError("Precursor and baseline node inventories differ")
        order = [nodes.index(n) for n in manifest["node_ids"]]
        for group, rows in groups.items():
            name = names[group]
            features = bundle[name + "_features"][:, order]
            windows = bundle[name + "_windows"]
            starts = bundle[name + "_window_start_s"]
            for row in rows:
                i = int(row["sample_index"])
                if i not in expected or valid[i]:
                    raise ValueError("Duplicate or misplaced precursor sample")
                t = int(payload["target_start_position"][i])
                np.testing.assert_array_equal(json.loads(row["input_window_indices"]), windows[t-30:t])
                np.testing.assert_allclose(float(row["anchor_time_s"]), starts[t-1])
                first = max(0, t-60) if mode == "near_far" else t-30
                np.testing.assert_allclose(np.diff(starts[first:t]), 60.)
                # The near arm is strictly a deterministic transform of existing encoder x.
                # Missing observations retain the encoder's encoded value, not a new raw cue.
                near = payload["x"][i].numpy().copy()
                near[..., :21] = near[..., :21] * std + mean
                observed = payload["observation_mask"][i].numpy().astype(bool)
                np.testing.assert_allclose(near[observed], features[t-30:t][observed], rtol=1e-5, atol=5e-4)
                history = np.concatenate((features[first:t-30], near), axis=0)
                result[i] = torch.from_numpy(pack_history_precursor(history, len(history), mode))
                result[i] *= payload["node_mask"][i, :, None]
                valid[i] = True
    if set(torch.where(valid)[0].tolist()) != expected:
        raise ValueError("Incomplete precursor sample coverage")
    return {**payload, "event_precursor": result, "event_precursor_valid": valid}, contract
