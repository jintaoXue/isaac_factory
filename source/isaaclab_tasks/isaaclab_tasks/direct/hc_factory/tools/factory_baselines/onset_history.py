"""Onset gate derived exclusively from the existing thirty observed encoder windows."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import torch

from factory_bn_shared.bundle import file_hash
from factory_bn_shared.remain import ops_hot_mask


def observed_history_hot(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Invert frozen normalization and smooth only the supplied observed history."""
    if x.ndim != 3 or x.shape[0] != 30 or x.shape[-1] != 27:
        raise ValueError("Onset history requires exactly thirty existing 27-channel windows")
    if mean.shape != (21,) or std.shape != (21,):
        raise ValueError("Expected frozen normalization for twenty-one continuous channels")
    if not np.isfinite(x).all() or not np.isfinite(mean).all() or not (np.isfinite(std) & (std > 0)).all():
        raise ValueError("Non-finite history or invalid normalization")
    raw = np.asarray(x, dtype=np.float32).copy()
    raw[..., :21] = raw[..., :21] * std + mean
    if not np.isfinite(raw).all():
        raise ValueError("Non-finite reconstructed operational history")
    return ops_hot_mask(raw, window_size_s=60., min_hot_windows=8, gap_windows=1)[-1].astype(np.float32)


def attach_onset_history(
    payload: dict, manifest: dict, dataset_dir: Path, enabled: bool,
    splits: tuple[str, ...], expected_contract: dict | None = None,
) -> tuple[dict, dict | None]:
    """Create a derived input in memory without changing labels, decoder or frozen files."""
    if not enabled:
        if expected_contract is not None:
            raise ValueError("Checkpoint expects observed onset history but joint reporting is disabled")
        return payload, None
    if not splits or len(set(splits)) != len(splits) or set(splits) - {"train", "validation", "test"}:
        raise ValueError("Invalid onset-history split selection")
    if manifest["input_windows"] != 30 or manifest["window_size_s"] != 60:
        raise ValueError("Onset history requires the fixed thirty-minute input contract")
    contract = {
        "version": "factory_baseline_observed_onset_history_v1",
        "field": "event_history_hot", "encoder_windows": 30, "extra_history_windows": 0,
        "source": "existing_normalized_x_inverted_with_frozen_training_normalization",
        "rule": "ops_hot_mask_on_thirty_observed_windows_only_then_last",
        "window_size_s": 60., "min_hot_windows": 8, "gap_windows": 1,
        "legacy_hist_last_hot_used_as_model_input": False,
        "dataset_manifest_sha256": file_hash(dataset_dir / "dataset_manifest.json"),
        "normalization_sha256": file_hash(dataset_dir / "normalization.json"),
        "builder_source_sha256": file_hash(Path(__file__)),
        "operational_hot_source_sha256": file_hash(Path(inspect.getfile(ops_hot_mask))),
    }
    if expected_contract is not None and expected_contract != contract:
        raise ValueError("Checkpoint observed-onset-history contract differs")
    norm = json.loads((dataset_dir / "normalization.json").read_text())
    mean, std = (np.asarray(norm[k], dtype=np.float32) for k in ("feature_mean", "feature_std"))
    expected = [int(i) for split in splits for i in payload["split_indices"][split]]
    if len(set(expected)) != len(expected):
        raise ValueError("Duplicated onset-history sample indices")
    result = torch.zeros(payload["x"].shape[0], payload["x"].shape[2])
    valid = torch.zeros(len(result), dtype=torch.bool)
    for i in expected:
        if not 0 <= i < len(result):
            raise ValueError("Onset-history index outside frozen payload")
        result[i] = torch.from_numpy(observed_history_hot(payload["x"][i].numpy(), mean, std))
        result[i] *= payload["node_mask"][i]
        valid[i] = True
    return {**payload, "event_history_hot": result, "event_history_hot_valid": valid}, contract
