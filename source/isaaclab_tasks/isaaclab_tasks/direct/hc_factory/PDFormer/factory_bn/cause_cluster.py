"""Cause-aligned station clusters from operational X (not occupancy y).

Same gates as ``tools/bn_agg/labels._process_root_cause``. Ids:
normal, transport_delay, material_shortage, blocked_downstream,
starved_upstream, queue_buildup, high_utilization.
"""

from __future__ import annotations

import numpy as np

CAUSE_ALIGNED_NAMES: tuple[str, ...] = (
    "normal",
    "transport_delay",
    "material_shortage",
    "blocked_downstream",
    "starved_upstream",
    "queue_buildup",
    "high_utilization",
)
CAUSE_NAME_TO_ID: dict[str, int] = {n: i for i, n in enumerate(CAUSE_ALIGNED_NAMES)}
N_CAUSE_CLUSTERS = len(CAUSE_ALIGNED_NAMES)
CLUSTER_PAD_ID = N_CAUSE_CLUSTERS  # 7: unknown / missing

# FEATURE_COLS indices (export_dataset.FEATURE_COLS)
_Q = 0
_WAIT = 1
_ACTIVE = 4
_BLOCK = 6
_STARVE = 7
_ROUTE = 13
_INBOUND = 14
_SHORT = 15
_BUF = 16
_TPM = 19

_HOT_SHORTAGE_PROP = 0.25
_HOT_INBOUND_S = 20.0
_HOT_ROUTE_S = 20.0
_HOT_QUEUE_CAUSE = 1.0
_HOT_WAIT_CAUSE = 20.0
_HOT_BLOCK_FRAC = 0.40
_HOT_BUF = 0.70
_HOT_ACTIVE = 0.8


def seed_cluster_ids(
    features: np.ndarray,
    *,
    window_size_s: float = 60.0,
) -> np.ndarray:
    """(T, N) int64 cluster ids from operational features."""
    feats = np.asarray(features, dtype=np.float32)
    if feats.ndim != 3:
        raise ValueError(f"features must be (T,N,F), got {feats.shape}")
    t_len, n_nodes, f = feats.shape
    wlen = max(float(window_size_s), 1.0)
    out = np.zeros((t_len, n_nodes), dtype=np.int64)
    short = feats[:, :, _SHORT] if f > _SHORT else np.zeros((t_len, n_nodes), dtype=np.float32)
    inbound = feats[:, :, _INBOUND] if f > _INBOUND else np.zeros((t_len, n_nodes), dtype=np.float32)
    route = feats[:, :, _ROUTE] if f > _ROUTE else np.zeros((t_len, n_nodes), dtype=np.float32)
    q = feats[:, :, _Q] if f > _Q else np.zeros((t_len, n_nodes), dtype=np.float32)
    wait = feats[:, :, _WAIT] if f > _WAIT else np.zeros((t_len, n_nodes), dtype=np.float32)
    buf = feats[:, :, _BUF] if f > _BUF else np.zeros((t_len, n_nodes), dtype=np.float32)
    blocked = feats[:, :, _BLOCK] if f > _BLOCK else np.zeros((t_len, n_nodes), dtype=np.float32)
    starved = feats[:, :, _STARVE] if f > _STARVE else np.zeros((t_len, n_nodes), dtype=np.float32)
    active = feats[:, :, _ACTIVE] if f > _ACTIVE else np.zeros((t_len, n_nodes), dtype=np.float32)

    out[:] = CAUSE_NAME_TO_ID["high_utilization"]
    out[active < _HOT_ACTIVE] = CAUSE_NAME_TO_ID["normal"]
    starve_m = starved > 0
    out[starve_m] = CAUSE_NAME_TO_ID["starved_upstream"]
    block_m = (blocked >= starved) & (blocked > _HOT_BLOCK_FRAC * wlen)
    out[block_m] = CAUSE_NAME_TO_ID["blocked_downstream"]
    queue_m = (q >= _HOT_QUEUE_CAUSE) | (wait >= _HOT_WAIT_CAUSE) | (buf >= _HOT_BUF)
    out[queue_m] = CAUSE_NAME_TO_ID["queue_buildup"]
    transport_m = (inbound >= _HOT_INBOUND_S) | (route >= _HOT_ROUTE_S)
    out[transport_m] = CAUSE_NAME_TO_ID["transport_delay"]
    short_m = short >= _HOT_SHORTAGE_PROP
    out[short_m] = CAUSE_NAME_TO_ID["material_shortage"]
    return out


def future_tpm_target(
    features: np.ndarray,
    *,
    t: int,
    horizon: int,
) -> np.ndarray:
    """(N,) 1 if a station is a TPM turning-point in ``[t, t+horizon)``."""
    feats = np.asarray(features, dtype=np.float32)
    n_nodes = int(feats.shape[1])
    if feats.shape[-1] <= _TPM:
        return np.zeros((n_nodes,), dtype=np.float32)
    end = min(int(t) + max(int(horizon), 1), int(feats.shape[0]))
    start = max(int(t), 0)
    if start >= end:
        return np.zeros((n_nodes,), dtype=np.float32)
    return (feats[start:end, :, _TPM] > 0.5).any(axis=0).astype(np.float32)


def hist_tpm_flag(features: np.ndarray, t: int) -> np.ndarray:
    """(N,) TPM flag at the last history window ``t-1`` (label_idx)."""
    feats = np.asarray(features, dtype=np.float32)
    n_nodes = int(feats.shape[1])
    idx = int(t)
    if feats.shape[-1] <= _TPM or idx < 0 or idx >= feats.shape[0]:
        return np.zeros((n_nodes,), dtype=np.float32)
    return (feats[idx, :, _TPM] > 0.5).astype(np.float32)
