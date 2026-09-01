"""Remaining-jobs length plus fixed-H occupancy for A.1.

Causal input at time t is how many jobs are still unfinished. Occupancy y is
the next H windows (short-order default H=15 min). ``remain_len`` is still
windows until remaining jobs hit zero. Live inference predicts both.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# FEATURE_COLS indices (export_dataset.FEATURE_COLS)
_Q_IDX = 0  # queue_length_s
_ACTIVE_IDX = 4  # active_pct_s
_BLOCK_IDX = 6  # blocked_time_s
_STARVE_IDX = 7  # starved_time_s
_UNAVAIL_IDX = 9  # unavailable_pct_s
_UP_IDX = 11  # upstream_blocked_ratio_s
_DOWN_IDX = 12  # downstream_starved_ratio_s
_ROUTE_IDX = 13  # route_delay_s
_INBOUND_IDX = 14  # inbound_wait_s
_SHORT_IDX = 15  # material_shortage_propagation_s
_DIST_IDX = 18  # disturbance_active_s — environment X, not occupancy y
_TP_IDX = 19  # is_turning_point
# FEATURE_COLS (21) + type one-hot; RESOURCE_TYPES = machine, gantry, human,
# transport_robot, buffer
_TYPE_MACHINE_IDX = 21
_TYPE_GANTRY_IDX = 22
_TYPE_HUMAN_IDX = 23
_TYPE_AGV_IDX = 24
OCC_SUPERVISE_TYPE_INDICES = (_TYPE_MACHINE_IDX, _TYPE_GANTRY_IDX, _TYPE_AGV_IDX)
# Appended after type one-hot so occupancy type indices stay 21–25.
_LABOR_FEAT_IDX = 26

# Future-X recon: occupancy / delay / stall columns only. TPM + type one-hot = 0.
_OPS_RECON_HIGH = {
    0: 2.0,  # queue
    2: 2.0,  # occupancy_ratio
    6: 2.0,  # blocked
    13: 2.0,  # route_delay
    14: 2.0,  # inbound_wait
    1: 1.0,  # avg_waiting
    3: 1.0,  # queue_growth
    4: 1.2,  # active_pct
    7: 1.5,  # starved
    8: 1.0,  # stop_time
    9: 1.5,  # unavailable (carrier STOP)
    11: 1.0,  # upstream_blocked
    12: 1.0,  # downstream_starved
    15: 1.5,  # material_shortage
    26: 1.0,  # labor_saturated
}
_OPS_RECON_ZERO = (19, 20, 21, 22, 23, 24, 25)


def ops_recon_channel_weight(feature_dim: int, floor: float = 0.05) -> np.ndarray:
    """Per-channel SmoothL1 weights for unsupervised future-X recon."""
    dim = max(int(feature_dim), 1)
    w = np.full((dim,), float(floor), dtype=np.float32)
    for idx in _OPS_RECON_ZERO:
        if idx < dim:
            w[idx] = 0.0
    for idx, val in _OPS_RECON_HIGH.items():
        if idx < dim:
            w[idx] = float(val)
    return w

# Keep in sync with tools/bn_agg/constants.py
_HOT_SCORE = 0.55
_HOT_SHORTAGE_PROP = 0.25
_HOT_SHORTAGE_STARVE_FRAC = 0.50
_HOT_INBOUND_S = 20.0
_HOT_ROUTE_S = 20.0
_HOT_INBOUND_STARVE_FRAC = 0.30
_HOT_BLOCK_FRAC = 0.40
_HOT_QUEUE_PILEUP = 2.0
_HOT_COUPLED_STALL_FRAC = 0.40
_HOT_COUPLED_UP = 0.15
_HOT_COUPLED_DOWN = 0.15
_HOT_DOWNTIME_UNAVAIL = 0.50
_HOT_OPERATOR_STALL_FRAC = 0.40
_HOT_LABOR_ACTIVE = 0.80
_HOT_HUMAN_PRESENT_EPS = 0.05
_HOT_GAP_WINDOWS = 1
# Main-task positives: stalls long enough to warn on (≈8 min, I=1.0 events are 6–15).
_HOT_MIN_WINDOWS = 8
_HOT_AGV_MIN_WINDOWS = 8


def _type_allow(feats: np.ndarray, idx: int) -> np.ndarray:
    if feats.shape[2] <= idx:
        return np.zeros(feats.shape[:2], dtype=bool)
    return feats[:, :, idx] > 0.5


def smooth_occupancy_runs(
    hot: np.ndarray,
    *,
    gap_windows: int = _HOT_GAP_WINDOWS,
    min_windows: int = _HOT_MIN_WINDOWS,
    min_windows_by_node: np.ndarray | None = None,
) -> np.ndarray:
    """Fill ≤gap cold holes, then drop occupancy runs shorter than min_windows."""
    grid = np.asarray(hot, dtype=np.float32).copy()
    if grid.ndim != 2:
        raise ValueError(f"hot must be (T, N), got {grid.shape}")
    t_len, n_nodes = grid.shape
    gap = max(int(gap_windows), 0)
    min_default = max(int(min_windows), 1)
    per_node = None
    if min_windows_by_node is not None:
        per_node = np.asarray(min_windows_by_node, dtype=np.int32).reshape(-1)
        if per_node.size != n_nodes:
            raise ValueError(
                f"min_windows_by_node length {per_node.size} != N={n_nodes}"
            )
    if t_len == 0 or (gap == 0 and min_default <= 1 and per_node is None):
        return grid
    for n in range(n_nodes):
        col = grid[:, n]
        min_w = int(per_node[n]) if per_node is not None else min_default
        min_w = max(min_w, 1)
        if min_w > 1:
            i = 0
            while i < t_len:
                if col[i] < 0.5:
                    i += 1
                    continue
                j = i + 1
                while j < t_len and col[j] >= 0.5:
                    j += 1
                if (j - i) < min_w:
                    col[i:j] = 0.0
                i = j
        if gap > 0:
            i = 0
            while i < t_len:
                if col[i] >= 0.5:
                    i += 1
                    continue
                j = i
                while j < t_len and col[j] < 0.5:
                    j += 1
                if i > 0 and j < t_len and (j - i) <= gap:
                    col[i:j] = 1.0
                i = j
    return grid


def labor_saturated_mask(features: np.ndarray) -> np.ndarray:
    """(T, 1) bool: every present on-duty human is busy this window."""
    feats = np.asarray(features, dtype=np.float32)
    if feats.ndim != 3:
        raise ValueError(f"features must be (T,N,F), got {feats.shape}")
    t_len, _n_nodes, f = feats.shape
    sat = np.zeros((t_len, 1), dtype=bool)
    if f <= _TYPE_HUMAN_IDX or f <= _ACTIVE_IDX or f <= _UNAVAIL_IDX:
        return sat
    human = _type_allow(feats, _TYPE_HUMAN_IDX)
    present = human.any(axis=0) & (
        (feats[:, :, _ACTIVE_IDX].max(axis=0) > _HOT_HUMAN_PRESENT_EPS)
        | (feats[:, :, _UNAVAIL_IDX].max(axis=0) > _HOT_HUMAN_PRESENT_EPS)
    )
    on_duty = present[np.newaxis, :] & (feats[:, :, _UNAVAIL_IDX] < _HOT_DOWNTIME_UNAVAIL)
    n_on = on_duty.sum(axis=1, keepdims=True)
    busy_ok = np.where(on_duty, feats[:, :, _ACTIVE_IDX] >= _HOT_LABOR_ACTIVE, True)
    return busy_ok.all(axis=1, keepdims=True) & (n_on >= 1)


def ensure_labor_saturated_feature(features: np.ndarray) -> np.ndarray:
    """Append / refresh column 26: labor saturation on machine nodes only.

    Unused human slots (type one-hot 0) are ignored. Idle machines in a
    saturated window still get 1 — that is context X, not occupancy y.
    """
    feats = np.asarray(features, dtype=np.float32)
    if feats.ndim != 3:
        raise ValueError(f"features must be (T,N,F), got {feats.shape}")
    t_len, n_nodes, f = feats.shape
    sat = labor_saturated_mask(feats)
    machine = (
        _type_allow(feats, _TYPE_MACHINE_IDX)
        if f > _TYPE_MACHINE_IDX
        else np.ones((t_len, n_nodes), dtype=bool)
    )
    col = (sat & machine).astype(np.float32)
    if f > _LABOR_FEAT_IDX:
        out = feats.copy()
        out[:, :, _LABOR_FEAT_IDX] = col
        return out
    if f == _LABOR_FEAT_IDX:
        return np.concatenate([feats, col[:, :, np.newaxis]], axis=-1)
    pad = np.zeros((t_len, n_nodes, _LABOR_FEAT_IDX - f), dtype=np.float32)
    return np.concatenate([feats, pad, col[:, :, np.newaxis]], axis=-1)


def _station_type_masks(
    feats: np.ndarray,
    machine_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Return (machine, carrier, have_types) bool masks, shape (T, N)."""
    t_len, n_nodes, f = feats.shape
    have_types = f >= _TYPE_MACHINE_IDX + 5
    if machine_mask is None:
        if have_types:
            machine = _type_allow(feats, _TYPE_MACHINE_IDX)
        else:
            machine = np.ones((t_len, n_nodes), dtype=bool)
    else:
        machine = np.asarray(machine_mask, dtype=bool)
        if machine.ndim == 1:
            machine = np.broadcast_to(machine, (t_len, n_nodes))
    if have_types:
        carrier = _type_allow(feats, _TYPE_GANTRY_IDX) | _type_allow(feats, _TYPE_AGV_IDX)
    else:
        carrier = np.zeros((t_len, n_nodes), dtype=bool)
    return machine, carrier, have_types


def ops_occupancy_raw(
    features: np.ndarray,
    *,
    window_size_s: float = 60.0,
    machine_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Process occupancy from operational X only (no score, no TPM flags)."""
    feats = np.asarray(features, dtype=np.float32)
    if feats.ndim != 3:
        raise ValueError(f"features must be (T,N,F), got {feats.shape}")
    w = max(float(window_size_s), 1e-9)
    t_len, n_nodes, f = feats.shape
    machine, carrier, have_types = _station_type_masks(feats, machine_mask)
    gantry = _type_allow(feats, _TYPE_GANTRY_IDX) if have_types else np.zeros((t_len, n_nodes), dtype=bool)
    agv = _type_allow(feats, _TYPE_AGV_IDX) if have_types else np.zeros((t_len, n_nodes), dtype=bool)
    hot = np.zeros((t_len, n_nodes), dtype=bool)
    if f > _STARVE_IDX:
        stall = feats[:, :, _BLOCK_IDX] + feats[:, :, _STARVE_IDX]
        q = feats[:, :, _Q_IDX]
        hot = hot | (((q >= _HOT_QUEUE_PILEUP - 1e-9) & (stall > 0.0)) & machine)
        hot = hot | (
            ((feats[:, :, _BLOCK_IDX] >= _HOT_BLOCK_FRAC * w) & (q >= 1.0 - 1e-9)) & machine
        )
        hot = hot | (
            (
                (stall >= _HOT_COUPLED_STALL_FRAC * w)
                & (
                    (feats[:, :, _UP_IDX] >= _HOT_COUPLED_UP)
                    | (feats[:, :, _DOWN_IDX] >= _HOT_COUPLED_DOWN)
                    | (feats[:, :, _INBOUND_IDX] >= _HOT_INBOUND_S)
                    | (feats[:, :, _ROUTE_IDX] >= _HOT_ROUTE_S)
                    | (feats[:, :, _SHORT_IDX] >= _HOT_SHORTAGE_PROP)
                    | (q >= 1.0 - 1e-9)
                )
            )
            & machine
        )
    if f > _SHORT_IDX:
        hot = hot | (
            (
                (feats[:, :, _STARVE_IDX] >= _HOT_SHORTAGE_STARVE_FRAC * w)
                & (feats[:, :, _SHORT_IDX] >= _HOT_SHORTAGE_PROP)
            )
            & machine
        )
    if f > _INBOUND_IDX:
        inbound = feats[:, :, _INBOUND_IDX]
        route = feats[:, :, _ROUTE_IDX]
        delay = (inbound >= _HOT_INBOUND_S) | (route >= _HOT_ROUTE_S)
        hot = hot | (
            (delay & (feats[:, :, _STARVE_IDX] >= _HOT_INBOUND_STARVE_FRAC * w)) & machine
        )
        if f > _STARVE_IDX:
            stall = feats[:, :, _BLOCK_IDX] + feats[:, :, _STARVE_IDX]
            stall_ok = stall >= _HOT_INBOUND_STARVE_FRAC * w
            # Gantry: route or inbound delay plus stall. AGV driving (route_delay) is not a block.
            hot = hot | ((delay & stall_ok) & gantry)
            hot = hot | (((inbound >= _HOT_INBOUND_S) & stall_ok) & agv)
    if f > _UNAVAIL_IDX:
        hot = hot | ((feats[:, :, _UNAVAIL_IDX] >= _HOT_DOWNTIME_UNAVAIL) & (gantry | agv | carrier))
    if f > _DIST_IDX:
        hot = hot | (
            (
                (feats[:, :, _DIST_IDX] >= 0.5)
                & (feats[:, :, _UNAVAIL_IDX] >= _HOT_DOWNTIME_UNAVAIL)
            )
            & machine
        )
    if have_types and f > _UNAVAIL_IDX and f > _STARVE_IDX:
        human = _type_allow(feats, _TYPE_HUMAN_IDX)
        operator_absent = ((feats[:, :, _UNAVAIL_IDX] >= _HOT_DOWNTIME_UNAVAIL) & human).any(
            axis=1, keepdims=True
        )
        stall = feats[:, :, _BLOCK_IDX] + feats[:, :, _STARVE_IDX]
        labor_sat = labor_saturated_mask(feats)
        operator_wait = operator_absent | labor_sat
        hot = hot | (((stall >= _HOT_OPERATOR_STALL_FRAC * w) & operator_wait) & machine)
    if f > _DIST_IDX:
        dist = feats[:, :, _DIST_IDX] >= 0.5
        q = feats[:, :, _Q_IDX] if f > _Q_IDX else np.zeros((t_len, n_nodes), dtype=np.float32)
        stall = (
            feats[:, :, _BLOCK_IDX] + feats[:, :, _STARVE_IDX]
            if f > _STARVE_IDX
            else np.zeros((t_len, n_nodes), dtype=np.float32)
        )
        unav = feats[:, :, _UNAVAIL_IDX] if f > _UNAVAIL_IDX else np.zeros((t_len, n_nodes), dtype=np.float32)
        real = (q >= 1.0 - 1e-9) | (stall >= _HOT_COUPLED_STALL_FRAC * w) | (unav >= _HOT_DOWNTIME_UNAVAIL)
        hot = hot & ~(dist & ~real)
    return hot


def ops_hot_mask(
    features: np.ndarray,
    *,
    window_size_s: float = 60.0,
    machine_mask: np.ndarray | None = None,
    min_hot_windows: int | None = None,
    gap_windows: int | None = None,
) -> np.ndarray:
    """Unsupervised occupancy y: operational stall / delay, no bottleneck_score.

    Default: every supervised node min 8 windows (drops 1–7 min flicker).
    An explicit ``min_hot_windows`` applies to every node (tests).
    """
    hot = ops_occupancy_raw(
        features, window_size_s=window_size_s, machine_mask=machine_mask
    )
    min_w = _HOT_MIN_WINDOWS if min_hot_windows is None else int(min_hot_windows)
    gap = _HOT_GAP_WINDOWS if gap_windows is None else int(gap_windows)
    n_nodes = int(hot.shape[1])
    per_node = np.full((n_nodes,), min_w, dtype=np.int32)
    if min_w >= _HOT_MIN_WINDOWS:
        agv = _type_allow(np.asarray(features, dtype=np.float32), _TYPE_AGV_IDX)
        if agv.any():
            per_node[agv[0]] = max(min_w, _HOT_AGV_MIN_WINDOWS)
    return smooth_occupancy_runs(
        hot.astype(np.float32),
        gap_windows=gap,
        min_windows=min_w,
        min_windows_by_node=per_node,
    )


def node_hot_mask(
    features: np.ndarray,
    scores: np.ndarray,
    *,
    score_threshold: float = _HOT_SCORE,
    window_size_s: float = 60.0,
    machine_mask: np.ndarray | None = None,
    min_hot_windows: int | None = None,
    gap_windows: int | None = None,
) -> np.ndarray:
    """A.1 occupancy: process machines plus delayed / STOP gantry / AGV.

    Machines: score ≥ threshold, TPM turning-point, plus the same operational
    rules as ``ops_hot_mask``. Gantry: route or inbound delay plus stall, or
    STOP. AGV: inbound wait plus stall, or freeze/STOP — not route_delay while
    driving. Human / buffer stay 0. Column 18 alone is not enough.
    Isolated 1-min runs are dropped, then 1-window holes between remaining runs are filled.
    """
    scores = np.asarray(scores, dtype=np.float32)
    feats = np.asarray(features, dtype=np.float32)
    if feats.ndim != 3:
        raise ValueError(f"features must be (T,N,F), got {feats.shape}")
    machine, _carrier, _have_types = _station_type_masks(feats, machine_mask)
    hot = ops_occupancy_raw(feats, window_size_s=window_size_s, machine_mask=machine_mask)
    hot = hot | ((scores[:, :, 0] >= float(score_threshold)) & machine)
    if feats.shape[2] > _TP_IDX:
        hot = hot | ((feats[:, :, _TP_IDX] > 0.5) & machine)
    min_w = _HOT_MIN_WINDOWS if min_hot_windows is None else int(min_hot_windows)
    gap = _HOT_GAP_WINDOWS if gap_windows is None else int(gap_windows)
    return smooth_occupancy_runs(hot.astype(np.float32), gap_windows=gap, min_windows=min_w)


def occupancy_node_mask(features: np.ndarray) -> np.ndarray:
    """(N,) 1 for machine / gantry / AGV columns (human and buffer stay 0)."""
    feats = np.asarray(features)
    if feats.ndim != 3:
        raise ValueError(f"features must be (T,N,F), got {feats.shape}")
    n_nodes = int(feats.shape[1])
    if feats.shape[2] < _TYPE_MACHINE_IDX + 5:
        return np.ones((n_nodes,), dtype=np.float32)
    mask = np.zeros((n_nodes,), dtype=np.float32)
    for idx in OCC_SUPERVISE_TYPE_INDICES:
        if feats.shape[2] > idx:
            mask = np.maximum(mask, (feats[0, :, idx] > 0.5).astype(np.float32))
    return mask


def jobs_remaining_series(
    kpi_rows: list[dict[str, Any]],
    window_start_s: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Unfinished jobs at each window start (complete_s > t0, or still open)."""
    t_len = int(np.asarray(window_start_s).shape[0])
    if not kpi_rows:
        rem = np.zeros((t_len,), dtype=np.float32)
        return rem, 0.0
    completes: list[float] = []
    for row in kpi_rows:
        raw = row.get("complete_s", "")
        if raw in ("", None):
            completes.append(float("inf"))
        else:
            try:
                completes.append(float(raw))
            except (TypeError, ValueError):
                completes.append(float("inf"))
    n_jobs = float(len(kpi_rows))
    starts = np.asarray(window_start_s, dtype=np.float64)
    rem = np.zeros((t_len,), dtype=np.float32)
    for ti, t0 in enumerate(starts):
        rem[ti] = sum(1 for c in completes if c > t0)
    return rem, n_jobs


def first_done_index(jobs_remaining: np.ndarray) -> int:
    """First window index where remaining jobs is 0; else length (still working)."""
    rem = np.asarray(jobs_remaining, dtype=np.float32)
    zeros = np.flatnonzero(rem <= 0)
    if zeros.size:
        return int(zeros[0])
    return int(rem.shape[0])


def pack_remain_target(
    scores: np.ndarray,
    hot: np.ndarray,
    *,
    t: int,
    done_ti: int,
    max_remain_windows: int,
    occupancy_horizon_windows: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Pad future occupancy ``[t, t+H)`` to ``max_remain_windows``.

    ``remain_len`` is still windows until jobs hit zero. Occupancy y is capped
    at ``occupancy_horizon_windows`` (A.1 fixed H, default = ``max_remain_windows``).
    """
    n_nodes = int(scores.shape[1])
    k_max = int(max_remain_windows)
    y_score = np.zeros((k_max, n_nodes, 1), dtype=np.float32)
    y_hot = np.zeros((k_max, n_nodes), dtype=np.float32)
    mask = np.zeros((k_max,), dtype=np.float32)
    remain_len = max(int(done_ti) - int(t), 0)
    k_occ = int(occupancy_horizon_windows) if occupancy_horizon_windows else k_max
    k_occ = max(1, min(k_max, k_occ))
    length = min(remain_len, k_occ)
    if length > 0:
        y_score[:length] = np.asarray(scores[t : t + length], dtype=np.float32)
        y_hot[:length] = np.asarray(hot[t : t + length], dtype=np.float32)
        mask[:length] = 1.0
    return y_score, y_hot, mask, remain_len


def pack_future_features(
    features: np.ndarray,
    cluster_id: np.ndarray | None,
    *,
    t: int,
    done_ti: int,
    max_remain_windows: int,
    occupancy_horizon_windows: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Pad future X and station cluster ids ``[t, t+H)`` to ``max_remain_windows``."""
    feats = np.asarray(features, dtype=np.float32)
    n_nodes = int(feats.shape[1])
    f_dim = int(feats.shape[2])
    k_max = int(max_remain_windows)
    y_x = np.zeros((k_max, n_nodes, f_dim), dtype=np.float32)
    y_cluster = np.full((k_max, n_nodes), -1, dtype=np.int64)
    remain_len = max(int(done_ti) - int(t), 0)
    k_occ = int(occupancy_horizon_windows) if occupancy_horizon_windows else k_max
    k_occ = max(1, min(k_max, k_occ))
    length = min(remain_len, k_occ)
    if length > 0:
        y_x[:length] = feats[t : t + length]
        if cluster_id is not None:
            cid = np.asarray(cluster_id)
            if cid.ndim == 2:
                y_cluster[:length] = cid[t : t + length]
    return y_x, y_cluster


def sinusoidal_time_pe(k_max: int, dim: int, device=None) -> "np.ndarray":
    """Standard transformer PE for future step index 0..K-1."""
    position = np.arange(k_max, dtype=np.float32)[:, None]
    div = np.exp(np.arange(0, dim, 2, dtype=np.float32) * (-np.log(10000.0) / max(dim, 1)))
    pe = np.zeros((k_max, dim), dtype=np.float32)
    pe[:, 0::2] = np.sin(position * div)
    pe[:, 1::2] = np.cos(position * div[: pe[:, 1::2].shape[1]])
    return pe


def occupancy_to_events(
    hot: np.ndarray,
    *,
    resource_ids: list[str],
    first_future_start_s: float,
    window_size_s: float,
    threshold: float = 0.5,
) -> list[dict[str, Any]]:
    """Connected 1-runs on a (K, N) occupancy grid → A.1 start / duration / station."""
    grid = np.asarray(hot, dtype=np.float32)
    if grid.ndim != 2:
        raise ValueError(f"hot must be (K, N), got {grid.shape}")
    k_len, n_nodes = grid.shape
    events: list[dict[str, Any]] = []
    eid = 0
    for ni in range(n_nodes):
        col = grid[:, ni]
        i = 0
        while i < k_len:
            if col[i] < threshold:
                i += 1
                continue
            j = i + 1
            while j < k_len and col[j] >= threshold:
                j += 1
            n_win = j - i
            start_s = float(first_future_start_s) + i * float(window_size_s)
            dur = n_win * float(window_size_s)
            rid = resource_ids[ni] if ni < len(resource_ids) else str(ni)
            events.append(
                {
                    "event_id": eid,
                    "resource_id": rid,
                    "start_s": start_s,
                    "end_s": start_s + dur,
                    "duration_s": dur,
                    "n_windows": int(n_win),
                    "mean_hot": float(col[i:j].mean()),
                }
            )
            eid += 1
            i = j
    events.sort(key=lambda e: (e["start_s"], -e["duration_s"]))
    return events


def _event_iou(a: dict[str, Any], b: dict[str, Any]) -> float:
    inter = min(float(a["end_s"]), float(b["end_s"])) - max(float(a["start_s"]), float(b["start_s"]))
    if inter <= 0:
        return 0.0
    union = float(a["duration_s"]) + float(b["duration_s"]) - inter
    return float(inter / union) if union > 0 else 0.0


def match_occupancy_events(
    pred: list[dict[str, Any]],
    true: list[dict[str, Any]],
    *,
    iou_min: float = 0.5,
) -> tuple[int, int, int]:
    """Greedy one-to-one match on same ``resource_id`` and temporal IoU.

    Returns (n_match, n_pred, n_true).
    """
    unused = set(range(len(true)))
    n_match = 0
    for p in sorted(pred, key=lambda e: -float(e.get("duration_s") or 0)):
        best_i = -1
        best_iou = -1.0
        for i in unused:
            t = true[i]
            if str(t.get("resource_id")) != str(p.get("resource_id")):
                continue
            iou = _event_iou(p, t)
            if iou > best_iou:
                best_iou = iou
                best_i = i
        if best_i >= 0 and best_iou >= float(iou_min):
            unused.discard(best_i)
            n_match += 1
    return n_match, len(pred), len(true)


def node_event_targets(
    y_hot: np.ndarray,
    *,
    min_windows: int = 8,
    remain_mask: np.ndarray | None = None,
    occ_node_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Longest occupancy run per node in H → will / start_idx / duration_windows.

    One event per station in the forecast window (who / when / how long).
    Runs shorter than ``min_windows`` are not reported.
    """
    grid = np.asarray(y_hot, dtype=np.float32)
    squeeze = grid.ndim == 2
    if squeeze:
        grid = grid[None, ...]
    if grid.ndim != 3:
        raise ValueError(f"y_hot must be (K,N) or (B,K,N), got {grid.shape}")
    batch, k_len, n_nodes = grid.shape
    will = np.zeros((batch, n_nodes), dtype=np.float32)
    start = np.zeros((batch, n_nodes), dtype=np.int64)
    dur = np.zeros((batch, n_nodes), dtype=np.float32)
    min_w = max(int(min_windows), 1)
    rm = None if remain_mask is None else np.asarray(remain_mask, dtype=np.float32)
    occ = None if occ_node_mask is None else np.asarray(occ_node_mask, dtype=np.float32)
    for b in range(batch):
        k_use = k_len
        if rm is not None:
            row = rm[b] if rm.ndim == 2 else rm.reshape(-1)
            k_use = int(row[:k_len].sum())
            k_use = max(0, min(k_use, k_len))
        node_ok = np.ones(n_nodes, dtype=bool)
        if occ is not None:
            nrow = occ[b] if occ.ndim == 2 else occ.reshape(-1)
            node_ok = nrow[:n_nodes] > 0.5
        if k_use <= 0:
            continue
        for n in range(n_nodes):
            if not bool(node_ok[n]):
                continue
            col = grid[b, :k_use, n]
            best_len = 0
            best_i = 0
            i = 0
            while i < k_use:
                if col[i] < 0.5:
                    i += 1
                    continue
                j = i + 1
                while j < k_use and col[j] >= 0.5:
                    j += 1
                if (j - i) > best_len:
                    best_len = j - i
                    best_i = i
                i = j
            if best_len >= min_w:
                will[b, n] = 1.0
                start[b, n] = int(best_i)
                dur[b, n] = float(best_len)
    if squeeze:
        return will[0], start[0], dur[0]
    return will, start, dur


def rasterize_node_events(
    will: np.ndarray,
    start: np.ndarray,
    dur: np.ndarray,
    k_len: int,
    *,
    threshold: float = 0.5,
    min_windows: int = 8,
) -> np.ndarray:
    """Fill (B, K, N) or (K, N) occupancy from per-node will / start / duration."""
    will_a = np.asarray(will, dtype=np.float32)
    start_a = np.asarray(start)
    dur_a = np.asarray(dur, dtype=np.float32)
    squeeze = will_a.ndim == 1
    if squeeze:
        will_a = will_a[None]
        start_a = start_a[None]
        dur_a = dur_a[None]
    batch, n_nodes = will_a.shape
    k_len = int(k_len)
    grid = np.zeros((batch, k_len, n_nodes), dtype=np.float32)
    min_w = max(int(min_windows), 1)
    for b in range(batch):
        for n in range(n_nodes):
            if will_a[b, n] < float(threshold):
                continue
            s = int(np.clip(int(start_a[b, n]), 0, max(k_len - 1, 0)))
            d = int(np.round(float(dur_a[b, n])))
            d = max(d, min_w)
            d = min(d, k_len - s)
            if d < min_w:
                continue
            grid[b, s : s + d, n] = 1.0
    if squeeze:
        return grid[0]
    return grid


def occupancy_event_metrics(
    y_hot: np.ndarray,
    hot_prob: np.ndarray,
    remain_mask: np.ndarray,
    occ_node_mask: np.ndarray,
    resource_ids: list[str],
    *,
    threshold: float = 0.65,
    min_windows: int = 8,
    iou_min: float = 0.5,
    window_size_s: float = 60.0,
) -> dict[str, float]:
    """Event P/R/F1 for A.1 (which station, start, duration)."""
    y = np.asarray(y_hot, dtype=np.float32)
    p = np.asarray(hot_prob, dtype=np.float32)
    rm = np.asarray(remain_mask, dtype=np.float32)
    occ = np.asarray(occ_node_mask, dtype=np.float32)
    if y.ndim != 3:
        return {"event_precision": 0.0, "event_recall": 0.0, "event_f1": 0.0, "event_n_pred": 0.0, "event_n_true": 0.0}
    ids = [str(x) for x in resource_ids]
    tp = fp = fn = 0
    min_w = max(int(min_windows), 1)
    for b in range(y.shape[0]):
        k = int(rm[b].sum()) if rm.ndim == 2 else int(y.shape[1])
        k = max(0, min(k, y.shape[1]))
        if k <= 0:
            continue
        node = occ[b] if occ.ndim == 2 else occ
        node = (node.reshape(-1) > 0.5)
        n = min(int(node.size), y.shape[2], len(ids) if ids else y.shape[2])
        yb = y[b, :k, :n].copy()
        pb = (p[b, :k, :n] >= float(threshold)).astype(np.float32)
        yb[:, ~node[:n]] = 0.0
        pb[:, ~node[:n]] = 0.0
        if min_w > 1:
            yb = smooth_occupancy_runs(yb, gap_windows=1, min_windows=min_w)
            pb = smooth_occupancy_runs(pb, gap_windows=1, min_windows=min_w)
        rids = ids[:n] if ids else [str(i) for i in range(n)]
        true_e = occupancy_to_events(yb, resource_ids=rids, first_future_start_s=0.0, window_size_s=window_size_s)
        pred_e = occupancy_to_events(pb, resource_ids=rids, first_future_start_s=0.0, window_size_s=window_size_s, threshold=0.5)
        m, n_pred, n_true = match_occupancy_events(pred_e, true_e, iou_min=iou_min)
        tp += m
        fp += max(n_pred - m, 0)
        fn += max(n_true - m, 0)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {
        "event_precision": float(prec),
        "event_recall": float(rec),
        "event_f1": float(f1),
        "event_n_pred": float(tp + fp),
        "event_n_true": float(tp + fn),
    }


def gaussian_start_soft_labels(
    start_idx: np.ndarray,
    n_bins: int,
    sigma: float = 1.0,
) -> np.ndarray:
    """Soft start targets: Gaussian over 1-min bins, σ≈1 ⇒ mass within ±2 min."""
    start = np.asarray(start_idx, dtype=np.float32).reshape(-1)
    bins = max(int(n_bins), 1)
    k = np.arange(bins, dtype=np.float32)
    sig = max(float(sigma), 1e-6)
    dist = (k[None, :] - start[:, None]) / sig
    w = np.exp(-0.5 * dist * dist)
    w = w / np.clip(w.sum(axis=-1, keepdims=True), 1e-12, None)
    return w.astype(np.float32)


def _prf(tp: float, fp: float, fn: float) -> tuple[float, float, float]:
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2.0 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return float(prec), float(rec), float(f1)


def _align_hist_last(hist_last_hot: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    last = np.asarray(hist_last_hot, dtype=np.float32)
    if last.ndim == 1:
        last = np.broadcast_to(last, shape)
    elif last.shape[0] != shape[0]:
        last = last[: shape[0]]
    if last.shape[-1] != shape[-1]:
        last = last[..., : shape[-1]]
    return np.asarray(last, dtype=np.float32)


def apply_ongoing_will_force(
    will_prob: np.ndarray,
    start_idx: np.ndarray,
    dur: np.ndarray,
    hist_last_hot: np.ndarray | None,
    *,
    threshold: float,
    min_windows: int,
    will_floor: float = 0.62,
    force_will: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Start=0 if last minute is hot. Optionally force will when remaining is long."""
    wp = np.asarray(will_prob, dtype=np.float32)
    start = np.asarray(start_idx, dtype=np.int64)
    if hist_last_hot is None:
        return wp, start
    last = _align_hist_last(hist_last_hot, wp.shape)
    pred_dur = np.asarray(dur, dtype=np.float32)
    if pred_dur.shape != wp.shape:
        n = min(pred_dur.shape[0], wp.shape[0])
        m = min(pred_dur.shape[-1], wp.shape[-1])
        pred_dur = pred_dur[:n, :m]
        last = last[:n, :m]
    force = (
        bool(force_will)
        & (last > 0.5)
        & (pred_dur >= float(min_windows))
        & (wp >= float(will_floor))
    )
    wp = np.where(force, np.maximum(wp, float(threshold)), wp)
    start = np.where(last > 0.5, 0, start)
    return wp, start


def _empty_report_metrics() -> dict[str, float]:
    keys = (
        "who_precision",
        "who_recall",
        "who_f1",
        "report_precision",
        "report_recall",
        "report_f1",
        "start_mae",
        "dur_mae",
        "who_recall_ongoing",
        "report_recall_ongoing",
        "start_mae_ongoing",
        "dur_mae_ongoing",
        "n_true_ongoing",
        "who_recall_upcoming",
        "report_recall_upcoming",
        "start_mae_upcoming",
        "dur_mae_upcoming",
        "n_true_upcoming",
        "n_pred_who",
        "n_true_who",
    )
    return {k: 0.0 for k in keys}


def station_report_metrics(
    y_hot: np.ndarray,
    will_prob: np.ndarray,
    start_idx: np.ndarray,
    dur: np.ndarray,
    remain_mask: np.ndarray,
    occ_node_mask: np.ndarray,
    *,
    threshold: float = 0.70,
    min_windows: int = 8,
    start_tol_windows: int = 3,
    hist_last_hot: np.ndarray | None = None,
    will_floor: float = 0.62,
    force_ongoing_will: bool = False,
) -> dict[str, float]:
    """Main A.1 score: station match and start error ≤ ``start_tol_windows`` min.

    Who = per-station will. When = start MAE on who-TPs; >tol is a report miss.
    How long = duration MAE on who-TPs. Ongoing (true start=0) vs upcoming.
    IoU≥0.5 stays in ``occupancy_event_metrics`` as an appendix.
    """
    out = _empty_report_metrics()
    y = np.asarray(y_hot, dtype=np.float32)
    if y.ndim != 3:
        return out
    wp = np.asarray(will_prob, dtype=np.float32)
    sp = np.asarray(start_idx)
    dp = np.asarray(dur, dtype=np.float32)
    rm = np.asarray(remain_mask, dtype=np.float32)
    occ = np.asarray(occ_node_mask, dtype=np.float32)
    y_will, y_start, y_dur = node_event_targets(
        y, min_windows=min_windows, remain_mask=rm, occ_node_mask=occ
    )
    pred_start = np.asarray(sp, dtype=np.int64)
    pred_dur = np.asarray(dp, dtype=np.float32)
    if wp.shape != y_will.shape:
        n = min(wp.shape[0], y_will.shape[0])
        m = min(wp.shape[-1], y_will.shape[-1])
        wp = wp[:n, :m]
        pred_start = pred_start[:n, :m]
        pred_dur = pred_dur[:n, :m]
        y_will = y_will[:n, :m]
        y_start = y_start[:n, :m]
        y_dur = y_dur[:n, :m]
    wp, pred_start = apply_ongoing_will_force(
        wp,
        pred_start,
        pred_dur,
        hist_last_hot,
        threshold=float(threshold),
        min_windows=int(min_windows),
        will_floor=float(will_floor),
        force_will=bool(force_ongoing_will),
    )
    pred_will = (wp >= float(threshold)).astype(np.float32)
    if occ.ndim == 1:
        node_ok = occ.reshape(1, -1) > 0.5
    else:
        node_ok = occ > 0.5
    node_ok = node_ok[: pred_will.shape[0], : pred_will.shape[-1]]
    pred_pos = (pred_will > 0.5) & node_ok
    true_pos = (y_will > 0.5) & node_ok
    who_tp = pred_pos & true_pos
    start_err = np.abs(pred_start.astype(np.int64) - y_start.astype(np.int64))
    report_hit = who_tp & (start_err <= int(start_tol_windows))
    tp_who = float(who_tp.sum())
    n_pred = float(pred_pos.sum())
    n_true = float(true_pos.sum())
    wp_, wr, wf = _prf(tp_who, n_pred - tp_who, n_true - tp_who)
    rp, rr, rf = _prf(
        float(report_hit.sum()),
        n_pred - float(report_hit.sum()),
        n_true - float(report_hit.sum()),
    )
    out["who_precision"] = wp_
    out["who_recall"] = wr
    out["who_f1"] = wf
    out["report_precision"] = rp
    out["report_recall"] = rr
    out["report_f1"] = rf
    out["n_pred_who"] = n_pred
    out["n_true_who"] = n_true
    if tp_who > 0:
        out["start_mae"] = float(start_err[who_tp].mean())
        out["dur_mae"] = float(np.abs(pred_dur - y_dur)[who_tp].mean())
    ongoing = true_pos & (y_start == 0)
    upcoming = true_pos & (y_start > 0)
    for name, mask in (("ongoing", ongoing), ("upcoming", upcoming)):
        n_m = float(mask.sum())
        out[f"n_true_{name}"] = n_m
        if n_m <= 0:
            continue
        out[f"who_recall_{name}"] = float((who_tp & mask).sum()) / n_m
        out[f"report_recall_{name}"] = float((report_hit & mask).sum()) / n_m
        who_m = who_tp & mask
        if who_m.any():
            out[f"start_mae_{name}"] = float(start_err[who_m].mean())
            out[f"dur_mae_{name}"] = float(np.abs(pred_dur - y_dur)[who_m].mean())
    return out
