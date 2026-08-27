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
_HOT_MIN_WINDOWS = 2


def _type_allow(feats: np.ndarray, idx: int) -> np.ndarray:
    if feats.shape[2] <= idx:
        return np.zeros(feats.shape[:2], dtype=bool)
    return feats[:, :, idx] > 0.5


def smooth_occupancy_runs(
    hot: np.ndarray,
    *,
    gap_windows: int = _HOT_GAP_WINDOWS,
    min_windows: int = _HOT_MIN_WINDOWS,
) -> np.ndarray:
    """Fill ≤gap cold holes, then drop occupancy runs shorter than min_windows."""
    grid = np.asarray(hot, dtype=np.float32).copy()
    if grid.ndim != 2:
        raise ValueError(f"hot must be (T, N), got {grid.shape}")
    t_len, n_nodes = grid.shape
    gap = max(int(gap_windows), 0)
    min_w = max(int(min_windows), 1)
    if t_len == 0 or (gap == 0 and min_w <= 1):
        return grid
    for n in range(n_nodes):
        col = grid[:, n]
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

    Machines: score ≥ threshold, TPM turning-point, queue pileup, kitting
    starve, inbound-wait starve, blocked backlog, coupled stall, operator
    wait (stall while a human node is STOP, or every on-duty human is busy),
    or injected-and-STOP downtime.
    Carriers: route / inbound delay plus stall, or STOP this window. Human /
    buffer stay 0. Column 18 alone is not enough.
    Isolated 1-min runs are dropped, then 1-window holes between remaining runs are filled.
    """
    scores = np.asarray(scores, dtype=np.float32)
    feats = np.asarray(features, dtype=np.float32)
    if feats.ndim != 3:
        raise ValueError(f"features must be (T,N,F), got {feats.shape}")
    w = max(float(window_size_s), 1e-9)
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
    allow = machine | carrier
    hot = (scores[:, :, 0] >= float(score_threshold)) & machine
    if f > _TP_IDX:
        hot = hot | ((feats[:, :, _TP_IDX] > 0.5) & machine)
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
            hot = hot | ((delay & (stall >= _HOT_INBOUND_STARVE_FRAC * w)) & carrier)
    if f > _UNAVAIL_IDX:
        hot = hot | ((feats[:, :, _UNAVAIL_IDX] >= _HOT_DOWNTIME_UNAVAIL) & carrier)
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
