"""Remaining-jobs horizon for A.1 occupancy (window × station until order done).

Causal input at time t is how many jobs are still unfinished. Training labels
are the occupancy grid from the next window through the first window where
remaining jobs hit zero. Live inference predicts that length and the grid.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# FEATURE_COLS indices (export_dataset.FEATURE_COLS)
_TP_IDX = 19  # is_turning_point
_L2_IDX = 18  # disturbance_active_s


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


def node_hot_mask(
    features: np.ndarray,
    scores: np.ndarray,
    *,
    score_threshold: float = 0.55,
) -> np.ndarray:
    """A.1 occupancy: score ≥ threshold, plus turning-point or L2 if present."""
    scores = np.asarray(scores, dtype=np.float32)
    feats = np.asarray(features, dtype=np.float32)
    if feats.ndim != 3:
        raise ValueError(f"features must be (T,N,F), got {feats.shape}")
    hot = scores[:, :, 0] >= float(score_threshold)
    if feats.shape[-1] > max(_TP_IDX, _L2_IDX):
        hot = hot | (feats[:, :, _TP_IDX] > 0.5) | (feats[:, :, _L2_IDX] > 0.5)
    return hot.astype(np.float32)


def pack_remain_target(
    scores: np.ndarray,
    hot: np.ndarray,
    *,
    t: int,
    done_ti: int,
    max_remain_windows: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Pad future occupancy ``[t, done_ti)`` to ``max_remain_windows``."""
    n_nodes = int(scores.shape[1])
    k_max = int(max_remain_windows)
    y_score = np.zeros((k_max, n_nodes, 1), dtype=np.float32)
    y_hot = np.zeros((k_max, n_nodes), dtype=np.float32)
    mask = np.zeros((k_max,), dtype=np.float32)
    remain_len = max(int(done_ti) - int(t), 0)
    length = min(remain_len, k_max)
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
