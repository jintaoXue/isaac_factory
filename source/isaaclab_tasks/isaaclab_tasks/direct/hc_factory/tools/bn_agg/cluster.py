"""Unsupervised clustering of window feature rows (no bottleneck scores).

Fits k-means on operational station features, then a second k-means on the
mean-pooled plant snapshot so each window also gets a line-level cluster id.

``cause_aligned`` seeds one centroid per process cause plus a normal centroid
so clusters stay interpretable. It still does not write occupancy / event y.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .io_util import _read_csv, _write_csv
from .labels import _process_root_cause

CAUSE_ALIGNED_NAMES: tuple[str, ...] = (
    "normal",
    "transport_delay",
    "material_shortage",
    "blocked_downstream",
    "starved_upstream",
    "queue_buildup",
    "high_utilization",
)
CAUSE_ALIGNED_MODES = ("vanilla", "cause_aligned")

# Operational X only. Do not include bottleneck_score / is_window_peak / TPM flags.
CLUSTER_FEATURE_COLS: tuple[str, ...] = (
    "queue_length_s",
    "avg_waiting_time_s",
    "occupancy_ratio_s",
    "queue_growth_rate_s",
    "active_pct_s",
    "current_active_duration_s",
    "blocked_time_s",
    "starved_time_s",
    "stop_time_s",
    "unavailable_pct_s",
    "inter_departure_var_s",
    "upstream_blocked_ratio_s",
    "downstream_starved_ratio_s",
    "route_delay_s",
    "inbound_wait_s",
    "material_shortage_propagation_s",
    "affiliated_buffer_occ_s",
    "tb_minus_ts_s",
    "disturbance_active_s",
    "labor_saturated_s",
)

RESOURCE_TYPE_ORDER: tuple[str, ...] = (
    "machine",
    "gantry",
    "human",
    "transport_robot",
    "buffer",
)


def _f(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    v = row.get(key, "")
    if v is None or v == "":
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _type_onehot(rtype: str) -> list[float]:
    name = str(rtype or "").strip().lower()
    if name == "agv":
        name = "transport_robot"
    vec = [0.0] * len(RESOURCE_TYPE_ORDER)
    if name in RESOURCE_TYPE_ORDER:
        vec[RESOURCE_TYPE_ORDER.index(name)] = 1.0
    return vec


def row_vector(row: dict[str, Any]) -> np.ndarray:
    ops = [_f(row, c) for c in CLUSTER_FEATURE_COLS]
    return np.asarray(ops + _type_onehot(str(row.get("resource_type") or "")), dtype=np.float64)


def _window_key(row: dict[str, Any]) -> tuple:
    return (
        str(row.get("run_id") or ""),
        str(row.get("env_id") or ""),
        float(row.get("window_size_s") or 0.0),
        int(float(row.get("window_index") or 0)),
    )


def kmeans_pp(
    x: np.ndarray,
    n_clusters: int,
    *,
    seed: int = 42,
    n_iter: int = 40,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (labels, centroids, inertia). ``x`` is (N, F) float."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] == 0:
        raise ValueError("k-means needs a non-empty (N, F) matrix")
    n, _fdim = x.shape
    k = max(1, min(int(n_clusters), n))
    rng = np.random.default_rng(int(seed))
    centroids = np.empty((k, x.shape[1]), dtype=np.float64)
    centroids[0] = x[int(rng.integers(0, n))]
    closest = np.full(n, np.inf)
    for i in range(1, k):
        d2 = ((x - centroids[i - 1]) ** 2).sum(axis=1)
        closest = np.minimum(closest, d2)
        w = closest.copy()
        s = float(w.sum())
        if s <= 1e-12:
            centroids[i] = x[int(rng.integers(0, n))]
        else:
            centroids[i] = x[int(rng.choice(n, p=w / s))]
    labels = np.zeros(n, dtype=np.int64)
    for _ in range(max(int(n_iter), 1)):
        d = ((x[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        labels = d.argmin(axis=1).astype(np.int64)
        for j in range(k):
            mask = labels == j
            if not np.any(mask):
                centroids[j] = x[int(rng.integers(0, n))]
            else:
                centroids[j] = x[mask].mean(axis=0)
    d = ((x[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    labels = d.argmin(axis=1).astype(np.int64)
    inertia = float(d[np.arange(n), labels].sum())
    return labels, centroids, inertia


def _standardize_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return mean, std


def _apply_scale(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mean) / std


def _predict(x: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    d = ((x[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    return d.argmin(axis=1).astype(np.int64)


def seed_cluster_name(row: dict[str, Any]) -> str:
    """Map a station-window to normal or one of the six process causes."""
    reason = _process_root_cause(row)
    if reason in CAUSE_ALIGNED_NAMES[1:]:
        return reason
    return "normal"


def _majority_name(names: list[str], fallback: str = "normal") -> str:
    if not names:
        return fallback
    counts: dict[str, int] = defaultdict(int)
    for n in names:
        counts[str(n)] += 1
    return max(counts, key=lambda k: (counts[k], -CAUSE_ALIGNED_NAMES.index(k) if k in CAUSE_ALIGNED_NAMES else -99))


def fit_cause_aligned_cluster_model(
    rows: list[dict[str, Any]],
    *,
    seed: int = 42,
    window_size: float | None = 60.0,
    n_iter: int = 12,
) -> dict[str, Any]:
    """Seeded k-means: one locked centroid per cause + normal.

    Empty cause groups are dropped. Occupancy y is not taken from cluster ids.
    """
    used = [
        r
        for r in rows
        if window_size is None or abs(_f(r, "window_size_s") - float(window_size)) < 1e-9
    ]
    if not used:
        raise ValueError("no feature rows to cluster")
    seeds = [seed_cluster_name(r) for r in used]
    x = np.stack([row_vector(r) for r in used], axis=0)
    mean, std = _standardize_fit(x)
    xs = _apply_scale(x, mean, std)
    names: list[str] = []
    init: list[np.ndarray] = []
    for name in CAUSE_ALIGNED_NAMES:
        idx = [i for i, s in enumerate(seeds) if s == name]
        if not idx:
            continue
        names.append(name)
        init.append(xs[idx].mean(axis=0))
    if not init:
        raise ValueError("no seeded centroids")
    centroids = np.stack(init, axis=0)
    labels = np.zeros(xs.shape[0], dtype=np.int64)
    name_to_k = {n: i for i, n in enumerate(names)}
    rng = np.random.default_rng(int(seed))
    for _ in range(max(int(n_iter), 1)):
        d = ((xs[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        labels = d.argmin(axis=1).astype(np.int64)
        for i, row_seed in enumerate(seeds):
            locked = name_to_k.get(row_seed)
            if locked is not None:
                labels[i] = int(locked)
        for j in range(len(names)):
            mask = labels == j
            if np.any(mask):
                centroids[j] = xs[mask].mean(axis=0)
            else:
                centroids[j] = xs[int(rng.integers(0, xs.shape[0]))]
    d = ((xs[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    labels = d.argmin(axis=1).astype(np.int64)
    for i, row_seed in enumerate(seeds):
        locked = name_to_k.get(row_seed)
        if locked is not None:
            labels[i] = int(locked)
    inertia = float(((xs - centroids[labels]) ** 2).sum())
    sizes = np.bincount(labels, minlength=len(names)).tolist()
    purity = []
    for j, name in enumerate(names):
        mask = labels == j
        n = int(mask.sum())
        correct = int(sum(1 for i in np.where(mask)[0] if seeds[i] == name))
        purity.append(0.0 if n <= 0 else correct / n)

    groups: dict[tuple, list[int]] = defaultdict(list)
    for i, row in enumerate(used):
        groups[_window_key(row)].append(i)
    win_keys = list(groups)
    win_x = np.stack([xs[groups[key]].mean(axis=0) for key in win_keys], axis=0)
    win_names: list[str] = []
    win_init: list[np.ndarray] = []
    win_seed = [
        _majority_name([seeds[i] for i in groups[key]]) for key in win_keys
    ]
    for name in names:
        idx = [i for i, s in enumerate(win_seed) if s == name]
        if not idx:
            continue
        win_names.append(name)
        win_init.append(win_x[idx].mean(axis=0))
    if not win_init:
        win_names = list(names)
        win_c = centroids.copy()
        win_lab = np.zeros(win_x.shape[0], dtype=np.int64)
        win_inertia = 0.0
    else:
        win_c = np.stack(win_init, axis=0)
        win_name_to_k = {n: i for i, n in enumerate(win_names)}
        win_lab = ((win_x[:, None, :] - win_c[None, :, :]) ** 2).sum(axis=2).argmin(axis=1)
        win_lab = win_lab.astype(np.int64)
        for i, ws in enumerate(win_seed):
            locked = win_name_to_k.get(ws)
            if locked is not None:
                win_lab[i] = int(locked)
        win_inertia = float(((win_x - win_c[win_lab]) ** 2).sum())
    win_sizes = np.bincount(win_lab, minlength=len(win_names)).tolist()
    return {
        "mode": "cause_aligned",
        "n_clusters": int(len(names)),
        "n_window_clusters": int(len(win_names)),
        "seed": int(seed),
        "window_size_s": None if window_size is None else float(window_size),
        "feature_cols": list(CLUSTER_FEATURE_COLS),
        "resource_types": list(RESOURCE_TYPE_ORDER),
        "cluster_names": names,
        "window_cluster_names": win_names,
        "cluster_purity": purity,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "centroids": centroids.tolist(),
        "window_centroids": win_c.tolist(),
        "n_rows": int(xs.shape[0]),
        "n_windows": int(win_x.shape[0]),
        "inertia": inertia,
        "window_inertia": win_inertia,
        "cluster_sizes": sizes,
        "window_cluster_sizes": win_sizes,
    }


def fit_cluster_model(
    rows: list[dict[str, Any]],
    n_clusters: int = 8,
    *,
    seed: int = 42,
    window_size: float | None = 60.0,
) -> dict[str, Any]:
    """Fit node-level and window-level k-means. Scores are never used."""
    used = [
        r
        for r in rows
        if window_size is None or abs(_f(r, "window_size_s") - float(window_size)) < 1e-9
    ]
    if not used:
        raise ValueError("no feature rows to cluster")
    x = np.stack([row_vector(r) for r in used], axis=0)
    mean, std = _standardize_fit(x)
    xs = _apply_scale(x, mean, std)
    k = max(1, min(int(n_clusters), xs.shape[0]))
    node_lab, node_c, node_inertia = kmeans_pp(xs, k, seed=seed)
    groups: dict[tuple, list[int]] = defaultdict(list)
    for i, row in enumerate(used):
        groups[_window_key(row)].append(i)
    win_keys = list(groups)
    win_x = np.stack([xs[groups[key]].mean(axis=0) for key in win_keys], axis=0)
    k_w = max(1, min(k, win_x.shape[0]))
    win_lab, win_c, win_inertia = kmeans_pp(win_x, k_w, seed=seed + 1)
    sizes = np.bincount(node_lab, minlength=k).tolist()
    win_sizes = np.bincount(win_lab, minlength=k_w).tolist()
    return {
        "mode": "vanilla",
        "n_clusters": int(k),
        "n_window_clusters": int(k_w),
        "seed": int(seed),
        "window_size_s": None if window_size is None else float(window_size),
        "feature_cols": list(CLUSTER_FEATURE_COLS),
        "resource_types": list(RESOURCE_TYPE_ORDER),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "centroids": node_c.tolist(),
        "window_centroids": win_c.tolist(),
        "n_rows": int(xs.shape[0]),
        "n_windows": int(win_x.shape[0]),
        "inertia": node_inertia,
        "window_inertia": win_inertia,
        "cluster_sizes": sizes,
        "window_cluster_sizes": win_sizes,
    }


def assign_rows(rows: list[dict[str, Any]], model: dict[str, Any]) -> list[dict[str, Any]]:
    """Write ``cluster_id`` (station) and ``window_cluster_id`` (plant snapshot)."""
    if not rows:
        return rows
    mean = np.asarray(model["mean"], dtype=np.float64)
    std = np.asarray(model["std"], dtype=np.float64)
    node_c = np.asarray(model["centroids"], dtype=np.float64)
    win_c = np.asarray(model["window_centroids"], dtype=np.float64)
    ws = model.get("window_size_s")
    x = np.stack([row_vector(r) for r in rows], axis=0)
    xs = _apply_scale(x, mean, std)
    node_lab = _predict(xs, node_c)
    groups: dict[tuple, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        groups[_window_key(row)].append(i)
    win_lab_by_key: dict[tuple, int] = {}
    for key, idxs in groups.items():
        mean_v = xs[idxs].mean(axis=0, keepdims=True)
        win_lab_by_key[key] = int(_predict(mean_v, win_c)[0])
    for i, row in enumerate(rows):
        if ws is not None and abs(_f(row, "window_size_s") - float(ws)) >= 1e-9:
            row["cluster_id"] = -1
            row["window_cluster_id"] = -1
            row["cluster_name"] = ""
            row["window_cluster_name"] = ""
            continue
        node_id = int(node_lab[i])
        win_id = int(win_lab_by_key[_window_key(row)])
        row["cluster_id"] = node_id
        row["window_cluster_id"] = win_id
        node_names = model.get("cluster_names") or []
        win_names = model.get("window_cluster_names") or []
        row["cluster_name"] = (
            str(node_names[node_id]) if 0 <= node_id < len(node_names) else ""
        )
        row["window_cluster_name"] = (
            str(win_names[win_id]) if 0 <= win_id < len(win_names) else ""
        )
    return rows


def save_cluster_model(model: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(model, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_cluster_model(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _feature_tables(derived_root: Path) -> list[Path]:
    return sorted(derived_root.glob("**/window_feature_table.csv"))


def cluster_derived_run(
    derived_root: Path,
    n_clusters: int = 8,
    *,
    seed: int = 42,
    window_size: float | None = 60.0,
    model_path: Path | None = None,
    refit: bool = True,
    cluster_mode: str = "vanilla",
) -> dict[str, Any]:
    """Fit or reuse a cluster model and rewrite every ``window_feature_table.csv``."""
    derived_root = Path(derived_root)
    tables = _feature_tables(derived_root)
    if not tables:
        raise FileNotFoundError(f"no window_feature_table.csv under {derived_root}")
    model_path = Path(model_path) if model_path else derived_root / "cluster_model.json"
    all_rows: list[tuple[Path, list[dict[str, Any]]]] = []
    pool: list[dict[str, Any]] = []
    for path in tables:
        rows = _read_csv(path)
        all_rows.append((path, rows))
        pool.extend(rows)
    mode = str(cluster_mode or "vanilla").strip().lower()
    if mode not in CAUSE_ALIGNED_MODES:
        raise ValueError(f"cluster_mode must be one of {CAUSE_ALIGNED_MODES}, got {cluster_mode!r}")
    if refit or not model_path.is_file():
        if mode == "cause_aligned":
            model = fit_cause_aligned_cluster_model(pool, seed=seed, window_size=window_size)
        else:
            model = fit_cluster_model(pool, n_clusters, seed=seed, window_size=window_size)
        save_cluster_model(model, model_path)
    else:
        model = load_cluster_model(model_path)
    for path, rows in all_rows:
        assign_rows(rows, model)
        _write_csv(path, rows)
    summary = {
        "model_path": str(model_path),
        "n_tables": len(tables),
        "n_clusters": model.get("n_clusters"),
        "n_window_clusters": model.get("n_window_clusters"),
        "cluster_sizes": model.get("cluster_sizes"),
        "window_cluster_sizes": model.get("window_cluster_sizes"),
        "inertia": model.get("inertia"),
        "window_inertia": model.get("window_inertia"),
        "n_rows": model.get("n_rows"),
        "n_windows": model.get("n_windows"),
        "refit": bool(refit or not model_path.is_file()),
        "mode": model.get("mode", mode),
        "cluster_names": model.get("cluster_names"),
        "cluster_purity": model.get("cluster_purity"),
    }
    (derived_root / "cluster_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary
