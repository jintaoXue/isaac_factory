"""Export HC Factory bottleneck derived tables into BNPDFormer training arrays.

Reads::

    <run_dir>/derived/episode_*/env_00/{window_feature_table,bottleneck_label}.csv

Multiple ``--run_dir`` values are merged into one bundle (episode keys
``{run_id}__episode_XX``).

Writes::

    PDFormer/raw_data/<tag>/
        meta.json
        node_map.json
        episodes.npz   # packed tensors for factory_bn.train
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import sys

import numpy as np

_PDFORMER_ROOT = Path(__file__).resolve().parent.parent
if str(_PDFORMER_ROOT) not in sys.path:
    sys.path.insert(0, str(_PDFORMER_ROOT))

from factory_bn.causes import ROOT_CAUSE_CLASSES, encode_root_cause
from factory_bn.graph import (
    RESOURCE_TYPES,
    build_factory_adjacency,
    hop_distance_matrix,
    resolve_to_node_id,
    semantic_distance_matrix,
    type_onehot,
)
from factory_bn.remain import ensure_labor_saturated_feature, jobs_remaining_series


FEATURE_COLS = [
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
    "is_turning_point",
    "is_momentary_bn",
]

TARGET_COL = "bottleneck_score_s"


def _f(row: dict[str, str], key: str, default: float = 0.0) -> float:
    v = row.get(key, "")
    if v is None or v == "":
        return default
    return float(v)


def _discover_episodes(derived_root: Path) -> list[Path]:
    eps = sorted(derived_root.glob("episode_*/env_00"))
    if not eps:
        # legacy flat layout
        flat = derived_root / "env_00"
        if flat.is_dir():
            return [flat]
    return eps


def _load_feature_table(path: Path, window_size: float) -> list[dict[str, str]]:
    rows = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if float(row["window_size_s"]) == window_size:
                rows.append(row)
    return rows


def _load_labels(path: Path, window_size: float) -> dict[int, dict[str, str]]:
    out: dict[int, dict[str, str]] = {}
    if not path.exists():
        return out
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if float(row["window_size_s"]) == window_size:
                out[int(float(row["window_index"]))] = row
    return out


def _load_job_kpi(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _kpi_completed(kpi: list[dict[str, str]]) -> int:
    n = 0
    for row in kpi:
        flag = str(row.get("completed") or "").strip().lower()
        if flag in ("1", "true", "yes"):
            n += 1
            continue
        raw = row.get("complete_s") or ""
        if raw not in ("", "None", "nan"):
            try:
                if float(raw) >= 0:
                    n += 1
            except (TypeError, ValueError):
                pass
    return n


def _has_deadlock_reset(ep_dir: Path) -> bool:
    """Raw ``episode_XX/env_00/disturbance_log.csv`` (sibling of derived/)."""
    try:
        raw = ep_dir.parents[2] / ep_dir.parent.name / "env_00" / "disturbance_log.csv"
    except IndexError:
        return False
    if not raw.is_file():
        return False
    with raw.open(newline="") as f:
        for row in csv.DictReader(f):
            if (row.get("disturbance_type") or "").strip() == "deadlock_reset":
                return True
    return False


def _load_events(path: Path, window_size: float) -> list[dict[str, Any]]:
    """Load bottleneck_event.csv rows for STGNPP-style event sequences."""
    events: list[dict[str, Any]] = []
    if not path.exists():
        return events
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if float(row.get("window_size_s") or 0) != window_size:
                continue
            events.append(
                {
                    "resource_id": row["resource_id"],
                    "start_s": float(row["start_s"]),
                    "end_s": float(row["end_s"]),
                    "duration_s": float(row["duration_s"]),
                    "start_window_index": int(float(row["start_window_index"])),
                    "end_window_index": int(float(row["end_window_index"])),
                    "event_id": int(float(row.get("event_id") or 0)),
                    "max_score": float(row.get("max_score") or 0),
                }
            )
    events.sort(key=lambda e: e["start_s"])
    return events


def _collect_nodes(feature_rows: list[dict[str, str]]) -> tuple[list[str], list[str]]:
    seen: dict[str, str] = {}
    for row in feature_rows:
        rid = row["resource_id"]
        if rid not in seen:
            seen[rid] = row["resource_type"]
    # stable sort: type order then id
    type_rank = {t: i for i, t in enumerate(RESOURCE_TYPES)}
    items = sorted(seen.items(), key=lambda kv: (type_rank.get(kv[1], 99), kv[0]))
    return [k for k, _ in items], [v for _, v in items]


def _pivot_episode(
    feature_rows: list[dict[str, str]],
    labels: dict[int, dict[str, str]],
    resource_ids: list[str],
    resource_types: list[str],
    events: list[dict[str, Any]] | None = None,
    job_kpi_rows: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    node_index = {rid: i for i, rid in enumerate(resource_ids)}
    n = len(resource_ids)
    f_dim = len(FEATURE_COLS) + len(RESOURCE_TYPES)

    by_window: dict[int, dict[str, dict[str, str]]] = {}
    for row in feature_rows:
        w = int(float(row["window_index"]))
        by_window.setdefault(w, {})[row["resource_id"]] = row

    windows = sorted(by_window.keys())
    win_to_ti = {w: i for i, w in enumerate(windows)}
    t_len = len(windows)
    features = np.zeros((t_len, n, f_dim), dtype=np.float32)
    scores = np.zeros((t_len, n, 1), dtype=np.float32)
    will = np.zeros((t_len,), dtype=np.float32)
    mark = np.full((t_len,), -1, dtype=np.int64)
    cause = np.full((t_len,), -1, dtype=np.int64)
    tts = np.full((t_len,), -1.0, dtype=np.float32)
    duration = np.full((t_len,), -1.0, dtype=np.float32)
    is_hot = np.zeros((t_len,), dtype=np.float32)
    window_start = np.zeros((t_len,), dtype=np.float32)

    type_mat = np.array([type_onehot(t) for t in resource_types], dtype=np.float32)

    for ti, w in enumerate(windows):
        window_start[ti] = _f(next(iter(by_window[w].values())), "window_start_s")
        for rid, row in by_window[w].items():
            if rid not in node_index:
                continue
            ni = node_index[rid]
            feat = [_f(row, c) for c in FEATURE_COLS] + type_mat[ni].tolist()
            features[ti, ni] = feat
            scores[ti, ni, 0] = _f(row, TARGET_COL)

        lab = labels.get(w)
        if lab is not None:
            will[ti] = float(lab.get("will_bottleneck") or 0)
            is_hot[ti] = float(lab.get("is_bottleneck_window") or 0)
            fut = (lab.get("future_bottleneck_object_id") or "").strip()
            mapped = resolve_to_node_id(fut, node_index) if fut else None
            if mapped is not None:
                mark[ti] = node_index[mapped]
            tts_v = lab.get("time_to_start") or ""
            dur_v = lab.get("duration") or ""
            if tts_v not in ("", "None", "nan"):
                tts[ti] = float(tts_v)
            if dur_v not in ("", "None", "nan"):
                duration[ti] = float(dur_v)
            cause[ti] = encode_root_cause(lab.get("root_cause_reason") or "")

    ev_rows = events or []
    ev_node, ev_start_s, ev_dur, ev_start_ti = [], [], [], []
    for e in ev_rows:
        mapped = resolve_to_node_id(e["resource_id"], node_index)
        if mapped is None:
            continue
        sw = e["start_window_index"]
        if sw not in win_to_ti:
            ti = int(np.argmin(np.abs(window_start - e["start_s"])))
        else:
            ti = win_to_ti[sw]
        ev_node.append(node_index[mapped])
        ev_start_s.append(e["start_s"])
        ev_dur.append(e["duration_s"])
        ev_start_ti.append(ti)

    jobs_rem, jobs_total = jobs_remaining_series(job_kpi_rows or [], window_start)
    features = ensure_labor_saturated_feature(features)

    return {
        "windows": np.asarray(windows, dtype=np.int64),
        "window_start_s": window_start,
        "features": features,
        "scores": scores,
        "will_bottleneck": will,
        "mark_node": mark,
        "cause": cause,
        "time_to_start": tts,
        "duration": duration,
        "is_bottleneck_window": is_hot,
        "jobs_remaining": jobs_rem,
        "jobs_total": np.float32(jobs_total),
        "event_node": np.asarray(ev_node, dtype=np.int64),
        "event_start_s": np.asarray(ev_start_s, dtype=np.float32),
        "event_duration_s": np.asarray(ev_dur, dtype=np.float32),
        "event_start_ti": np.asarray(ev_start_ti, dtype=np.int64),
    }


def _write_libcity_atomic(
    out_dir: Path,
    resource_ids: list[str],
    resource_types: list[str],
    adj: np.ndarray,
    episode_series: list[np.ndarray],
    time_intervals: int,
) -> None:
    """Write LibCity atomic files (concatenated episodes) for optional PDFormer baseline."""
    out_dir.mkdir(parents=True, exist_ok=True)
    name = "FactoryBN"

    with (out_dir / f"{name}.geo").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["geo_id", "type", "coordinates", "resource_id", "resource_type"])
        for i, (rid, rtype) in enumerate(zip(resource_ids, resource_types)):
            w.writerow([i, "Point", "[]", rid, rtype])

    with (out_dir / f"{name}.rel").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rel_id", "type", "origin_id", "destination_id", "cost"])
        rid_i = 0
        n = adj.shape[0]
        for i in range(n):
            for j in range(n):
                if i != j and adj[i, j] > 0:
                    w.writerow([rid_i, "geo", i, j, float(1.0 / adj[i, j])])
                    rid_i += 1

    # Concatenate episodes with a small gap; dynamic columns = score + ops features
    # LibCity expects: dyna_id, type, time, entity_id, <data_col...>
    data_cols = [TARGET_COL] + FEATURE_COLS
    with (out_dir / f"{name}.dyna").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dyna_id", "type", "time", "entity_id"] + data_cols)
        dyna_id = 0
        t_cursor = 0
        for series in episode_series:
            # series: (T, N, 1+13) with score first
            t_len, n, _ = series.shape
            for t in range(t_len):
                ts = f"2026-01-01T00:00:{t_cursor + t:02d}" if False else f"2026-01-01 00:00:00"
                # use synthetic ISO timestamps advancing by time_intervals seconds
                total_s = (t_cursor + t) * time_intervals
                hh = total_s // 3600
                mm = (total_s % 3600) // 60
                ss = total_s % 60
                day = 1 + hh // 24
                hh = hh % 24
                ts = f"2026-01-{day:02d} {hh:02d}:{mm:02d}:{ss:02d}"
                for ni in range(n):
                    row = [dyna_id, "state", ts, ni] + [float(series[t, ni, c]) for c in range(series.shape[-1])]
                    w.writerow(row)
                    dyna_id += 1
            t_cursor += t_len + 2  # gap between episodes

    config = {
        "geo": {"including_types": ["Point"], "Point": {}},
        "rel": {"including_types": ["geo"], "geo": {"cost": "num"}},
        "dyna": {
            "including_types": ["state"],
            "state": {c: "num" for c in data_cols},
        },
        "info": {
            "data_col": data_cols,
            "weight_col": "cost",
            "data_files": [name],
            "geo_file": name,
            "rel_file": name,
            "output_dim": 1,
            "time_intervals": time_intervals,
            "init_weight_inf_or_zero": "inf",
            "set_weight_link_or_dist": "dist",
            "calculate_weight_adj": False,
            "weight_adj_epsilon": 0.1,
        },
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


def _collect_run_episodes(
    run_dir: Path,
    window_size: float,
    *,
    name_prefix: str | None = None,
    require_complete: int = 0,
    skip_deadlock: bool = False,
) -> tuple[
    dict[str, str],
    list[
        tuple[
            str,
            list[dict[str, str]],
            dict[int, dict[str, str]],
            list[dict[str, Any]],
            list[dict[str, str]],
        ]
    ],
]:
    derived = run_dir / "derived"
    if not derived.is_dir():
        raise FileNotFoundError(f"derived/ not found under {run_dir}")

    episode_dirs = _discover_episodes(derived)
    if not episode_dirs:
        raise FileNotFoundError(f"No episode env dirs under {derived}")

    all_ids: dict[str, str] = {}
    per_ep_rows: list[
        tuple[
            str,
            list[dict[str, str]],
            dict[int, dict[str, str]],
            list[dict[str, Any]],
            list[dict[str, str]],
        ]
    ] = []
    for ep_dir in episode_dirs:
        feat_path = ep_dir / "window_feature_table.csv"
        lab_path = ep_dir / "bottleneck_label.csv"
        ev_path = ep_dir / "bottleneck_event.csv"
        if not feat_path.exists():
            continue
        rows = _load_feature_table(feat_path, window_size)
        labs = _load_labels(lab_path, window_size)
        events = _load_events(ev_path, window_size)
        kpi = _load_job_kpi(ep_dir / "job_kpi.csv")
        if not rows:
            continue
        if skip_deadlock and _has_deadlock_reset(ep_dir):
            print(f"[export] skip {ep_dir.parent.name}: deadlock_reset")
            continue
        if require_complete > 0:
            n_done = _kpi_completed(kpi)
            if n_done < int(require_complete):
                print(
                    f"[export] skip {ep_dir.parent.name}: "
                    f"completed {n_done}/{len(kpi)} < {require_complete}"
                )
                continue
        for row in rows:
            all_ids.setdefault(row["resource_id"], row["resource_type"])
        ep_base = ep_dir.parent.name if ep_dir.parent.name.startswith("episode_") else "episode_00"
        ep_name = f"{name_prefix}__{ep_base}" if name_prefix else ep_base
        per_ep_rows.append((ep_name, rows, labs, events, kpi))
    return all_ids, per_ep_rows


def load_derived_episode(
    episode_dir: Path,
    window_size: float = 60.0,
) -> dict[str, Any]:
    """Load one ``derived/episode_XX/env_00`` folder into a pivot dict.

    Adds ``resource_ids`` / ``resource_types`` / ``name`` on top of
    ``_pivot_episode`` arrays. Node order is local to this episode; callers
    that load a checkpoint should reindex to ``data_meta['resource_ids']``.
    """
    episode_dir = Path(episode_dir)
    feat_path = episode_dir / "window_feature_table.csv"
    if not feat_path.is_file():
        raise FileNotFoundError(f"missing {feat_path}")
    rows = _load_feature_table(feat_path, window_size)
    if not rows:
        raise RuntimeError(f"no window_size={window_size} rows in {feat_path}")
    labs = _load_labels(episode_dir / "bottleneck_label.csv", window_size)
    events = _load_events(episode_dir / "bottleneck_event.csv", window_size)
    kpi = _load_job_kpi(episode_dir / "job_kpi.csv")
    resource_ids, resource_types = _collect_nodes(rows)
    pivoted = _pivot_episode(
        rows, labs, resource_ids, resource_types, events, job_kpi_rows=kpi
    )
    pivoted["resource_ids"] = resource_ids
    pivoted["resource_types"] = resource_types
    parent = episode_dir.parent.name
    pivoted["name"] = parent if parent.startswith("episode_") else episode_dir.name
    return pivoted


def export_runs(
    run_dirs: list[Path],
    out_dir: Path,
    window_size: float = 60.0,
    write_atomic: bool = True,
    require_complete: int = 0,
    skip_deadlock: bool = False,
) -> Path:
    """Export one or more bottleneck runs into a single FactoryBN training bundle.

    Multiple runs get episode names ``{run_id}__episode_XX`` to avoid collisions.
    Nodes / adjacency are the union across all runs.
    """
    if not run_dirs:
        raise ValueError("At least one run_dir is required")

    labeled: list[tuple[str, Path]] = []
    for p in run_dirs:
        p = Path(p)
        labeled.append((p.name, p.resolve()))
    multi = len(labeled) > 1

    all_ids: dict[str, str] = {}
    per_ep_rows: list[
        tuple[
            str,
            list[dict[str, str]],
            dict[int, dict[str, str]],
            list[dict[str, Any]],
            list[dict[str, str]],
        ]
    ] = []
    for prefix_name, run_dir in labeled:
        prefix = prefix_name if multi else None
        ids, rows = _collect_run_episodes(
            run_dir,
            window_size,
            name_prefix=prefix,
            require_complete=require_complete,
            skip_deadlock=skip_deadlock,
        )
        for rid, rtype in ids.items():
            all_ids.setdefault(rid, rtype)
        per_ep_rows.extend(rows)

    if not per_ep_rows:
        raise RuntimeError("No feature rows found for the requested window_size")

    # Deduplicate episode names if a run is listed twice
    seen_names: set[str] = set()
    for ep_name, *_ in per_ep_rows:
        if ep_name in seen_names:
            raise ValueError(f"Duplicate episode key after merge: {ep_name}")
        seen_names.add(ep_name)

    type_rank = {t: i for i, t in enumerate(RESOURCE_TYPES)}
    items = sorted(all_ids.items(), key=lambda kv: (type_rank.get(kv[1], 99), kv[0]))
    resource_ids = [k for k, _ in items]
    resource_types = [v for _, v in items]

    adj = build_factory_adjacency(resource_ids, resource_types)
    sh_mx = hop_distance_matrix(adj)

    episodes_payload: dict[str, Any] = {}
    score_series_for_atomic: list[np.ndarray] = []
    concat_features = []

    for ep_name, rows, labs, events, kpi in per_ep_rows:
        packed = _pivot_episode(
            rows, labs, resource_ids, resource_types, events=events, job_kpi_rows=kpi
        )
        episodes_payload[ep_name] = packed
        atomic = np.concatenate(
            [packed["scores"], packed["features"][:, :, : len(FEATURE_COLS)]],
            axis=-1,
        )
        score_series_for_atomic.append(atomic)
        concat_features.append(packed["features"][:, :, : len(FEATURE_COLS)])

    concat = np.concatenate(concat_features, axis=0)
    sem_mx = semantic_distance_matrix(concat)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "episodes.npz",
        resource_ids=np.asarray(resource_ids),
        resource_types=np.asarray(resource_types),
        adj_mx=adj,
        sh_mx=sh_mx,
        sem_mx=sem_mx,
        feature_cols=np.asarray(FEATURE_COLS),
        cause_classes=np.asarray(ROOT_CAUSE_CLASSES),
        window_size_s=np.asarray([window_size]),
        **{f"{ep}_features": v["features"] for ep, v in episodes_payload.items()},
        **{f"{ep}_scores": v["scores"] for ep, v in episodes_payload.items()},
        **{f"{ep}_will": v["will_bottleneck"] for ep, v in episodes_payload.items()},
        **{f"{ep}_mark": v["mark_node"] for ep, v in episodes_payload.items()},
        **{f"{ep}_cause": v["cause"] for ep, v in episodes_payload.items()},
        **{f"{ep}_tts": v["time_to_start"] for ep, v in episodes_payload.items()},
        **{f"{ep}_duration": v["duration"] for ep, v in episodes_payload.items()},
        **{f"{ep}_is_hot": v["is_bottleneck_window"] for ep, v in episodes_payload.items()},
        **{f"{ep}_windows": v["windows"] for ep, v in episodes_payload.items()},
        **{f"{ep}_window_start_s": v["window_start_s"] for ep, v in episodes_payload.items()},
        **{f"{ep}_event_node": v["event_node"] for ep, v in episodes_payload.items()},
        **{f"{ep}_event_start_s": v["event_start_s"] for ep, v in episodes_payload.items()},
        **{f"{ep}_event_duration_s": v["event_duration_s"] for ep, v in episodes_payload.items()},
        **{f"{ep}_event_start_ti": v["event_start_ti"] for ep, v in episodes_payload.items()},
        **{f"{ep}_jobs_remaining": v["jobs_remaining"] for ep, v in episodes_payload.items()},
        **{f"{ep}_jobs_total": np.asarray([v["jobs_total"]], dtype=np.float32) for ep, v in episodes_payload.items()},
        episode_names=np.asarray(list(episodes_payload.keys())),
    )

    meta = {
        "run_dir": str(labeled[0][1]) if len(labeled) == 1 else None,
        "run_dirs": [str(p) for _, p in labeled],
        "run_names": [name for name, _ in labeled],
        "window_size_s": window_size,
        "num_nodes": len(resource_ids),
        "feature_dim": int(next(iter(episodes_payload.values()))["features"].shape[-1])
        if episodes_payload
        else len(FEATURE_COLS) + len(RESOURCE_TYPES) + 1,
        "ops_feature_dim": len(FEATURE_COLS),
        "resource_types": RESOURCE_TYPES,
        "feature_cols": FEATURE_COLS,
        "cause_classes": list(ROOT_CAUSE_CLASSES),
        "target_col": TARGET_COL,
        "episodes": {
            ep: {
                "T": int(v["features"].shape[0]),
                "will_positive": int(v["will_bottleneck"].sum()),
                "cause_labeled": int((v["cause"] >= 0).sum()),
                "hot_windows": int(v["is_bottleneck_window"].sum()),
                "n_events": int(len(v["event_node"])),
                "jobs_total": float(v["jobs_total"]),
                "jobs_remaining_start": float(v["jobs_remaining"][0]) if len(v["jobs_remaining"]) else 0.0,
            }
            for ep, v in episodes_payload.items()
        },
        "notes": [
            "Input X uses FEATURE_COLS + type one-hot (21–25) + labor_saturated at 26; does NOT include bottleneck_score_s (label leak).",
            "labor_saturated is broadcast onto machine nodes only; type one-hot indices stay 21–25.",
            "A.1 remain-to-jobs-done: jobs_remaining[t] unfinished jobs at window start; occupancy is [t, done).",
            "Score head (PDFormer): future bottleneck_score_s (dense per-node) over remaining windows.",
            "Event path (STGNPP): per-node sequences from bottleneck_event.csv; score and L2 onsets are not merged.",
            "Window will/mark/tts kept as auxiliary when events are sparse (still 180s near-term).",
            "A3 cause head: per-window root_cause_reason (L2 type or score heuristic); -1 = unlabeled.",
            "Coupling features are node-local (PROCESS_CHAIN, carrier delay, shortage at consumer).",
            "Multi-run merge uses episode keys {run_id}__episode_XX.",
        ],
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (out_dir / "node_map.json").write_text(
        json.dumps(
            {
                "resource_ids": resource_ids,
                "resource_types": resource_types,
                "geo_id": {rid: i for i, rid in enumerate(resource_ids)},
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    if write_atomic:
        _write_libcity_atomic(
            out_dir,
            resource_ids,
            resource_types,
            adj,
            score_series_for_atomic,
            time_intervals=int(window_size),
        )

    n_events = sum(len(v["event_node"]) for v in episodes_payload.values())
    cause_hist = {name: 0 for name in ROOT_CAUSE_CLASSES}
    n_cause = 0
    for v in episodes_payload.values():
        for cid in v["cause"]:
            if int(cid) < 0:
                continue
            n_cause += 1
            cause_hist[ROOT_CAUSE_CLASSES[int(cid)]] += 1
    print(
        f"[export] runs={len(run_dirs)} nodes={len(resource_ids)} "
        f"episodes={len(episodes_payload)} events={n_events} cause_labeled={n_cause} → {out_dir}"
    )
    if n_cause:
        brief = ", ".join(f"{k}={v}" for k, v in cause_hist.items() if v)
        print(f"  cause histogram: {brief}")
    for ep, v in episodes_payload.items():
        print(
            f"  {ep}: T={v['features'].shape[0]} will+={int(v['will_bottleneck'].sum())} "
            f"hot={int(v['is_bottleneck_window'].sum())} cause+={int((v['cause'] >= 0).sum())} "
            f"events={len(v['event_node'])} jobs={int(v['jobs_total'])}"
        )
    return out_dir


def export_run(
    run_dir: Path,
    out_dir: Path,
    window_size: float = 60.0,
    write_atomic: bool = True,
    require_complete: int = 0,
    skip_deadlock: bool = False,
) -> Path:
    return export_runs(
        [run_dir],
        out_dir,
        window_size=window_size,
        write_atomic=write_atomic,
        require_complete=require_complete,
        skip_deadlock=skip_deadlock,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Export FactoryBN training data")
    parser.add_argument(
        "--run_dir",
        type=str,
        action="append",
        required=True,
        help="bottleneck_dataset/<run_id> containing derived/ (repeat for multi-run merge)",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="default: PDFormer/raw_data/export",
    )
    parser.add_argument("--window_size", type=float, default=60.0)
    parser.add_argument("--no_atomic", action="store_true")
    parser.add_argument(
        "--require_complete",
        type=int,
        default=0,
        help="Skip episodes with fewer than N completed jobs (short order: 10).",
    )
    parser.add_argument(
        "--skip_deadlock",
        action="store_true",
        help="Skip episodes whose raw disturbance_log has deadlock_reset.",
    )
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    pdformer_root = here.parent
    run_dirs = [Path(p) for p in args.run_dir]
    out_dir = Path(args.out_dir).resolve() if args.out_dir else (pdformer_root / "raw_data" / "export")
    export_runs(
        run_dirs,
        out_dir,
        window_size=args.window_size,
        write_atomic=not args.no_atomic,
        require_complete=int(args.require_complete),
        skip_deadlock=bool(args.skip_deadlock),
    )


if __name__ == "__main__":
    main()
