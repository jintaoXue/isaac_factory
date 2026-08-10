#!/usr/bin/env python3
"""Offline Stage-C pipeline: raw bottleneck tables → window features + labels.

    Reads a single-run directory produced by BottleneckDataCollector (v0.2+), e.g.::

        output/bottleneck_dataset/<run_id>/episode_00/env_00/

    Legacy flat layout ``<run_id>/env_00/`` is also supported.

Writes::

    derived/window_feature_table.csv
    derived/bottleneck_label.csv
    derived/bottleneck_event.csv
    derived/job_kpi.csv              # per-job start / complete / cycle time
    derived/pipeline_summary.json    # includes order makespan & mean cycle

    Events in bottleneck_event.csv are the union of:
      - consecutive high-score windows (score coalescing), and
      - completed L2 rows from disturbance_log.csv (machine_failure /
        human_unavailable / transport_delay), so logistics gantry-down etc.
        are labeled even when bottleneck_score stays diffuse.

Usage::

    python tools/build_bottleneck_features.py \\
        --run_dir output/bottleneck_dataset/2026-07-18_17-46-35_seed42 \\
        --window_sizes 30,60 --horizon 60
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


ACTIVE_STATES = frozenset({"PROCESSING"})
BLOCKED_STATES = frozenset({"BLOCKED"})
STARVED_STATES = frozenset({"STARVED", "WAITING"})

# Spec §7.2 weights
W_QUEUE = 0.25
W_WAIT = 0.20
W_ACTIVE = 0.25
W_ACTIVE_DUR = 0.10
W_UPSTREAM = 0.10
W_DOWNSTREAM = 0.10

DEFAULT_SCORE_THRESHOLD = 0.55
DEFAULT_MIN_EVENT_WINDOWS = 2

# L2 rows in disturbance_log.csv (completed intervals have end_time_step set).
# Config-only rows (machine_config / logistics_config / …) are ignored.
DISTURBANCE_L2_TYPES = frozenset(
    {
        "transport_delay",  # logistics: gantry down
        "machine_failure",  # machine: station invalid
        "human_unavailable",  # human: worker leave
    }
)


@dataclass
class Interval:
    start: float
    end: float
    state: str


@dataclass
class ResourceTimeline:
    resource_id: str
    resource_type: str
    intervals: list[Interval] = field(default_factory=list)


def _discover_env_dirs(run_dir: Path, env_id: int | None) -> list[Path]:
    """Return env_* dirs under run_dir or under episode_*/ subfolders."""
    nested = sorted(run_dir.glob("episode_*/env_*"))
    if nested:
        env_dirs = nested
    else:
        env_dirs = sorted(run_dir.glob("env_*"))
    if env_id is not None:
        env_dirs = [d for d in env_dirs if d.name == f"env_{env_id:02d}"]
    return env_dirs


def _derived_out_dir(out_root: Path, run_dir: Path, env_dir: Path) -> Path:
    """Mirror episode nesting under derived/, e.g. derived/episode_00/env_00/."""
    try:
        rel = env_dir.relative_to(run_dir)
    except ValueError:
        return out_root / env_dir.name
    return out_root / rel


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _f(val: Any, default: float = 0.0) -> float:
    if val is None or val == "":
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _i(val: Any, default: int | None = None) -> int | None:
    if val is None or val == "":
        return default
    try:
        return int(float(val))
    except (TypeError, ValueError):
        return default


def build_timelines(
    events: list[dict[str, Any]], episode_end: float
) -> dict[str, ResourceTimeline]:
    """Convert event log into contiguous state intervals per resource."""
    by_rid: dict[str, list[dict]] = defaultdict(list)
    for e in events:
        by_rid[e["resource_id"]].append(e)

    timelines: dict[str, ResourceTimeline] = {}
    for rid, evs in by_rid.items():
        evs = sorted(evs, key=lambda x: (x["time_step"], x.get("logic_time_s", 0)))
        rtype = evs[0].get("resource_type", "unknown")
        intervals: list[Interval] = []
        # Assume IDLE before first event
        t0 = 0.0
        state = "IDLE"
        for e in evs:
            t = float(e.get("logic_time_s", e["time_step"]))
            if t > t0:
                intervals.append(Interval(t0, t, state))
            state = e.get("to_state") or state
            t0 = t
        if episode_end > t0:
            intervals.append(Interval(t0, episode_end, state))
        timelines[rid] = ResourceTimeline(rid, rtype, intervals)
    return timelines


def _overlap_duration(intervals: list[Interval], w0: float, w1: float, states: frozenset[str]) -> float:
    total = 0.0
    for iv in intervals:
        a = max(iv.start, w0)
        b = min(iv.end, w1)
        if b > a and iv.state in states:
            total += b - a
    return total


def _state_at(intervals: list[Interval], t: float) -> str | None:
    for iv in intervals:
        if iv.start <= t < iv.end:
            return iv.state
    if intervals and t >= intervals[-1].end:
        return intervals[-1].state
    return None


def _continuous_duration_ending_at(
    intervals: list[Interval], t: float, states: frozenset[str]
) -> float:
    """How long the resource has been continuously in ``states`` ending at ``t``."""
    idx = None
    for i, iv in enumerate(intervals):
        if iv.start <= t < iv.end or (i == len(intervals) - 1 and iv.start <= t <= iv.end):
            idx = i
            break
    if idx is None or intervals[idx].state not in states:
        return 0.0
    dur = t - intervals[idx].start
    for j in range(idx - 1, -1, -1):
        if (
            intervals[j].state in states
            and abs(intervals[j].end - intervals[j + 1].start) < 1e-9
        ):
            dur += intervals[j].end - intervals[j].start
        else:
            break
    return max(dur, 0.0)


def _station_id_to_resource_candidates(station_id: str) -> list[str]:
    if not station_id or station_id == "unknown":
        return []
    return [station_id]


def compute_window_features(
    timelines: dict[str, ResourceTimeline],
    job_rows: list[dict],
    buffer_rows: list[dict],
    transport_rows: list[dict],
    material_rows: list[dict],
    window_size: float,
    episode_end: float,
    run_id: str,
    env_id: int,
) -> list[dict]:
    # Precompute job waiting intervals keyed by station
    wait_by_station: dict[str, list[tuple[float, float]]] = defaultdict(list)
    departures_by_station: dict[str, list[float]] = defaultdict(list)
    open_queue: dict[tuple[int, str], tuple[float, str]] = {}

    for row in sorted(job_rows, key=lambda r: _f(r.get("time_step"))):
        event = row.get("event")
        job_id = _i(row.get("job_id"), -1)
        task = row.get("task") or ""
        station = row.get("station_id") or "unknown"
        t = _f(row.get("logic_time_s"), _f(row.get("time_step")))
        key = (job_id if job_id is not None else -1, task)
        if event == "queue_enter":
            open_queue[key] = (t, station)
        elif event == "queue_leave":
            if key in open_queue:
                t0, st = open_queue.pop(key)
                wait_by_station[st].append((t0, t))
            else:
                qe = _f(row.get("queue_enter_time_step"), t)
                wait_by_station[station].append((qe, t))
        elif event == "departure":
            departures_by_station[station].append(t)

    # Close open queues at episode end
    for key, (t0, st) in open_queue.items():
        wait_by_station[st].append((t0, episode_end))

    # Buffer occupancy time series (step events)
    buffer_occ: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    for row in sorted(buffer_rows, key=lambda r: _f(r.get("time_step"))):
        bid = row.get("buffer_id") or ""
        t = _f(row.get("logic_time_s"), _f(row.get("time_step")))
        occ = _f(row.get("occupancy"))
        ratio = _f(row.get("occupancy_ratio"))
        buffer_occ[bid].append((t, occ, ratio))

    def buffer_stats(bid: str, w0: float, w1: float) -> tuple[float, float, float]:
        series = buffer_occ.get(bid, [])
        if not series:
            return 0.0, 0.0, 0.0
        # last value at/before w0 and at/before w1
        before = [s for s in series if s[0] <= w0]
        during = [s for s in series if w0 < s[0] <= w1]
        start_occ = before[-1][1] if before else series[0][1]
        end_pts = before + during
        end_occ = end_pts[-1][1] if end_pts else start_occ
        # mean occupancy approx: average of samples in window, else start
        samples = [s[1] for s in series if w0 <= s[0] < w1]
        mean_occ = statistics.mean(samples) if samples else start_occ
        ratios = [s[2] for s in series if w0 <= s[0] < w1]
        if not ratios:
            ratios = [before[-1][2]] if before else [0.0]
        mean_ratio = statistics.mean(ratios)
        growth = (end_occ - start_occ) / window_size if window_size else 0.0
        return mean_occ, mean_ratio, growth

    # Transport delays overlapping window
    transport_completed = [
        r for r in transport_rows if r.get("status") == "completed" or r.get("transport_end_time_step")
    ]

    def route_delay_in_window(w0: float, w1: float) -> float:
        delays = []
        for r in transport_completed:
            end = _f(r.get("transport_end_time_step"), _f(r.get("dropoff_time_step")))
            start = _f(r.get("transport_start_time_step"), _f(r.get("request_time_step")))
            if end < w0 or start >= w1:
                continue
            req = _f(r.get("request_time_step"), start)
            delays.append(max(end - req, 0.0))
        return statistics.mean(delays) if delays else 0.0

    def shortage_ratio(w0: float, w1: float) -> float:
        rows = [
            r
            for r in material_rows
            if w0 <= _f(r.get("logic_time_s"), _f(r.get("time_step"))) < w1
        ]
        if not rows:
            return 0.0
        flagged = sum(1 for r in rows if str(r.get("shortage_flag")) in ("1", "True", "true"))
        return flagged / len(rows)

    # Ensure buffer resources appear even without resource events
    for bid in buffer_occ:
        if bid not in timelines:
            timelines[bid] = ResourceTimeline(bid, "buffer", [Interval(0.0, episode_end, "IDLE")])

    n_windows = max(int(math.ceil(episode_end / window_size)), 1)
    rows_out: list[dict] = []

    for wi in range(n_windows):
        w0 = wi * window_size
        w1 = min((wi + 1) * window_size, episode_end)
        if w1 <= w0:
            continue
        wlen = w1 - w0
        route_delay = route_delay_in_window(w0, w1)
        mat_short = shortage_ratio(w0, w1)

        # Global blocked/starved mass for upstream/downstream proxies
        total_blocked = 0.0
        total_starved = 0.0
        for tl in timelines.values():
            if tl.resource_type == "buffer":
                continue
            total_blocked += _overlap_duration(tl.intervals, w0, w1, BLOCKED_STATES)
            total_starved += _overlap_duration(tl.intervals, w0, w1, STARVED_STATES)
        n_non_buf = max(sum(1 for tl in timelines.values() if tl.resource_type != "buffer"), 1)
        global_blocked_ratio = total_blocked / (n_non_buf * wlen)
        global_starved_ratio = total_starved / (n_non_buf * wlen)

        for rid, tl in timelines.items():
            active = _overlap_duration(tl.intervals, w0, w1, ACTIVE_STATES)
            blocked = _overlap_duration(tl.intervals, w0, w1, BLOCKED_STATES)
            starved = _overlap_duration(tl.intervals, w0, w1, STARVED_STATES)
            active_pct = active / wlen
            blocked_time = blocked
            starved_time = starved
            active_dur = _continuous_duration_ending_at(tl.intervals, w1, ACTIVE_STATES)

            # Queue / waiting tied to station_id == resource_id
            waits = wait_by_station.get(rid, [])
            wait_lens = [max(min(b, w1) - max(a, w0), 0.0) for a, b in waits if b > w0 and a < w1]
            avg_wait = statistics.mean(wait_lens) if wait_lens else 0.0
            queue_length = float(sum(1 for a, b in waits if a < w1 and b > w0))

            deps = [d for d in departures_by_station.get(rid, []) if w0 <= d < w1]
            if len(deps) >= 3:
                gaps = [deps[i + 1] - deps[i] for i in range(len(deps) - 1)]
                inter_dep_var = statistics.pvariance(gaps) if len(gaps) >= 2 else 0.0
            else:
                inter_dep_var = 0.0

            if tl.resource_type == "buffer" or rid.startswith("storage_"):
                mean_occ, mean_ratio, growth = buffer_stats(rid, w0, w1)
                queue_length = mean_occ
                occupancy_ratio = mean_ratio
                queue_growth = growth
            else:
                occupancy_ratio = 0.0
                queue_growth = 0.0

            # Simple system coupling proxies
            upstream_blocked = global_blocked_ratio
            downstream_starved = global_starved_ratio
            if tl.resource_type in ("machine", "gantry"):
                # Local emphasis: own blocked/starved contribute
                upstream_blocked = min(1.0, global_blocked_ratio + blocked / wlen)
                downstream_starved = min(1.0, global_starved_ratio + starved / wlen)

            rows_out.append(
                {
                    "run_id": run_id,
                    "env_id": env_id,
                    "window_index": wi,
                    "window_start_s": w0,
                    "window_end_s": w1,
                    "window_size_s": window_size,
                    "resource_id": rid,
                    "resource_type": tl.resource_type,
                    "queue_length_s": round(queue_length, 6),
                    "avg_waiting_time_s": round(avg_wait, 6),
                    "occupancy_ratio_s": round(occupancy_ratio, 6),
                    "queue_growth_rate_s": round(queue_growth, 6),
                    "active_pct_s": round(active_pct, 6),
                    "current_active_duration_s": round(active_dur, 6),
                    "blocked_time_s": round(blocked_time, 6),
                    "starved_time_s": round(starved_time, 6),
                    "inter_departure_var_s": round(inter_dep_var, 6),
                    "upstream_blocked_ratio_s": round(upstream_blocked, 6),
                    "downstream_starved_ratio_s": round(downstream_starved, 6),
                    "route_delay_s": round(route_delay, 6),
                    "material_shortage_propagation_s": round(mat_short, 6),
                }
            )
    return rows_out


def _norm_across(values: list[float]) -> list[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        return [0.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def add_bottleneck_scores(feature_rows: list[dict]) -> list[dict]:
    """Add bottleneck_score_s per row; normalize features within each window."""
    by_window: dict[tuple, list[dict]] = defaultdict(list)
    for r in feature_rows:
        key = (r["run_id"], r["env_id"], r["window_size_s"], r["window_index"])
        by_window[key].append(r)

    for rows in by_window.values():
        q = _norm_across([r["queue_length_s"] for r in rows])
        w = _norm_across([r["avg_waiting_time_s"] for r in rows])
        a = [r["active_pct_s"] for r in rows]  # already 0-1
        d = _norm_across([r["current_active_duration_s"] for r in rows])
        for i, r in enumerate(rows):
            score = (
                W_QUEUE * q[i]
                + W_WAIT * w[i]
                + W_ACTIVE * a[i]
                + W_ACTIVE_DUR * d[i]
                + W_UPSTREAM * r["upstream_blocked_ratio_s"]
                + W_DOWNSTREAM * r["downstream_starved_ratio_s"]
            )
            r["bottleneck_score_s"] = round(score, 6)
            r["norm_queue_length_s"] = round(q[i], 6)
            r["norm_avg_waiting_time_s"] = round(w[i], 6)
            r["norm_current_active_duration_s"] = round(d[i], 6)
    return feature_rows


def _map_disturbance_resource_type(resource_id: str, raw_type: str) -> str:
    """Map disturbance_log target type onto feature-table resource_type."""
    rid = (resource_id or "").strip()
    rt = (raw_type or "").strip().lower()
    if rid.startswith("gantry_") or rt in ("gantry", "logistics"):
        return "gantry"
    if rid.startswith("human_") or rt == "human":
        return "human"
    if rt in ("machine", "gantry", "human", "transport_robot", "buffer"):
        return rt
    return raw_type or "machine"


def parse_disturbance_l2_intervals(disturbance_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Extract completed L2 intervals from disturbance_log.csv.

    Collector writes a start row (no end) and an end row (with end_time_step).
    We keep only completed intervals whose disturbance_type is in DISTURBANCE_L2_TYPES.
    """
    by_id: dict[str, dict[str, Any]] = {}
    for row in disturbance_rows:
        dtype = (row.get("disturbance_type") or "").strip()
        if dtype not in DISTURBANCE_L2_TYPES:
            continue
        did = (row.get("disturbance_id") or "").strip()
        if not did:
            continue
        start_s = _f(row.get("start_logic_time_s"), _f(row.get("start_time_step")))
        end_s = _f(row.get("end_logic_time_s"), _f(row.get("end_time_step"), default=-1.0))
        rid = (row.get("target_resource_id") or "").strip()
        rtype = _map_disturbance_resource_type(rid, row.get("target_resource_type") or "")
        cur = by_id.get(did)
        if cur is None:
            by_id[did] = {
                "disturbance_id": did,
                "disturbance_type": dtype,
                "resource_id": rid,
                "resource_type": rtype,
                "start_s": start_s,
                "end_s": end_s if end_s >= 0 else None,
                "run_id": row.get("run_id", ""),
                "env_id": _i(row.get("env_id"), 0),
            }
        else:
            cur["start_s"] = min(cur["start_s"], start_s) if cur["start_s"] is not None else start_s
            if end_s >= 0:
                cur["end_s"] = end_s if cur["end_s"] is None else max(float(cur["end_s"]), end_s)
            if rid:
                cur["resource_id"] = rid
                cur["resource_type"] = rtype

    out: list[dict[str, Any]] = []
    for ev in by_id.values():
        if not ev.get("resource_id"):
            continue
        if ev.get("end_s") is None:
            continue
        if float(ev["end_s"]) <= float(ev["start_s"]):
            continue
        out.append(ev)
    out.sort(key=lambda e: (e["start_s"], e["resource_id"]))
    return out


def _disturbance_to_window_event(
    interval: dict[str, Any],
    window_size: float,
    window_meta: dict[int, dict],
    windows: dict[int, list[dict]],
) -> dict[str, Any] | None:
    """Snap a disturbance L2 interval onto feature windows for one window_size."""
    if not window_meta:
        return None
    start_s = float(interval["start_s"])
    end_s = float(interval["end_s"])
    rid = interval["resource_id"]

    covering = [
        wi
        for wi, meta in window_meta.items()
        if meta["window_start_s"] < end_s and meta["window_end_s"] > start_s
    ]
    if not covering:
        # Fallback: nearest window by start time
        wi = min(
            window_meta.keys(),
            key=lambda i: abs(window_meta[i]["window_start_s"] - start_s),
        )
        covering = [wi]

    covering = sorted(covering)
    start_wi, end_wi = covering[0], covering[-1]
    max_score = 0.0
    for wi in covering:
        for r in windows.get(wi, []):
            if r["resource_id"] == rid:
                max_score = max(max_score, float(r.get("bottleneck_score_s") or 0.0))
                break

    meta0 = window_meta[start_wi]
    meta1 = window_meta[end_wi]
    return {
        "run_id": interval.get("run_id") or meta0["run_id"],
        "env_id": interval.get("env_id", meta0["env_id"]),
        "window_size_s": window_size,
        "resource_id": rid,
        "resource_type": interval["resource_type"],
        "start_window_index": start_wi,
        "end_window_index": end_wi,
        "start_s": start_s,
        "end_s": end_s,
        "duration_s": end_s - start_s,
        "max_score": round(max_score, 6),
        "n_windows": len(covering),
        "event_source": "disturbance_log",
        "disturbance_id": interval.get("disturbance_id", ""),
        "disturbance_type": interval.get("disturbance_type", ""),
    }


def _intervals_overlap(a0: float, a1: float, b0: float, b1: float) -> bool:
    return a0 < b1 and b0 < a1


def merge_score_and_disturbance_events(
    score_events: list[dict],
    disturbance_events: list[dict],
) -> list[dict]:
    """Union score-coalesced events with disturbance L2; expand overlaps on same resource."""
    merged: list[dict] = [dict(e) for e in score_events]
    for e in merged:
        e.setdefault("event_source", "score")
        e.setdefault("disturbance_id", "")
        e.setdefault("disturbance_type", "")

    for dist in disturbance_events:
        hit = None
        for ev in merged:
            if ev["window_size_s"] != dist["window_size_s"]:
                continue
            if ev["resource_id"] != dist["resource_id"]:
                continue
            if _intervals_overlap(ev["start_s"], ev["end_s"], dist["start_s"], dist["end_s"]):
                hit = ev
                break
        if hit is None:
            merged.append(dict(dist))
            continue
        hit["start_s"] = min(float(hit["start_s"]), float(dist["start_s"]))
        hit["end_s"] = max(float(hit["end_s"]), float(dist["end_s"]))
        hit["duration_s"] = float(hit["end_s"]) - float(hit["start_s"])
        hit["start_window_index"] = min(int(hit["start_window_index"]), int(dist["start_window_index"]))
        hit["end_window_index"] = max(int(hit["end_window_index"]), int(dist["end_window_index"]))
        hit["n_windows"] = max(int(hit.get("n_windows") or 0), int(dist.get("n_windows") or 0))
        hit["max_score"] = max(float(hit.get("max_score") or 0), float(dist.get("max_score") or 0))
        src = hit.get("event_source") or "score"
        hit["event_source"] = "both" if src in ("score", "both") else "disturbance_log"
        hit["disturbance_id"] = dist.get("disturbance_id") or hit.get("disturbance_id", "")
        hit["disturbance_type"] = dist.get("disturbance_type") or hit.get("disturbance_type", "")

    merged.sort(key=lambda e: (e["window_size_s"], e["start_s"], e["resource_id"]))
    for i, ev in enumerate(merged):
        ev["event_id"] = i
    return merged


def build_labels_and_events(
    feature_rows: list[dict],
    horizon: float,
    score_threshold: float,
    min_event_windows: int,
    disturbance_rows: list[dict[str, str]] | None = None,
) -> tuple[list[dict], list[dict]]:
    """Window-level labels + bottleneck events.

    Events come from:
      1) consecutive high-score windows (score coalescing), and
      2) completed L2 rows in disturbance_log (machine/human/logistics),
         so injected disturbances are labeled even when score stays diffuse
         (e.g. logistics gantry down).
    """
    by_ws: dict[float, list[dict]] = defaultdict(list)
    for r in feature_rows:
        by_ws[r["window_size_s"]].append(r)

    dist_intervals = parse_disturbance_l2_intervals(disturbance_rows or [])

    label_rows: list[dict] = []
    event_rows: list[dict] = []

    for window_size, rows in by_ws.items():
        # Per window: argmax resource
        windows: dict[int, list[dict]] = defaultdict(list)
        for r in rows:
            windows[r["window_index"]].append(r)

        window_meta: dict[int, dict] = {}
        for wi, rs in sorted(windows.items()):
            best = max(rs, key=lambda x: x["bottleneck_score_s"])
            window_meta[wi] = {
                "window_index": wi,
                "window_start_s": rs[0]["window_start_s"],
                "window_end_s": rs[0]["window_end_s"],
                "window_size_s": window_size,
                "run_id": rs[0]["run_id"],
                "env_id": rs[0]["env_id"],
                "bottleneck_node_t": best["resource_id"],
                "bottleneck_type_t": best["resource_type"],
                "bottleneck_score_t": best["bottleneck_score_s"],
                "is_bottleneck_window": int(best["bottleneck_score_s"] >= score_threshold),
            }

        # Merge consecutive high-score windows into events
        score_events: list[dict] = []
        cur: dict | None = None
        for wi in sorted(window_meta):
            meta = window_meta[wi]
            hot = meta["is_bottleneck_window"]
            node = meta["bottleneck_node_t"]
            if hot:
                if cur and cur["resource_id"] == node and wi == cur["end_window_index"] + 1:
                    cur["end_window_index"] = wi
                    cur["end_s"] = meta["window_end_s"]
                    cur["duration_s"] = cur["end_s"] - cur["start_s"]
                    cur["max_score"] = max(cur["max_score"], meta["bottleneck_score_t"])
                    cur["n_windows"] += 1
                else:
                    if cur and cur["n_windows"] >= min_event_windows:
                        score_events.append(cur)
                    cur = {
                        "run_id": meta["run_id"],
                        "env_id": meta["env_id"],
                        "window_size_s": window_size,
                        "resource_id": node,
                        "resource_type": meta["bottleneck_type_t"],
                        "start_window_index": wi,
                        "end_window_index": wi,
                        "start_s": meta["window_start_s"],
                        "end_s": meta["window_end_s"],
                        "duration_s": meta["window_end_s"] - meta["window_start_s"],
                        "max_score": meta["bottleneck_score_t"],
                        "n_windows": 1,
                        "event_source": "score",
                        "disturbance_id": "",
                        "disturbance_type": "",
                    }
            else:
                if cur and cur["n_windows"] >= min_event_windows:
                    score_events.append(cur)
                cur = None
        if cur and cur["n_windows"] >= min_event_windows:
            score_events.append(cur)

        dist_events: list[dict] = []
        for interval in dist_intervals:
            ev = _disturbance_to_window_event(interval, window_size, window_meta, windows)
            if ev is not None:
                dist_events.append(ev)

        events = merge_score_and_disturbance_events(score_events, dist_events)
        event_rows.extend(events)

        # Future labels per window (uses merged events, including disturbance L2)
        for wi, meta in sorted(window_meta.items()):
            t = meta["window_start_s"]
            future = [
                ev
                for ev in events
                if ev["start_s"] > t and ev["start_s"] <= t + horizon
            ]
            if future:
                first = min(future, key=lambda e: e["start_s"])
                will = 1
                fut_id = first["resource_id"]
                fut_type = first["resource_type"]
                tts = first["start_s"] - t
                dur = first["duration_s"]
            else:
                will = 0
                fut_id = ""
                fut_type = ""
                tts = ""
                dur = ""

            # Heuristic root cause; prefer disturbance type when L2 overlaps this window
            reason = ""
            for ev in events:
                if ev.get("event_source") in ("disturbance_log", "both") and _intervals_overlap(
                    meta["window_start_s"],
                    meta["window_end_s"],
                    float(ev["start_s"]),
                    float(ev["end_s"]),
                ):
                    reason = ev.get("disturbance_type") or "disturbance_l2"
                    break
            if not reason and meta["is_bottleneck_window"]:
                node_rows = [
                    r
                    for r in windows[wi]
                    if r["resource_id"] == meta["bottleneck_node_t"]
                ]
                if node_rows:
                    nr = node_rows[0]
                    if nr["blocked_time_s"] >= nr["starved_time_s"] and nr["blocked_time_s"] > 0:
                        reason = "blocked_downstream"
                    elif nr["starved_time_s"] > 0:
                        reason = "starved_upstream"
                    elif nr["active_pct_s"] >= 0.8:
                        reason = "high_utilization"
                    elif nr["queue_length_s"] > 0 or nr["avg_waiting_time_s"] > 0:
                        reason = "queue_buildup"
                    else:
                        reason = "score_threshold"

            label_rows.append(
                {
                    **meta,
                    "horizon_s": horizon,
                    "will_bottleneck": will,
                    "future_bottleneck_object_id": fut_id,
                    "future_bottleneck_type": fut_type,
                    "time_to_start": tts,
                    "duration": dur,
                    "root_cause_reason": reason,
                }
            )

    return label_rows, event_rows


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = fieldnames or list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def build_job_kpis(
    job_rows: list[dict[str, str]],
    run_id: str,
    env_id: int,
    episode_id: int | None = None,
) -> tuple[list[dict], dict]:
    """Per-job start/complete/cycle from job_trace + order-level throughput KPIs.

    Definitions (logic seconds, logic_dt=1 → same as time_step):
      - start: first ``job_selected`` for this job_id
      - complete: ``stage_complete`` (product enters progress["finished"]);
        fallback: last ``process_end`` on paint_rust_proof if stage_complete missing
      - cycle_time_s: complete - start (flow / sojourn time of one pipe)
      - order_makespan_s: max(complete) among completed jobs (from episode t=0)
    """
    by_job: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in job_rows:
        jid = r.get("job_id", "")
        if jid == "":
            continue
        by_job[jid].append(r)

    kpi_rows: list[dict] = []
    for jid in sorted(by_job.keys(), key=lambda x: int(float(x)) if str(x).replace(".", "", 1).isdigit() else str(x)):
        evs = by_job[jid]
        product_type = next((e.get("product_type") or "" for e in evs if e.get("product_type")), "")
        starts = [
            _f(e.get("logic_time_s"), _f(e.get("time_step")))
            for e in evs
            if e.get("event") == "job_selected"
        ]
        completes = [
            _f(e.get("logic_time_s"), _f(e.get("time_step")))
            for e in evs
            if e.get("event") == "stage_complete"
        ]
        paint_ends = [
            _f(e.get("logic_time_s"), _f(e.get("time_step")))
            for e in evs
            if e.get("event") == "process_end" and e.get("task") == "paint_rust_proof"
        ]
        start_s = min(starts) if starts else None
        if completes:
            complete_s = min(completes)
            complete_source = "stage_complete"
        elif paint_ends:
            complete_s = min(paint_ends)
            complete_source = "paint_process_end"
        else:
            complete_s = None
            complete_source = ""

        cycle = None
        if start_s is not None and complete_s is not None:
            cycle = complete_s - start_s

        kpi_rows.append(
            {
                "run_id": run_id,
                "env_id": env_id,
                "episode_id": "" if episode_id is None else episode_id,
                "job_id": jid,
                "product_type": product_type,
                "start_s": "" if start_s is None else round(start_s, 3),
                "complete_s": "" if complete_s is None else round(complete_s, 3),
                "cycle_time_s": "" if cycle is None else round(cycle, 3),
                "completed": 1 if complete_s is not None else 0,
                "complete_source": complete_source,
            }
        )

    completed = [r for r in kpi_rows if r["completed"] == 1 and r["cycle_time_s"] != ""]
    cycles = [float(r["cycle_time_s"]) for r in completed]
    complete_times = [float(r["complete_s"]) for r in completed]
    start_times = [float(r["start_s"]) for r in completed if r["start_s"] != ""]

    order_summary = {
        "n_jobs": len(kpi_rows),
        "n_completed": len(completed),
        "n_incomplete": len(kpi_rows) - len(completed),
        "order_makespan_s": round(max(complete_times), 3) if complete_times else None,
        "first_job_start_s": round(min(start_times), 3) if start_times else None,
        "last_job_complete_s": round(max(complete_times), 3) if complete_times else None,
        "mean_cycle_time_s": round(statistics.mean(cycles), 3) if cycles else None,
        "median_cycle_time_s": round(statistics.median(cycles), 3) if cycles else None,
        "std_cycle_time_s": round(statistics.pstdev(cycles), 3) if len(cycles) >= 2 else (0.0 if cycles else None),
        "min_cycle_time_s": round(min(cycles), 3) if cycles else None,
        "max_cycle_time_s": round(max(cycles), 3) if cycles else None,
        "throughput_jobs_per_hour": (
            round(len(completed) / (max(complete_times) / 3600.0), 4)
            if complete_times and max(complete_times) > 0
            else None
        ),
    }
    return kpi_rows, order_summary


def process_env_dir(
    env_dir: Path,
    out_dir: Path,
    window_sizes: list[float],
    horizon: float,
    score_threshold: float,
    min_event_windows: int,
) -> dict:
    events = _read_jsonl(env_dir / "resource_event_log.jsonl")
    job_rows = _read_csv(env_dir / "job_trace.csv")
    buffer_rows = _read_csv(env_dir / "buffer_event_log.csv")
    transport_rows = _read_csv(env_dir / "route_transport_task.csv")
    material_rows = _read_csv(env_dir / "material_inventory_log.csv")
    ep_rows = _read_csv(env_dir / "episode_config.csv")
    disturbance_rows = _read_csv(env_dir / "disturbance_log.csv")

    run_id = ep_rows[0]["run_id"] if ep_rows else env_dir.parent.name
    env_id = _i(ep_rows[0].get("env_id"), 0) if ep_rows else 0
    episode_id = _i(ep_rows[0].get("episode_id"), None) if ep_rows else None
    if episode_id is None:
        # Infer from path episode_XX/env_YY
        for part in env_dir.parts:
            if part.startswith("episode_"):
                try:
                    episode_id = int(part.split("_", 1)[1])
                except ValueError:
                    pass
                break

    times = []
    for e in events:
        times.append(_f(e.get("logic_time_s"), _f(e.get("time_step"))))
    for r in job_rows:
        times.append(_f(r.get("logic_time_s"), _f(r.get("time_step"))))
    episode_end = max(times) if times else 0.0
    if episode_end <= 0:
        raise RuntimeError(f"No usable timestamps in {env_dir}")

    timelines = build_timelines(events, episode_end)

    all_features: list[dict] = []
    for ws in window_sizes:
        feats = compute_window_features(
            timelines=dict(timelines),  # copy ids; buffers may be added
            job_rows=job_rows,
            buffer_rows=buffer_rows,
            transport_rows=transport_rows,
            material_rows=material_rows,
            window_size=ws,
            episode_end=episode_end,
            run_id=run_id,
            env_id=env_id if env_id is not None else 0,
        )
        all_features.extend(feats)

    all_features = add_bottleneck_scores(all_features)
    labels, event_rows = build_labels_and_events(
        all_features,
        horizon,
        score_threshold,
        min_event_windows,
        disturbance_rows=disturbance_rows,
    )
    job_kpi_rows, order_kpi = build_job_kpis(
        job_rows,
        run_id=run_id,
        env_id=env_id if env_id is not None else 0,
        episode_id=episode_id,
    )

    _write_csv(out_dir / "window_feature_table.csv", all_features)
    _write_csv(out_dir / "bottleneck_label.csv", labels)
    _write_csv(out_dir / "bottleneck_event.csv", event_rows)
    _write_csv(
        out_dir / "job_kpi.csv",
        job_kpi_rows,
        fieldnames=[
            "run_id",
            "env_id",
            "episode_id",
            "job_id",
            "product_type",
            "start_s",
            "complete_s",
            "cycle_time_s",
            "completed",
            "complete_source",
        ],
    )

    # Summary stats
    top_nodes = []
    for ws in window_sizes:
        ws_labels = [l for l in labels if l["window_size_s"] == ws]
        hot = [l for l in ws_labels if l["is_bottleneck_window"]]
        node_counts = defaultdict(int)
        for l in hot:
            node_counts[l["bottleneck_node_t"]] += 1
        top = sorted(node_counts.items(), key=lambda x: -x[1])[:5]
        top_nodes.append({"window_size_s": ws, "hot_windows": len(hot), "top_nodes": top})

    summary = {
        "run_id": run_id,
        "env_id": env_id,
        "episode_id": episode_id,
        "episode_end_s": episode_end,
        "n_resources": len(timelines),
        "n_feature_rows": len(all_features),
        "n_label_rows": len(labels),
        "n_events": len(event_rows),
        "n_job_kpi_rows": len(job_kpi_rows),
        "order_kpi": order_kpi,
        "window_sizes": window_sizes,
        "horizon_s": horizon,
        "score_threshold": score_threshold,
        "min_event_windows": min_event_windows,
        "n_disturbance_l2": len(parse_disturbance_l2_intervals(disturbance_rows)),
        "n_events_from_disturbance": sum(
            1
            for e in event_rows
            if e.get("event_source") in ("disturbance_log", "both")
        ),
        "per_window_size": top_nodes,
        "will_bottleneck_rate": {
            str(ws): (
                sum(1 for l in labels if l["window_size_s"] == ws and l["will_bottleneck"] == 1)
                / max(sum(1 for l in labels if l["window_size_s"] == ws), 1)
            )
            for ws in window_sizes
        },
    }
    (out_dir / "pipeline_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build offline bottleneck features & labels")
    parser.add_argument(
        "--run_dir",
        type=Path,
        required=True,
        help="Path to run dir containing env_XX/ subdirs",
    )
    parser.add_argument(
        "--env_id",
        type=int,
        default=None,
        help="Only process this env id (default: all env_*)",
    )
    parser.add_argument("--window_sizes", type=str, default="30,60", help="Comma-separated logic seconds")
    parser.add_argument("--horizon", type=float, default=60.0, help="Future horizon H (logic seconds)")
    parser.add_argument("--score_threshold", type=float, default=DEFAULT_SCORE_THRESHOLD)
    parser.add_argument("--min_event_windows", type=int, default=DEFAULT_MIN_EVENT_WINDOWS)
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help="Output directory (default: <run_dir>/derived)",
    )
    args = parser.parse_args()

    window_sizes = [float(x) for x in args.window_sizes.split(",") if x.strip()]
    run_dir = args.run_dir.resolve()
    out_root = (args.out_dir or (run_dir / "derived")).resolve()

    env_dirs = _discover_env_dirs(run_dir, args.env_id)
    if not env_dirs:
        raise SystemExit(f"No env_* directories under {run_dir} (checked flat and episode_*/ layouts)")

    summaries = []
    for env_dir in env_dirs:
        out_dir = _derived_out_dir(out_root, run_dir, env_dir)
        print(f"[build] {env_dir} → {out_dir}")
        summary = process_env_dir(
            env_dir=env_dir,
            out_dir=out_dir,
            window_sizes=window_sizes,
            horizon=args.horizon,
            score_threshold=args.score_threshold,
            min_event_windows=args.min_event_windows,
        )
        summaries.append(summary)
        print(
            f"  episode_end={summary['episode_end_s']:.0f}s  "
            f"features={summary['n_feature_rows']}  "
            f"labels={summary['n_label_rows']}  "
            f"events={summary['n_events']}"
        )
        okpi = summary.get("order_kpi") or {}
        if okpi.get("n_completed"):
            print(
                f"  jobs={okpi.get('n_completed')}/{okpi.get('n_jobs')}  "
                f"makespan={okpi.get('order_makespan_s')}s  "
                f"mean_cycle={okpi.get('mean_cycle_time_s')}s  "
                f"throughput={okpi.get('throughput_jobs_per_hour')}/h"
            )
        for ps in summary["per_window_size"]:
            print(f"  ws={ps['window_size_s']}: hot_windows={ps['hot_windows']} top={ps['top_nodes'][:3]}")

    (out_root / "all_env_summary.json").write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"[done] outputs under {out_root}")


if __name__ == "__main__":
    main()
