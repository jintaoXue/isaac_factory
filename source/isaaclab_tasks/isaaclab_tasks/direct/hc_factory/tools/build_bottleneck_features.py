#!/usr/bin/env python3
"""Offline Phase-B pipeline: raw bottleneck tables → window features + labels.

    Reads a single-run directory produced by BottleneckDataCollector v0.6, e.g.::

        output/bottleneck_dataset/<run_id>/episode_00/env_00/

Writes::

    derived_phase_b_v2_3/window_feature_table.csv
    derived_phase_b_v2_3/bottleneck_label.csv
    derived_phase_b_v2_3/bottleneck_event.csv
    derived_phase_b_v2_3/label_metadata.json
    derived_phase_b_v2_3/job_kpi.csv              # per-job start / complete / cycle time
    derived_phase_b_v2_3/pipeline_summary.json    # includes order makespan & mean cycle

Usage::

    python tools/build_bottleneck_features.py \\
        --run_dir output/bottleneck_dataset/2026-07-18_17-46-35_seed42 \\
        --window_size 30 --stride 30 --horizon 120
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

DEFAULT_SCORE_THRESHOLD = 0.50
DEFAULT_MIN_EVENT_WINDOWS = 2
DEFAULT_RELATIVE_MARGIN = 0.10
SYSTEM_HISTORY_WINDOWS = 3
MIN_WIP_GROWTH = 1.0
MIN_THROUGHPUT_DROP_RATIO = 0.25
SYSTEM_LOOKBACK_S = 120.0
WARMUP_S = 120.0
MIN_BASELINE_OPERATIONS = 2
LABEL_VERSION = "bstan_weak_v2_3"
COLLECTOR_VERSION = "v0.6"
DERIVED_DIR_NAME = "derived_phase_b_v2_3"

SCORE_CONFIG = {
    "process": {
        "queue_length": 0.25,
        "avg_waiting_time": 0.20,
        "active_pct": 0.25,
        "current_active_duration": 0.10,
        "upstream_blocked_ratio": 0.10,
        "downstream_starved_ratio": 0.10,
    },
    "buffer": {
        "occupancy_ratio": 0.50,
        "queue_length": 0.30,
        "positive_queue_growth_rate": 0.20,
    },
    "label_gates": {
        "relative_margin": DEFAULT_RELATIVE_MARGIN,
        "system_history_windows": SYSTEM_HISTORY_WINDOWS,
        "min_wip_growth": MIN_WIP_GROWTH,
        "min_throughput_drop_ratio": MIN_THROUGHPUT_DROP_RATIO,
        "system_lookback_s": SYSTEM_LOOKBACK_S,
        "warmup_s": WARMUP_S,
        "min_baseline_operations": MIN_BASELINE_OPERATIONS,
        "min_event_windows": DEFAULT_MIN_EVENT_WINDOWS,
    },
}

MODEL_FEATURE_FIELDS = (
    "queue_length_s",
    "avg_waiting_time_s",
    "occupancy_ratio_s",
    "queue_growth_rate_s",
    "active_pct_s",
    "blocked_ratio_s",
    "starved_ratio_s",
    "current_active_duration_s",
    "output_rate_s",
    "transport_waiting_time_s",
    "route_delay_s",
    "material_shortage_flag_s",
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
    """Return collector-v0.6 env directories under episode_* folders."""
    env_dirs = sorted(run_dir.glob("episode_*/env_*"))
    if env_id is not None:
        env_dirs = [d for d in env_dirs if d.name == f"env_{env_id:02d}"]
    return env_dirs


def _derived_out_dir(out_root: Path, run_dir: Path, env_dir: Path) -> Path:
    """Mirror episode nesting under derived/, e.g. derived/episode_00/env_00/."""
    return out_root / env_dir.relative_to(run_dir)


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


def _time_s(
    row: dict[str, Any],
    logic_dt: float,
    logic_key: str = "logic_time_s",
    step_key: str = "time_step",
) -> float:
    """Read logic seconds, falling back to a logic-step field."""
    if row.get(logic_key) not in (None, ""):
        return _f(row.get(logic_key))
    return _f(row.get(step_key)) * logic_dt


def _canonical_node_id(node_id: Any) -> str:
    value = str(node_id or "").strip()
    if not value:
        return ""
    if value.startswith("storage_"):
        return value
    if "storage" in value.lower():
        return f"storage_{value}"
    return value


def _json_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _paired_disturbance_events(
    rows: list[dict[str, Any]], logic_dt: float, episode_end: float
) -> list[dict[str, Any]]:
    """Return paired runtime intervals; CONFIG rows never become active features."""
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        phase = str(row.get("event_phase") or "").upper()
        if phase not in {"CONFIG", "START", "END", "SYSTEM"}:
            raise ValueError(f"Invalid disturbance event_phase: {phase!r}")
        if phase in {"START", "END"}:
            event_id = str(row.get("disturbance_id") or "")
            if not event_id:
                raise ValueError("Runtime disturbance is missing disturbance_id")
            if phase in grouped[event_id]:
                raise ValueError(f"Duplicate disturbance {phase}: {event_id}")
            grouped[event_id][phase] = row

    events = []
    for event_id, phases in sorted(grouped.items()):
        if "START" not in phases or "END" not in phases:
            raise ValueError(f"Unpaired runtime disturbance: {event_id}")
        start_row, end_row = phases["START"], phases["END"]
        actual_start = _i(start_row.get("actual_start_time_step"))
        actual_end = _i(end_row.get("actual_end_time_step"))
        target = str(start_row.get("actual_target_resource_id") or "")
        end_target = str(end_row.get("actual_target_resource_id") or "")
        if actual_start is None or actual_end is None:
            raise ValueError(f"Disturbance lacks actual interval: {event_id}")
        if not target or end_target != target:
            raise ValueError(f"Disturbance target mismatch: {event_id}")
        start = actual_start * logic_dt
        end = actual_end * logic_dt
        if end < start or end > episode_end:
            raise ValueError(f"Invalid disturbance interval: {event_id}={start}:{end}")
        events.append(
            {
                "event_id": event_id,
                "start": start,
                "end": end,
                "type": str(start_row.get("disturbance_type") or ""),
                "target": target,
            }
        )
    return events


def build_timelines(
    events: list[dict[str, Any]], episode_end: float, logic_dt: float = 1.0
) -> dict[str, ResourceTimeline]:
    """Convert event log into contiguous state intervals per resource."""
    by_rid: dict[str, list[dict]] = defaultdict(list)
    for e in events:
        by_rid[e["resource_id"]].append(e)

    timelines: dict[str, ResourceTimeline] = {}
    for rid, evs in by_rid.items():
        evs = sorted(evs, key=lambda x: _time_s(x, logic_dt))
        rtype = evs[0].get("resource_type", "unknown")
        intervals: list[Interval] = []
        # Assume IDLE before first event
        t0 = 0.0
        state = "IDLE"
        for e in evs:
            t = _time_s(e, logic_dt)
            if t > t0:
                intervals.append(Interval(t0, t, state))
            state = e.get("to_state") or state
            t0 = t
        if episode_end > t0:
            intervals.append(Interval(t0, episode_end, state))
        timelines[rid] = ResourceTimeline(rid, rtype, intervals)
    return timelines


def _overlap_duration(
    intervals: list[Interval], w0: float, w1: float, states: frozenset[str]
) -> float:
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
        if iv.start <= t < iv.end or (
            i == len(intervals) - 1 and iv.start <= t <= iv.end
        ):
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
    disturbance_rows: list[dict],
    episode_config: dict[str, Any],
    window_size: float,
    stride: float,
    episode_end: float,
    run_id: str,
    env_id: int,
    episode_id: int,
    logic_dt: float,
) -> list[dict]:
    # Precompute job waiting intervals keyed by station
    wait_by_station: dict[str, list[tuple[float, float]]] = defaultdict(list)
    departures_by_station: dict[str, list[float]] = defaultdict(list)
    open_queue: dict[tuple[int, str], tuple[float, str]] = {}
    job_lifetimes: dict[int, dict[str, float]] = defaultdict(dict)
    completed_operations: list[float] = []

    for row in sorted(job_rows, key=lambda r: _time_s(r, logic_dt)):
        event = row.get("event")
        job_id = _i(row.get("job_id"), -1)
        task = row.get("task") or ""
        task_type = row.get("task_type") or ""
        station = _canonical_node_id(row.get("station_id")) or "unknown"
        t = _time_s(row, logic_dt)
        key = (job_id if job_id is not None else -1, task)
        if job_id is not None and job_id >= 0 and event == "job_selected":
            job_lifetimes[job_id].setdefault("start", t)
        elif job_id is not None and job_id >= 0 and event == "stage_complete":
            job_lifetimes[job_id]["end"] = t
        if event == "queue_enter":
            open_queue[key] = (t, station)
        elif event == "queue_leave":
            if key in open_queue:
                t0, st = open_queue.pop(key)
                wait_by_station[st].append((t0, t))
            else:
                qe = _f(row.get("queue_enter_time_step"), t / logic_dt) * logic_dt
                wait_by_station[station].append((qe, t))
        elif event == "departure":
            departures_by_station[station].append(t)
            if task_type == "processing":
                completed_operations.append(t)

    # Close open queues at episode end
    for key, (t0, st) in open_queue.items():
        wait_by_station[st].append((t0, episode_end))

    # Buffer occupancy time series (step events)
    buffer_occ: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    for row in sorted(buffer_rows, key=lambda r: _time_s(r, logic_dt)):
        bid = _canonical_node_id(row.get("buffer_id"))
        t = _time_s(row, logic_dt)
        occ = _f(row.get("occupancy"))
        ratio = _f(row.get("occupancy_ratio"))
        buffer_occ[bid].append((t, occ, ratio))

    def buffer_stats(bid: str, w0: float, w1: float) -> tuple[float, float, float]:
        series = sorted(buffer_occ.get(bid, []))
        if not series:
            return 0.0, 0.0, 0.0
        before = [s for s in series if s[0] <= w0]
        current = before[-1] if before else series[0]
        start_occ = current[1]
        prev_t = w0
        occ_area = 0.0
        ratio_area = 0.0
        for point in (s for s in series if w0 < s[0] < w1):
            duration = point[0] - prev_t
            occ_area += current[1] * duration
            ratio_area += current[2] * duration
            current = point
            prev_t = point[0]
        duration = w1 - prev_t
        occ_area += current[1] * duration
        ratio_area += current[2] * duration
        end_points = [s for s in series if s[0] <= w1]
        end_occ = end_points[-1][1] if end_points else start_occ
        mean_occ = occ_area / (w1 - w0)
        mean_ratio = ratio_area / (w1 - w0)
        growth = (end_occ - start_occ) / window_size if window_size else 0.0
        return mean_occ, mean_ratio, growth

    transport_intervals: list[dict[str, Any]] = []
    for row in transport_rows:
        req = _f(row.get("request_time_step")) * logic_dt
        pickup = _f(row.get("pickup_time_step"), req / logic_dt) * logic_dt
        end = (
            _f(row.get("transport_end_time_step"), _f(row.get("dropoff_time_step")))
            * logic_dt
        )
        if end <= 0:
            continue
        targets = {
            _canonical_node_id(row.get("carrier_id")),
            _canonical_node_id(row.get("from_node")),
            _canonical_node_id(row.get("to_node")),
        }
        transport_intervals.append(
            {
                "start": req,
                "end": end,
                "waiting": max(pickup - req, 0.0),
                "delay": max(end - req, 0.0),
                "targets": {target for target in targets if target},
            }
        )

    process_cfg = _json_dict(episode_config.get("process_time_config"))
    buffer_cfg = _json_dict(episode_config.get("buffer_capacity_config"))
    material_targets: dict[str, set[str]] = defaultdict(set)
    for raw_bid, info in buffer_cfg.items():
        if not isinstance(info, dict):
            continue
        for material in info.get("supporting_materials", []) or []:
            material_targets[str(material)].add(_canonical_node_id(raw_bid))
    for product_cfg in process_cfg.values():
        if not isinstance(product_cfg, dict):
            continue
        for task_cfg in product_cfg.values():
            if not isinstance(task_cfg, dict):
                continue
            machine = str(task_cfg.get("machine") or "")
            candidates = {
                rid
                for rid in timelines
                if machine and (rid == machine or rid.startswith(f"{machine}_ws"))
            }
            for material in task_cfg.get("required_materials", []) or []:
                material_targets[str(material)].update(candidates)

    shortage_samples: list[tuple[float, bool, set[str]]] = []
    for row in material_rows:
        material = str(row.get("material_type") or row.get("material_id") or "")
        targets = set(material_targets.get(material, set()))
        location = _canonical_node_id(row.get("storage_location"))
        if location:
            targets.add(location)
        shortage_samples.append(
            (
                _time_s(row, logic_dt),
                str(row.get("shortage_flag")) in ("1", "True", "true"),
                targets,
            )
        )

    disturbance_events = _paired_disturbance_events(
        disturbance_rows, logic_dt, episode_end
    )
    disturbance_intervals = [
        (event["start"], event["end"]) for event in disturbance_events
    ]
    scenario_disturbance_dim = str(episode_config.get("disturbance_dim") or "none")
    scenario_disturbance_intensity = _f(
        episode_config.get("disturbance_intensity"), 0.0
    )

    # Ensure buffer resources appear even without resource events
    for bid in buffer_occ:
        if bid not in timelines:
            timelines[bid] = ResourceTimeline(
                bid, "buffer", [Interval(0.0, episode_end, "IDLE")]
            )

    if window_size <= 0 or stride <= 0:
        raise ValueError("window_size and stride must be positive")
    n_windows = max(int(math.floor((episode_end - window_size) / stride)) + 1, 0)
    rows_out: list[dict] = []

    for wi in range(n_windows):
        w0 = wi * stride
        w1 = w0 + window_size
        wlen = window_size

        # Global blocked/starved mass for upstream/downstream proxies
        total_blocked = 0.0
        total_starved = 0.0
        for tl in timelines.values():
            if tl.resource_type == "buffer":
                continue
            total_blocked += _overlap_duration(tl.intervals, w0, w1, BLOCKED_STATES)
            total_starved += _overlap_duration(tl.intervals, w0, w1, STARVED_STATES)
        n_non_buf = max(
            sum(1 for tl in timelines.values() if tl.resource_type != "buffer"), 1
        )
        global_blocked_ratio = total_blocked / (n_non_buf * wlen)
        global_starved_ratio = total_starved / (n_non_buf * wlen)
        total_wip = sum(
            1
            for life in job_lifetimes.values()
            if life.get("start", float("inf")) < w1
            and life.get("end", float("inf")) >= w0
        )
        wip_at_window_end = sum(
            1
            for life in job_lifetimes.values()
            if life.get("start", float("inf")) < w1
            and life.get("end", float("inf")) > w1
        )
        window_completed_operations = sum(
            1 for completed_at in completed_operations if w0 <= completed_at < w1
        )
        operation_throughput_rolling = window_completed_operations / wlen
        states_at_end = [_state_at(tl.intervals, w1) for tl in timelines.values()]
        num_busy = sum(state in ACTIVE_STATES for state in states_at_end)
        num_blocked = sum(state in BLOCKED_STATES for state in states_at_end)
        num_starved = sum(state in STARVED_STATES for state in states_at_end)
        disturbance_flag = int(
            any(end >= w0 and start < w1 for start, end in disturbance_intervals)
        )

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
            wait_lens = [
                max(min(b, w1) - max(a, w0), 0.0) for a, b in waits if b > w0 and a < w1
            ]
            avg_wait = statistics.mean(wait_lens) if wait_lens else 0.0
            queue_length = float(sum(1 for a, b in waits if a < w1 and b > w0))

            deps = [d for d in departures_by_station.get(rid, []) if w0 <= d < w1]
            output_rate = len(deps) / wlen
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

            local_transport = [
                item
                for item in transport_intervals
                if rid in item["targets"] and item["end"] >= w0 and item["start"] < w1
            ]
            transport_waiting = (
                statistics.mean(item["waiting"] for item in local_transport)
                if local_transport
                else 0.0
            )
            route_delay = (
                statistics.mean(item["delay"] for item in local_transport)
                if local_transport
                else 0.0
            )

            local_shortages = [
                flag
                for t, flag, targets in shortage_samples
                if rid in targets and w0 <= t < w1
            ]
            material_shortage = (
                sum(1 for flag in local_shortages if flag) / len(local_shortages)
                if local_shortages
                else 0.0
            )

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
                    "episode_id": episode_id,
                    "window_index": wi,
                    "window_start_step": int(round(w0 / logic_dt)),
                    "window_end_step": int(round(w1 / logic_dt)),
                    "window_start_s": w0,
                    "window_end_s": w1,
                    "window_size_s": window_size,
                    "stride_s": stride,
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
                    "blocked_ratio_s": round(blocked_time / wlen, 6),
                    "starved_ratio_s": round(starved_time / wlen, 6),
                    "output_rate_s": round(output_rate, 6),
                    "inter_departure_var_s": round(inter_dep_var, 6),
                    "upstream_blocked_ratio_s": round(upstream_blocked, 6),
                    "downstream_starved_ratio_s": round(downstream_starved, 6),
                    "transport_waiting_time_s": round(transport_waiting, 6),
                    "route_delay_s": round(route_delay, 6),
                    "material_shortage_flag_s": round(material_shortage, 6),
                    "total_WIP": wip_at_window_end,
                    "wip_overlap_count": total_wip,
                    "wip_at_window_end": wip_at_window_end,
                    "operation_throughput_rolling": round(
                        operation_throughput_rolling, 6
                    ),
                    "completed_operations_in_window": window_completed_operations,
                    "num_busy_resources": num_busy,
                    "num_blocked_resources": num_blocked,
                    "num_starved_resources": num_starved,
                    "scenario_disturbance_dim": scenario_disturbance_dim,
                    "scenario_disturbance_intensity": scenario_disturbance_intensity,
                    "runtime_disturbance_active": disturbance_flag,
                    "disturbance_flag": 0,
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
    """Add resource-type-aware weak scores normalized within each window/category."""
    by_window: dict[tuple, list[dict]] = defaultdict(list)
    for r in feature_rows:
        key = (
            r["run_id"],
            r["env_id"],
            r["episode_id"],
            r["window_size_s"],
            r["stride_s"],
            r["window_index"],
        )
        by_window[key].append(r)

    for rows in by_window.values():
        categories = {
            "buffer": [
                r
                for r in rows
                if r["resource_type"] == "buffer"
                or r["resource_id"].startswith("storage_")
            ],
            "process": [
                r
                for r in rows
                if r["resource_type"] != "buffer"
                and not r["resource_id"].startswith("storage_")
            ],
        }
        for category, category_rows in categories.items():
            q = _norm_across([r["queue_length_s"] for r in category_rows])
            growth = _norm_across(
                [max(r["queue_growth_rate_s"], 0.0) for r in category_rows]
            )
            w = _norm_across([r["avg_waiting_time_s"] for r in category_rows])
            d = _norm_across([r["current_active_duration_s"] for r in category_rows])
            for i, r in enumerate(category_rows):
                if category == "buffer":
                    score = (
                        0.50 * r["occupancy_ratio_s"] + 0.30 * q[i] + 0.20 * growth[i]
                    )
                else:
                    score = (
                        W_QUEUE * q[i]
                        + W_WAIT * w[i]
                        + W_ACTIVE * r["active_pct_s"]
                        + W_ACTIVE_DUR * d[i]
                        + W_UPSTREAM * r["upstream_blocked_ratio_s"]
                        + W_DOWNSTREAM * r["downstream_starved_ratio_s"]
                    )
                r["bottleneck_score_s"] = round(min(max(score, 0.0), 1.0), 6)
                r["norm_queue_length_s"] = round(q[i], 6)
                r["norm_avg_waiting_time_s"] = round(w[i], 6)
                r["norm_current_active_duration_s"] = round(d[i], 6)
                r["norm_positive_queue_growth_rate_s"] = round(growth[i], 6)
    return feature_rows


def _node_category(row: dict[str, Any]) -> str:
    if row["resource_type"] == "buffer" or row["resource_id"].startswith("storage_"):
        return "buffer"
    return "process"


def _symptom_type(row: dict[str, Any]) -> str:
    if _f(row.get("blocked_time_s")) > _f(row.get("starved_time_s")):
        return "blocked_downstream"
    if _f(row.get("starved_time_s")) > 0:
        return "starved_upstream"
    if _f(row.get("queue_growth_rate_s")) > 0:
        return "queue_growth"
    if _f(row.get("queue_length_s")) > 0 or _f(row.get("avg_waiting_time_s")) > 0:
        return "queue_buildup"
    if _f(row.get("active_pct_s")) >= 0.8:
        return "high_utilization"
    return "system_pressure"


def _candidate_cause(
    event: dict[str, Any], disturbance_events: list[dict[str, Any]], horizon: float
) -> dict[str, Any]:
    candidates = [
        item
        for item in disturbance_events
        if item["start"] <= event["start_s"]
        and event["start_s"] - item["start"] <= horizon
    ]
    if not candidates:
        return {
            "candidate_cause_type": "",
            "cause_target_resource_id": "",
            "cause_label_confidence": "",
            "disturbance_to_bottleneck_delay_s": "",
        }
    cause = max(candidates, key=lambda item: item["start"])
    delay = event["start_s"] - cause["start"]
    overlaps = cause["end"] >= event["start_s"]
    if cause["target"] == event["resource_id"]:
        confidence = 1.0
    elif overlaps:
        confidence = 0.6
    else:
        confidence = 0.3
    return {
        "candidate_cause_type": cause["type"],
        "cause_target_resource_id": cause["target"],
        "cause_label_confidence": confidence,
        "disturbance_to_bottleneck_delay_s": round(delay, 6),
    }


def build_labels_and_events(
    feature_rows: list[dict],
    horizon: float,
    score_threshold: float,
    min_event_windows: int,
    episode_end: float,
    disturbance_rows: list[dict[str, Any]] | None = None,
    logic_dt: float = 1.0,
) -> tuple[list[dict], list[dict]]:
    """Build weak bottleneck events and right-censored future labels."""
    by_ws: dict[tuple[float, float], list[dict]] = defaultdict(list)
    for r in feature_rows:
        by_ws[(r["window_size_s"], r["stride_s"])].append(r)

    label_rows: list[dict] = []
    event_rows: list[dict] = []

    score_config_json = json.dumps(SCORE_CONFIG, sort_keys=True, separators=(",", ":"))
    impact_hold_windows = max(min_event_windows, 1)
    disturbance_events = _paired_disturbance_events(
        disturbance_rows or [], logic_dt, episode_end
    )
    for (window_size, stride), rows in by_ws.items():
        # Per window: argmax resource
        windows: dict[int, list[dict]] = defaultdict(list)
        for r in rows:
            windows[r["window_index"]].append(r)

        window_meta: dict[int, dict] = {}
        system_history: list[dict[str, Any]] = []
        impact_history: list[list[str]] = []
        for wi, rs in sorted(windows.items()):
            target_rows = [row for row in rs if _node_category(row) == "process"]
            if not target_rows:
                raise ValueError(f"No productive resource candidates in window {wi}")
            best = max(target_rows, key=lambda x: x["bottleneck_score_s"])
            peers = sorted(
                (row["bottleneck_score_s"] for row in target_rows),
                reverse=True,
            )
            relative_margin = peers[0] - peers[1] if len(peers) > 1 else peers[0]
            point_wip = _f(rs[0].get("wip_at_window_end"))
            current_completed_operations = _i(
                rs[0].get("completed_operations_in_window")
            )
            if current_completed_operations is None:
                raise ValueError("completed_operations_in_window is required")
            current_system = {
                "point_wip": point_wip,
                "completed_operations": current_completed_operations,
            }

            point_history = system_history[-SYSTEM_HISTORY_WINDOWS:]
            point_wip_baseline = (
                statistics.mean(item["point_wip"] for item in point_history)
                if len(point_history) == SYSTEM_HISTORY_WINDOWS
                else point_wip
            )
            point_wip_growth = point_wip - point_wip_baseline

            lookback_windows = max(int(math.ceil(SYSTEM_LOOKBACK_S / stride)), 1)
            timeline = system_history + [current_system]
            recent_period = timeline[-lookback_windows:]
            baseline_period = timeline[-2 * lookback_windows : -lookback_windows]
            complete_periods = (
                len(recent_period) == lookback_windows
                and len(baseline_period) == lookback_windows
            )
            recent_completed = sum(
                item["completed_operations"] for item in recent_period
            )
            baseline_completed = sum(
                item["completed_operations"] for item in baseline_period
            )
            throughput_support_available = int(
                complete_periods and baseline_completed >= MIN_BASELINE_OPERATIONS
            )
            smoothed_throughput_drop = (
                max((baseline_completed - recent_completed) / baseline_completed, 0.0)
                if throughput_support_available
                else 0.0
            )
            raw_impact_reasons = []
            if point_wip_growth >= MIN_WIP_GROWTH:
                raw_impact_reasons.append("wip_growth")
            if smoothed_throughput_drop >= MIN_THROUGHPUT_DROP_RATIO:
                raw_impact_reasons.append("operation_throughput_drop")
            impact_history.append(raw_impact_reasons)
            held_impact_period = impact_history[-impact_hold_windows:]
            impact_reasons = list(
                dict.fromkeys(
                    reason for reasons in held_impact_period for reason in reasons
                )
            )
            impact_age_windows = next(
                (
                    age
                    for age, reasons in enumerate(reversed(held_impact_period))
                    if reasons
                ),
                -1,
            )
            system_impact = int(bool(impact_reasons))

            absolute_gate = best["bottleneck_score_s"] >= score_threshold
            relative_gate = relative_margin >= DEFAULT_RELATIVE_MARGIN
            warmup_gate = int(_f(rs[0].get("window_start_s")) >= WARMUP_S)
            context_gate = warmup_gate
            is_hot = absolute_gate and relative_gate and system_impact and context_gate
            window_meta[wi] = {
                "window_index": wi,
                "window_start_s": rs[0]["window_start_s"],
                "window_end_s": rs[0]["window_end_s"],
                "window_size_s": window_size,
                "stride_s": stride,
                "run_id": rs[0]["run_id"],
                "env_id": rs[0]["env_id"],
                "episode_id": rs[0]["episode_id"],
                "window_start_step": rs[0]["window_start_step"],
                "window_end_step": rs[0]["window_end_step"],
                "bottleneck_node_t": best["resource_id"],
                "bottleneck_type_t": best["resource_type"],
                "bottleneck_score_t": best["bottleneck_score_s"],
                "relative_score_margin_t": round(relative_margin, 6),
                "wip_growth_t": round(point_wip_growth, 6),
                "throughput_drop_ratio_t": round(smoothed_throughput_drop, 6),
                "baseline_completed_operations_t": baseline_completed,
                "recent_completed_operations_t": recent_completed,
                "throughput_support_available_t": throughput_support_available,
                "system_impact_raw_flag_t": int(bool(raw_impact_reasons)),
                "system_impact_raw_reason_t": "+".join(raw_impact_reasons),
                "system_impact_flag_t": system_impact,
                "system_impact_reason_t": "+".join(impact_reasons),
                "system_impact_age_windows_t": impact_age_windows,
                "warmup_gate_t": warmup_gate,
                "label_context_gate_t": context_gate,
                "bottleneck_symptom_type_t": _symptom_type(best),
                "is_bottleneck_window": int(is_hot),
            }
            system_history.append(current_system)

        # Merge consecutive high-score windows into events
        events: list[dict] = []
        cur: dict | None = None
        for wi in sorted(window_meta):
            meta = window_meta[wi]
            hot = meta["is_bottleneck_window"]
            node = meta["bottleneck_node_t"]
            if hot:
                if (
                    cur
                    and cur["resource_id"] == node
                    and wi == cur["end_window_index"] + 1
                ):
                    cur["end_window_index"] = wi
                    cur["end_s"] = meta["window_end_s"]
                    cur["duration_s"] = cur["end_s"] - cur["start_s"]
                    cur["max_score"] = max(cur["max_score"], meta["bottleneck_score_t"])
                    cur["n_windows"] += 1
                else:
                    if cur and cur["n_windows"] >= min_event_windows:
                        cur["duration_observed"] = 1
                        events.append(cur)
                    cur = {
                        "run_id": meta["run_id"],
                        "env_id": meta["env_id"],
                        "episode_id": meta["episode_id"],
                        "window_size_s": window_size,
                        "stride_s": stride,
                        "resource_id": node,
                        "resource_type": meta["bottleneck_type_t"],
                        "start_window_index": wi,
                        "end_window_index": wi,
                        "start_s": meta["window_start_s"],
                        "end_s": meta["window_end_s"],
                        "duration_s": meta["window_end_s"] - meta["window_start_s"],
                        "max_score": meta["bottleneck_score_t"],
                        "n_windows": 1,
                        "bottleneck_symptom_type": meta["bottleneck_symptom_type_t"],
                    }
            else:
                if cur and cur["n_windows"] >= min_event_windows:
                    cur["duration_observed"] = 1
                    events.append(cur)
                cur = None
        if cur and cur["n_windows"] >= min_event_windows:
            cur["duration_observed"] = 0
            events.append(cur)

        for i, ev in enumerate(events):
            ev["event_id"] = i
            ev["severity_weak"] = round(
                0.7 * ev["max_score"] + 0.3 * min(ev["duration_s"] / horizon, 1.0), 6
            )
            ev["label_version"] = LABEL_VERSION
            ev["score_threshold"] = score_threshold
            ev["relative_score_margin"] = DEFAULT_RELATIVE_MARGIN
            ev["min_event_windows"] = min_event_windows
            ev.update(_candidate_cause(ev, disturbance_events, horizon))
            event_rows.append(ev)

        # Future labels per window
        for wi, meta in sorted(window_meta.items()):
            anchor = meta["window_end_s"]
            label_observed = int(anchor + horizon <= episode_end + 1e-9)
            future = []
            if label_observed:
                future = [
                    ev
                    for ev in events
                    if ev["start_s"] > anchor and ev["start_s"] <= anchor + horizon
                ]
            if label_observed and future:
                first = min(future, key=lambda e: e["start_s"])
                will = 1
                fut_id = first["resource_id"]
                fut_type = first["resource_type"]
                tts = first["start_s"] - anchor
                dur = first["duration_s"]
                duration_observed = first["duration_observed"]
                severity = first["severity_weak"]
                symptom_type = first["bottleneck_symptom_type"]
                cause_type = first["candidate_cause_type"]
                cause_target = first["cause_target_resource_id"]
                cause_confidence = first["cause_label_confidence"]
                cause_delay = first["disturbance_to_bottleneck_delay_s"]
            elif label_observed:
                will = 0
                fut_id = ""
                fut_type = ""
                tts = ""
                dur = ""
                duration_observed = ""
                severity = ""
                symptom_type = ""
                cause_type = ""
                cause_target = ""
                cause_confidence = ""
                cause_delay = ""
            else:
                will = ""
                fut_id = ""
                fut_type = ""
                tts = ""
                dur = ""
                duration_observed = 0
                severity = ""
                symptom_type = ""
                cause_type = ""
                cause_target = ""
                cause_confidence = ""
                cause_delay = ""

            # Heuristic root cause without disturbance_log
            reason = ""
            if meta["is_bottleneck_window"]:
                node_rows = [
                    r
                    for r in windows[wi]
                    if r["resource_id"] == meta["bottleneck_node_t"]
                ]
                if node_rows:
                    nr = node_rows[0]
                    if (
                        nr["blocked_time_s"] >= nr["starved_time_s"]
                        and nr["blocked_time_s"] > 0
                    ):
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
                    "prediction_horizon": horizon,
                    "anchor_time_s": anchor,
                    "will_bottleneck": will,
                    "future_bottleneck_object_id": fut_id,
                    "future_bottleneck_type": fut_type,
                    "time_to_start": tts,
                    "duration": dur,
                    "severity_weak": severity,
                    "label_version": LABEL_VERSION,
                    "score_config": score_config_json,
                    "score_threshold": score_threshold,
                    "min_event_windows": min_event_windows,
                    "label_observed": label_observed,
                    "duration_observed": duration_observed,
                    "root_cause_reason": reason,
                    "bottleneck_symptom_type": symptom_type,
                    "candidate_cause_type": cause_type,
                    "cause_target_resource_id": cause_target,
                    "cause_label_confidence": cause_confidence,
                    "disturbance_to_bottleneck_delay_s": cause_delay,
                }
            )

    return label_rows, event_rows


def validate_phase_b_outputs(feature_rows: list[dict], label_rows: list[dict]) -> None:
    """Fail fast on Phase-B key, numeric, and censoring regressions."""
    feature_keys = [
        (
            row["run_id"],
            row["env_id"],
            row["episode_id"],
            row["window_size_s"],
            row["stride_s"],
            row["window_index"],
            row["resource_id"],
        )
        for row in feature_rows
    ]
    if len(feature_keys) != len(set(feature_keys)):
        raise ValueError("Duplicate Phase-B feature primary key")

    label_keys = [
        (
            row["run_id"],
            row["env_id"],
            row["episode_id"],
            row["window_size_s"],
            row["stride_s"],
            row["window_index"],
        )
        for row in label_rows
    ]
    if len(label_keys) != len(set(label_keys)):
        raise ValueError("Duplicate Phase-B label primary key")

    for row in feature_rows:
        for feature_name in MODEL_FEATURE_FIELDS:
            value = float(row[feature_name])
            if not math.isfinite(value):
                raise ValueError(
                    f"Non-finite feature {feature_name} for {row['resource_id']}"
                )
    for row in label_rows:
        if row["label_observed"] == 0 and row["will_bottleneck"] != "":
            raise ValueError("Censored label must not be encoded as a negative sample")
        if row["label_observed"] == 1 and row["will_bottleneck"] not in (0, 1):
            raise ValueError("Observed label must be binary")


def _write_csv(
    path: Path, rows: list[dict], fieldnames: list[str] | None = None
) -> None:
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
    logic_dt: float = 1.0,
) -> tuple[list[dict], dict]:
    """Per-job start/complete/cycle from job_trace + order-level throughput KPIs.

    Definitions (logic seconds, logic_dt=1 → same as time_step):
      - start: first ``job_selected`` for this job_id
      - complete: ``stage_complete`` (product enters progress["finished"])
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
    for jid in sorted(
        by_job.keys(),
        key=lambda x: int(float(x)) if str(x).replace(".", "", 1).isdigit() else str(x),
    ):
        evs = by_job[jid]
        product_type = next(
            (e.get("product_type") or "" for e in evs if e.get("product_type")), ""
        )
        starts = [_time_s(e, logic_dt) for e in evs if e.get("event") == "job_selected"]
        completes = [
            _time_s(e, logic_dt) for e in evs if e.get("event") == "stage_complete"
        ]
        start_s = min(starts) if starts else None
        if completes:
            complete_s = min(completes)
            complete_source = "stage_complete"
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
        "last_job_complete_s": round(max(complete_times), 3)
        if complete_times
        else None,
        "mean_cycle_time_s": round(statistics.mean(cycles), 3) if cycles else None,
        "median_cycle_time_s": round(statistics.median(cycles), 3) if cycles else None,
        "std_cycle_time_s": round(statistics.pstdev(cycles), 3)
        if len(cycles) >= 2
        else (0.0 if cycles else None),
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
    stride: float | None,
    horizon: float,
    score_threshold: float,
    min_event_windows: int,
) -> dict:
    events = _read_jsonl(env_dir / "resource_event_log.jsonl")
    job_rows = _read_csv(env_dir / "job_trace.csv")
    buffer_rows = _read_csv(env_dir / "buffer_event_log.csv")
    transport_rows = _read_csv(env_dir / "route_transport_task.csv")
    material_rows = _read_csv(env_dir / "material_inventory_log.csv")
    disturbance_rows = _read_csv(env_dir / "disturbance_log.csv")
    lifecycle_rows = _read_csv(env_dir / "episode_lifecycle.csv")
    ep_rows = _read_csv(env_dir / "episode_config.csv")

    if len(ep_rows) != 1:
        raise ValueError(
            f"Expected one episode_config row in {env_dir}, got {len(ep_rows)}"
        )
    if ep_rows[0].get("collector_version") != COLLECTOR_VERSION:
        raise ValueError(
            f"Expected collector {COLLECTOR_VERSION!r} in {env_dir}, "
            f"got {ep_rows[0].get('collector_version')!r}"
        )

    run_id = ep_rows[0].get("run_id") or ""
    env_id = _i(ep_rows[0].get("env_id"))
    episode_id = _i(ep_rows[0].get("episode_id"))
    if not run_id or env_id is None or episode_id is None:
        raise ValueError(f"Invalid run/env/episode identity in {env_dir}")

    episode_config: dict[str, Any] = ep_rows[0]
    logic_dt = _f(episode_config.get("logic_dt"))
    if logic_dt <= 0:
        raise ValueError(f"Invalid logic_dt in {env_dir}: {logic_dt}")

    aborted_rows = [
        row for row in lifecycle_rows if str(row.get("event", "")).upper() == "ABORTED"
    ]
    if aborted_rows:
        aborted = aborted_rows[-1]
        summary = {
            "status": "skipped",
            "skip_reason": "aborted_episode",
            "termination_event": "ABORTED",
            "termination_reason": aborted.get("termination_reason") or "unknown",
            "run_id": run_id,
            "env_id": env_id,
            "episode_id": episode_id,
            "abort_time_s": _time_s(aborted, logic_dt),
            "completed_jobs": _i(aborted.get("completed_jobs"), 0),
        }
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "pipeline_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return summary

    times = []
    for e in events:
        times.append(_time_s(e, logic_dt))
    for r in job_rows:
        times.append(_time_s(r, logic_dt))
    lifecycle_ends = [
        _time_s(row, logic_dt)
        for row in lifecycle_rows
        if str(row.get("event", "")).upper() == "END"
    ]
    episode_end = (
        max(lifecycle_ends) if lifecycle_ends else (max(times) if times else 0.0)
    )
    if episode_end <= 0:
        raise RuntimeError(f"No usable timestamps in {env_dir}")

    timelines = build_timelines(events, episode_end, logic_dt)

    all_features: list[dict] = []
    for ws in window_sizes:
        current_stride = stride if stride is not None else ws
        feats = compute_window_features(
            timelines=dict(timelines),  # copy ids; buffers may be added
            job_rows=job_rows,
            buffer_rows=buffer_rows,
            transport_rows=transport_rows,
            material_rows=material_rows,
            disturbance_rows=disturbance_rows,
            episode_config=episode_config,
            window_size=ws,
            stride=current_stride,
            episode_end=episode_end,
            run_id=run_id,
            env_id=env_id if env_id is not None else 0,
            episode_id=episode_id if episode_id is not None else 0,
            logic_dt=logic_dt,
        )
        all_features.extend(feats)

    all_features = add_bottleneck_scores(all_features)
    if not all_features:
        raise RuntimeError(
            f"No complete windows in {env_dir}; episode_end={episode_end}, window_sizes={window_sizes}"
        )
    labels, event_rows = build_labels_and_events(
        all_features,
        horizon,
        score_threshold,
        min_event_windows,
        episode_end,
        disturbance_rows=disturbance_rows,
        logic_dt=logic_dt,
    )
    job_kpi_rows, order_kpi = build_job_kpis(
        job_rows,
        run_id=run_id,
        env_id=env_id if env_id is not None else 0,
        episode_id=episode_id,
        logic_dt=logic_dt,
    )
    validate_phase_b_outputs(all_features, labels)

    _write_csv(out_dir / "window_feature_table.csv", all_features)
    _write_csv(out_dir / "bottleneck_label.csv", labels)
    _write_csv(out_dir / "bottleneck_event.csv", event_rows)
    label_metadata = {
        "label_version": LABEL_VERSION,
        "score_config": SCORE_CONFIG,
        "score_threshold": score_threshold,
        "relative_score_margin": DEFAULT_RELATIVE_MARGIN,
        "system_impact_config": {
            "history_windows": SYSTEM_HISTORY_WINDOWS,
            "min_wip_growth": MIN_WIP_GROWTH,
            "min_throughput_drop_ratio": MIN_THROUGHPUT_DROP_RATIO,
            "system_lookback_s": SYSTEM_LOOKBACK_S,
            "warmup_s": WARMUP_S,
            "min_baseline_operations": MIN_BASELINE_OPERATIONS,
            "impact_hold_windows": min_event_windows,
            "target_node_category": "process",
        },
        "min_event_windows": min_event_windows,
        "prediction_horizon": horizon,
        "window_sizes": window_sizes,
        "strides_s": {
            str(ws): stride if stride is not None else ws for ws in window_sizes
        },
        "logic_dt": logic_dt,
    }
    (out_dir / "label_metadata.json").write_text(
        json.dumps(label_metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
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
    feature_node_ids = {row["resource_id"] for row in all_features}
    resource_event_node_ids = set(timelines)
    buffer_node_ids = {
        node_id for node_id in feature_node_ids if node_id.startswith("storage_")
    }
    top_nodes = []
    for ws in window_sizes:
        ws_labels = [l for l in labels if l["window_size_s"] == ws]
        hot = [l for l in ws_labels if l["is_bottleneck_window"]]
        node_counts = defaultdict(int)
        for l in hot:
            node_counts[l["bottleneck_node_t"]] += 1
        top = sorted(node_counts.items(), key=lambda x: -x[1])[:5]
        top_nodes.append(
            {"window_size_s": ws, "hot_windows": len(hot), "top_nodes": top}
        )

    summary = {
        "status": "processed",
        "label_version": LABEL_VERSION,
        "run_id": run_id,
        "env_id": env_id,
        "episode_id": episode_id,
        "episode_end_s": episode_end,
        "n_resources": len(feature_node_ids),
        "n_resource_event_nodes": len(resource_event_node_ids),
        "n_buffer_nodes": len(buffer_node_ids),
        "n_feature_rows": len(all_features),
        "n_label_rows": len(labels),
        "n_events": len(event_rows),
        "n_job_kpi_rows": len(job_kpi_rows),
        "order_kpi": order_kpi,
        "window_sizes": window_sizes,
        "strides_s": {
            str(ws): stride if stride is not None else ws for ws in window_sizes
        },
        "horizon_s": horizon,
        "score_threshold": score_threshold,
        "min_event_windows": min_event_windows,
        "per_window_size": top_nodes,
        "will_bottleneck_rate": {
            str(ws): (
                sum(
                    1
                    for l in labels
                    if l["window_size_s"] == ws
                    and l["label_observed"] == 1
                    and l["will_bottleneck"] == 1
                )
                / max(
                    sum(
                        1
                        for l in labels
                        if l["window_size_s"] == ws and l["label_observed"] == 1
                    ),
                    1,
                )
            )
            for ws in window_sizes
        },
        "threshold_sensitivity": {
            f"{threshold:.2f}": (
                sum(
                    1
                    for label in labels
                    if label["bottleneck_score_t"] >= threshold
                    and label["relative_score_margin_t"] >= DEFAULT_RELATIVE_MARGIN
                    and label["system_impact_flag_t"] == 1
                    and label["label_context_gate_t"] == 1
                )
                / max(len(labels), 1)
            )
            for threshold in (
                max(score_threshold - 0.05, 0.0),
                score_threshold,
                min(score_threshold + 0.05, 1.0),
            )
        },
        "observed_label_rows": sum(1 for l in labels if l["label_observed"] == 1),
        "censored_label_rows": sum(1 for l in labels if l["label_observed"] == 0),
    }
    (out_dir / "pipeline_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build offline bottleneck features & labels"
    )
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
    parser.add_argument(
        "--window_size",
        type=float,
        default=None,
        help="Single complete-window size in logic seconds (takes precedence over --window_sizes)",
    )
    parser.add_argument(
        "--window_sizes",
        type=str,
        default="30,60",
        help="Comma-separated logic seconds",
    )
    parser.add_argument(
        "--stride",
        type=float,
        default=None,
        help="Window stride in logic seconds (default: non-overlapping, equal to each window size)",
    )
    parser.add_argument(
        "--horizon", type=float, default=120.0, help="Future horizon H (logic seconds)"
    )
    parser.add_argument(
        "--score_threshold",
        type=float,
        default=None,
        help="Override process-resource score threshold (default: 0.50).",
    )
    parser.add_argument(
        "--min_event_windows", type=int, default=DEFAULT_MIN_EVENT_WINDOWS
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help=f"Output directory (default: <run_dir>/{DERIVED_DIR_NAME})",
    )
    args = parser.parse_args()

    score_threshold = (
        args.score_threshold
        if args.score_threshold is not None
        else DEFAULT_SCORE_THRESHOLD
    )

    window_sizes = (
        [args.window_size]
        if args.window_size is not None
        else [float(x) for x in args.window_sizes.split(",") if x.strip()]
    )
    run_dir = args.run_dir.resolve()
    out_root = (args.out_dir or (run_dir / DERIVED_DIR_NAME)).resolve()

    env_dirs = _discover_env_dirs(run_dir, args.env_id)
    if not env_dirs:
        raise SystemExit(f"No episode_*/env_* directories under {run_dir}")

    summaries = []
    for env_dir in env_dirs:
        out_dir = _derived_out_dir(out_root, run_dir, env_dir)
        print(f"[build] {env_dir} → {out_dir}")
        summary = process_env_dir(
            env_dir=env_dir,
            out_dir=out_dir,
            window_sizes=window_sizes,
            stride=args.stride,
            horizon=args.horizon,
            score_threshold=score_threshold,
            min_event_windows=args.min_event_windows,
        )
        summaries.append(summary)
        if summary.get("status") == "skipped":
            print(
                f"  skipped={summary['skip_reason']}  "
                f"reason={summary['termination_reason']}  "
                f"abort_time={summary['abort_time_s']:.0f}s"
            )
            continue
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
            print(
                f"  ws={ps['window_size_s']}: hot_windows={ps['hot_windows']} top={ps['top_nodes'][:3]}"
            )

    (out_root / "all_env_summary.json").write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"[done] outputs under {out_root}")


if __name__ == "__main__":
    main()
