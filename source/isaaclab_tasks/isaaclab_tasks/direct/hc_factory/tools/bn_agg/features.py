"""Per-window node features and bottleneck_score_s (PDFormer dense target)."""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from typing import Any

from .constants import (
    ACTIVE_STATES,
    BLOCKED_STATES,
    BUFFER_MACHINE_AFFINITY,
    MATERIAL_CONSUMER,
    PROCESS_NEIGHBORS,
    SCORE_PEAK_FLOOR,
    SCORE_PEAK_RATIO,
    STARVED_STATES,
    STOP_STATES,
    W_ACTIVE,
    W_ACTIVE_DUR,
    W_DOWNSTREAM,
    W_QUEUE,
    W_STALL,
    W_STOP,
    W_UPSTREAM,
    W_WAIT,
)
from .io_util import _f, _i
from .resolve import resolve_feature_resource_id, wait_resource_id
from .timelines import (
    Interval,
    ResourceTimeline,
    _continuous_duration_ending_at,
    _intervals_overlap,
    _overlap_duration,
)

def _is_buffer(resource_id: str, resource_type: str) -> bool:
    return resource_type == "buffer" or str(resource_id).startswith("storage_")


def _window_len_s(row: dict) -> float:
    w0 = float(row.get("window_start_s") or 0)
    w1 = float(row.get("window_end_s") or 0)
    ws = float(row.get("window_size_s") or 0)
    return max(w1 - w0, ws, 1e-9)


def _stall_pct(row: dict) -> float:
    wlen = _window_len_s(row)
    stall = float(row.get("blocked_time_s") or 0) + float(row.get("starved_time_s") or 0)
    return min(stall / wlen, 1.0)


def row_is_hot(row: dict | None, score_threshold: float | None = None) -> bool:
    """STGNPP score-event candidate: process-chain turning point only.

    Dual-path split
    ---------------
    PDFormer (dense): every node/window keeps ``bottleneck_score_s``,
    ``is_window_peak``, ``is_momentary_bn``, occupancy, stall times, etc.
    STGNPP (sparse): score-events are TPM turning-points; L2 disturbances
    are a *separate* event source in ``labels.py`` (not merged onsets).

    Inventory sitting in a warehouse, routine STARVED/BLOCKED, and "who is
    busiest" must not mint events — those are PDFormer features.
    ``score_threshold`` is kept for the call signature; high score alone
    does not create events (scores are dense by design).
    """
    del score_threshold
    if not row:
        return False
    if _is_buffer(str(row.get("resource_id") or ""), str(row.get("resource_type") or "")):
        return False
    return int(row.get("is_turning_point") or 0) == 1


def _mean_ratio(timelines: dict[str, ResourceTimeline], rids: list[str], w0: float, w1: float, states: frozenset[str]) -> float:
    wlen = max(w1 - w0, 1e-9)
    vals = []
    for rid in rids:
        tl = timelines.get(rid)
        if tl is None:
            continue
        vals.append(_overlap_duration(tl.intervals, w0, w1, states) / wlen)
    return statistics.mean(vals) if vals else 0.0


def _affiliated_machines(storage_id: str) -> list[str]:
    for key, machines in BUFFER_MACHINE_AFFINITY.items():
        if key in storage_id:
            return machines
    return []


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
    disturbance_intervals: list[dict[str, Any]] | None = None,
    closed_windows_only: bool = False,
) -> list[dict]:
    # Precompute job waiting intervals keyed by *downstream* resource.
    wait_by_station: dict[str, list[tuple[float, float]]] = defaultdict(list)
    departures_by_station: dict[str, list[float]] = defaultdict(list)
    open_queue: dict[tuple[int, str], tuple[float, str]] = {}
    known_ids = list(timelines.keys())

    for row in sorted(job_rows, key=lambda r: _f(r.get("time_step"))):
        event = row.get("event")
        job_id = _i(row.get("job_id"), -1)
        task = row.get("task") or ""
        station = wait_resource_id(row, known_ids)
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
            # Departures stay on the finishing station (throughput of that node).
            dep_st = resolve_feature_resource_id(
                (row.get("station_id") or "unknown").strip(), known_ids
            ) or (row.get("station_id") or "unknown")
            departures_by_station[dep_st].append(t)

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

    # Transport tasks overlapping the window — attach delay to carrier / destination.
    transport_completed = [
        r for r in transport_rows if r.get("status") == "completed" or r.get("transport_end_time_step")
    ]
    known_ids = list(timelines.keys())

    def _task_delay(r: dict) -> tuple[float, float, float]:
        end = _f(r.get("transport_end_time_step"), _f(r.get("dropoff_time_step")))
        start = _f(r.get("transport_start_time_step"), _f(r.get("request_time_step")))
        req = _f(r.get("request_time_step"), start)
        return start, end, max(end - req, 0.0)

    def route_delay_by_carrier(w0: float, w1: float) -> dict[str, float]:
        by_c: dict[str, list[float]] = defaultdict(list)
        for r in transport_completed:
            start, end, delay = _task_delay(r)
            if end < w0 or start >= w1:
                continue
            cid = (r.get("carrier_id") or "").strip()
            if cid:
                by_c[cid].append(delay)
        return {k: statistics.mean(v) for k, v in by_c.items()}

    def inbound_wait_by_dest(w0: float, w1: float) -> dict[str, float]:
        by_d: dict[str, list[float]] = defaultdict(list)
        for r in transport_completed:
            start, end, delay = _task_delay(r)
            if end < w0 or start >= w1:
                continue
            dest = resolve_feature_resource_id((r.get("to_node") or "").strip(), known_ids)
            if dest:
                by_d[dest].append(delay)
        return {k: statistics.mean(v) for k, v in by_d.items()}

    def shortage_by_consumer(w0: float, w1: float) -> dict[str, float]:
        hits: dict[str, list[int]] = defaultdict(list)
        for r in material_rows:
            t = _f(r.get("logic_time_s"), _f(r.get("time_step")))
            if not (w0 <= t < w1):
                continue
            sku = (r.get("material_type") or r.get("submaterial") or "").strip()
            consumer = MATERIAL_CONSUMER.get(sku)
            if not consumer:
                continue
            flagged = str(r.get("shortage_flag")) in ("1", "True", "true")
            hits[consumer].append(1 if flagged else 0)
        return {k: (sum(v) / len(v) if v else 0.0) for k, v in hits.items()}

    dist_ivs = disturbance_intervals or []

    def disturbance_active(rid: str, w0: float, w1: float) -> float:
        for iv in dist_ivs:
            mapped = resolve_feature_resource_id(iv.get("resource_id") or "", known_ids)
            if mapped != rid:
                continue
            if _intervals_overlap(w0, w1, float(iv["start_s"]), float(iv["end_s"])):
                return 1.0
        return 0.0

    # Ensure buffer resources appear even without resource events
    for bid in buffer_occ:
        if bid not in timelines:
            timelines[bid] = ResourceTimeline(bid, "buffer", [Interval(0.0, episode_end, "IDLE")])
    known_ids = list(timelines.keys())

    n_windows = max(int(math.ceil(episode_end / window_size)), 1)
    rows_out: list[dict] = []

    for wi in range(n_windows):
        w0 = wi * window_size
        w1 = min((wi + 1) * window_size, episode_end)
        if w1 <= w0:
            continue
        if closed_windows_only and (w1 - w0) + 1e-9 < window_size:
            continue
        wlen = w1 - w0
        delay_carrier = route_delay_by_carrier(w0, w1)
        inbound_dest = inbound_wait_by_dest(w0, w1)
        shortage_map = shortage_by_consumer(w0, w1)

        buffer_ratio: dict[str, float] = {}
        for rid, tl in timelines.items():
            if tl.resource_type == "buffer" or rid.startswith("storage_"):
                _occ, mean_ratio, _g = buffer_stats(rid, w0, w1)
                buffer_ratio[rid] = mean_ratio

        window_rows: list[dict] = []
        for rid, tl in timelines.items():
            active = _overlap_duration(tl.intervals, w0, w1, ACTIVE_STATES)
            blocked = _overlap_duration(tl.intervals, w0, w1, BLOCKED_STATES)
            starved = _overlap_duration(tl.intervals, w0, w1, STARVED_STATES)
            stopped = _overlap_duration(tl.intervals, w0, w1, STOP_STATES)
            active_pct = active / wlen
            unavailable_pct = stopped / wlen
            active_dur = _continuous_duration_ending_at(tl.intervals, w1, ACTIVE_STATES)

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
                _mean_occ, mean_ratio, growth = buffer_stats(rid, w0, w1)
                occupancy_ratio = mean_ratio
                queue_growth = growth
                # Stock count is occupancy, not station WIP. Putting it in
                # queue_length_s made every window an STGNPP event.
                queue_length = 0.0
            else:
                occupancy_ratio = 0.0
                queue_growth = 0.0

            nbr = PROCESS_NEIGHBORS.get(rid)
            if nbr:
                ups, downs = nbr["up"], nbr["down"]
                if ups:
                    upstream_blocked = _mean_ratio(timelines, ups, w0, w1, BLOCKED_STATES)
                else:
                    upstream_blocked = starved / wlen
                if downs:
                    downstream_starved = _mean_ratio(timelines, downs, w0, w1, STARVED_STATES)
                else:
                    downstream_starved = blocked / wlen
            else:
                upstream_blocked = 0.0
                downstream_starved = 0.0

            aff_occ = 0.0
            aff_vals = []
            for bid, ratio in buffer_ratio.items():
                if rid in _affiliated_machines(bid):
                    aff_vals.append(ratio)
            if aff_vals:
                aff_occ = statistics.mean(aff_vals)

            window_rows.append(
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
                    "blocked_time_s": round(blocked, 6),
                    "starved_time_s": round(starved, 6),
                    "stop_time_s": round(stopped, 6),
                    "unavailable_pct_s": round(unavailable_pct, 6),
                    "inter_departure_var_s": round(inter_dep_var, 6),
                    "upstream_blocked_ratio_s": round(upstream_blocked, 6),
                    "downstream_starved_ratio_s": round(downstream_starved, 6),
                    "route_delay_s": round(delay_carrier.get(rid, 0.0), 6),
                    "inbound_wait_s": round(inbound_dest.get(rid, 0.0), 6),
                    "material_shortage_propagation_s": round(shortage_map.get(rid, 0.0), 6),
                    "affiliated_buffer_occ_s": round(aff_occ, 6),
                    "tb_minus_ts_s": round((blocked - starved) / wlen, 6),
                    "disturbance_active_s": round(disturbance_active(rid, w0, w1), 6),
                    "is_turning_point": 0,
                    "is_momentary_bn": 0,
                }
            )

        # Momentary bottleneck: longest current active period (Roser / IJPR DT).
        candidates = [
            r
            for r in window_rows
            if r["resource_type"] in ("machine", "gantry") and r["current_active_duration_s"] > 0
        ]
        if candidates:
            best_dur = max(r["current_active_duration_s"] for r in candidates)
            for r in candidates:
                if r["current_active_duration_s"] >= best_dur - 1e-9:
                    r["is_momentary_bn"] = 1

        # Turning point on the process chain (Li / Lai TPM): upstream TB-TS>0, self<0.
        by_rid = {r["resource_id"]: r for r in window_rows}
        for rid, nbr in PROCESS_NEIGHBORS.items():
            row = by_rid.get(rid)
            if row is None or not nbr["up"]:
                continue
            up_vals = [by_rid[u]["tb_minus_ts_s"] for u in nbr["up"] if u in by_rid]
            if not up_vals:
                continue
            if statistics.mean(up_vals) > 0 and row["tb_minus_ts_s"] < 0:
                if row["blocked_time_s"] + row["starved_time_s"] > 0:
                    row["is_turning_point"] = 1

        rows_out.extend(window_rows)
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
        stall = [_stall_pct(r) for r in rows]
        for i, r in enumerate(rows):
            score = (
                W_QUEUE * q[i]
                + W_WAIT * w[i]
                + W_STALL * stall[i]
                + W_ACTIVE * a[i]
                + W_ACTIVE_DUR * d[i]
                + W_UPSTREAM * r["upstream_blocked_ratio_s"]
                + W_DOWNSTREAM * r["downstream_starved_ratio_s"]
                + W_STOP * float(r.get("unavailable_pct_s") or 0.0)
            )
            r["bottleneck_score_s"] = round(score, 6)
            r["norm_queue_length_s"] = round(q[i], 6)
            r["norm_avg_waiting_time_s"] = round(w[i], 6)
            r["norm_current_active_duration_s"] = round(d[i], 6)
        peak = max(float(r["bottleneck_score_s"]) for r in rows)
        for r in rows:
            # Feature for PDFormer (who is relatively busiest). Not an event trigger.
            r["is_window_peak"] = int(
                peak >= SCORE_PEAK_FLOOR
                and float(r["bottleneck_score_s"]) >= SCORE_PEAK_RATIO * peak - 1e-12
            )
    return feature_rows
