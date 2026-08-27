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
    HOT_BLOCK_FRAC,
    HOT_COUPLED_DOWN,
    HOT_COUPLED_STALL_FRAC,
    HOT_COUPLED_UP,
    HOT_DOWNTIME_UNAVAIL,
    HOT_HUMAN_PRESENT_EPS,
    HOT_INBOUND_S,
    HOT_INBOUND_STARVE_FRAC,
    HOT_LABOR_ACTIVE,
    HOT_OPERATOR_STALL_FRAC,
    HOT_QUEUE_PILEUP,
    HOT_ROUTE_S,
    HOT_SHORTAGE_PROP,
    HOT_SHORTAGE_STARVE_FRAC,
    MATERIAL_CONSUMER,
    MATERIAL_CONSUMERS,
    MATERIAL_SHORTAGE_TASKS,
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
from .resolve import resolve_feature_resource_id, resolve_feature_resource_ids, wait_resource_id
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


def _is_carrier(resource_type: str) -> bool:
    return resource_type in ("gantry", "transport_robot")


def row_is_hot(row: dict | None, score_threshold: float | None = None) -> bool:
    """STGNPP score-event candidate: process stall on machines / gantry / AGV.

    Dual-path split
    ---------------
    PDFormer (dense): every node/window keeps ``bottleneck_score_s``,
    ``is_window_peak``, ``is_momentary_bn``, occupancy, stall times, etc.
    STGNPP (sparse): TPM turning-points, queue pileup, kitting starve with
    high shortage, inbound-wait starve, blocked backlog, or carrier delay
    with stall. Injected L2 is environment context (``disturbance_active_s``),
    never an event onset.

    Inventory sitting in a warehouse, routine STARVED/BLOCKED, and "who is
    busiest" must not mint events — those are PDFormer features.
    ``score_threshold`` is kept for the call signature; high score alone
    does not create events (scores are dense by design).
    """
    del score_threshold
    if not row:
        return False
    rtype = str(row.get("resource_type") or "")
    if _is_buffer(str(row.get("resource_id") or ""), rtype):
        return False
    if rtype == "machine":
        return (
            int(row.get("is_turning_point") or 0) == 1
            or _row_is_queue_pileup(row)
            or _row_is_material_starve(row)
            or _row_is_inbound_starve(row)
            or _row_is_blocked_backlog(row)
            or _row_is_coupled_stall(row)
            or _row_is_operator_wait(row)
            or _row_is_process_downtime(row)
        )
    if _is_carrier(rtype):
        return _row_is_carrier_delay(row) or _row_is_carrier_stop(row)
    return False


def _row_is_queue_pileup(row: dict) -> bool:
    """Rare station WIP pile (not warehouse stock). Complements TPM starve events."""
    if str(row.get("resource_type") or "") != "machine":
        return False
    q = float(row.get("queue_length_s") or 0)
    stall = float(row.get("blocked_time_s") or 0) + float(row.get("starved_time_s") or 0)
    return q >= HOT_QUEUE_PILEUP - 1e-9 and stall > 0.0


def _row_is_material_starve(row: dict) -> bool:
    """Kitting WAITING while the kit SKU at this station is short.

    TPM turning-points need upstream blocked and self starved. Kit shortage
    starves the workbench *and* often starves grooving (buffers absorb WIP),
    so TB−TS never crosses. This is still a process phenomenon — not a copy
    of disturbance_log.
    """
    if str(row.get("resource_type") or "") != "machine":
        return False
    shortage = float(row.get("material_shortage_propagation_s") or 0)
    if shortage < HOT_SHORTAGE_PROP:
        return False
    wlen = _window_len_s(row)
    starved = float(row.get("starved_time_s") or 0)
    return starved >= HOT_SHORTAGE_STARVE_FRAC * wlen


def _row_is_inbound_starve(row: dict) -> bool:
    """Station starved while inbound / route wait is high (logistics signature)."""
    if str(row.get("resource_type") or "") != "machine":
        return False
    inbound = float(row.get("inbound_wait_s") or 0)
    route = float(row.get("route_delay_s") or 0)
    if inbound < HOT_INBOUND_S and route < HOT_ROUTE_S:
        return False
    wlen = _window_len_s(row)
    starved = float(row.get("starved_time_s") or 0)
    return starved >= HOT_INBOUND_STARVE_FRAC * wlen


def _row_is_blocked_backlog(row: dict) -> bool:
    """Upstream blocked with a waiting job (machine-down / closed-station backup)."""
    if str(row.get("resource_type") or "") != "machine":
        return False
    wlen = _window_len_s(row)
    blocked = float(row.get("blocked_time_s") or 0)
    q = float(row.get("queue_length_s") or 0)
    return blocked >= HOT_BLOCK_FRAC * wlen and q >= 1.0 - 1e-9


def _row_is_coupled_stall(row: dict) -> bool:
    """Sustained BLOCKED/WAITING plus line coupling — keeps human/logistics pulses long."""
    if str(row.get("resource_type") or "") != "machine":
        return False
    wlen = _window_len_s(row)
    stall = float(row.get("blocked_time_s") or 0) + float(row.get("starved_time_s") or 0)
    if stall < HOT_COUPLED_STALL_FRAC * wlen:
        return False
    return (
        float(row.get("upstream_blocked_ratio_s") or 0) >= HOT_COUPLED_UP
        or float(row.get("downstream_starved_ratio_s") or 0) >= HOT_COUPLED_DOWN
        or float(row.get("inbound_wait_s") or 0) >= HOT_INBOUND_S
        or float(row.get("route_delay_s") or 0) >= HOT_ROUTE_S
        or float(row.get("material_shortage_propagation_s") or 0) >= HOT_SHORTAGE_PROP
        or float(row.get("queue_length_s") or 0) >= 1.0 - 1e-9
    )


def _row_is_operator_wait(row: dict) -> bool:
    """Station stalled while workers are unavailable or all on-duty humans are busy.

    STOP is leave (observed, not an L2 class name). Labor saturation is the
    5→3 case: nobody is on leave, but every present human is PROCESSING.
    """
    if str(row.get("resource_type") or "") != "machine":
        return False
    absent = float(row.get("operator_absent_s") or 0) >= HOT_DOWNTIME_UNAVAIL
    sat = float(row.get("labor_saturated_s") or 0) >= 0.5
    if not absent and not sat:
        return False
    wlen = _window_len_s(row)
    stall = float(row.get("blocked_time_s") or 0) + float(row.get("starved_time_s") or 0)
    return stall >= HOT_OPERATOR_STALL_FRAC * wlen


def _row_is_carrier_delay(row: dict) -> bool:
    """Gantry / AGV delayed and not making progress (process, not freeze X)."""
    if not _is_carrier(str(row.get("resource_type") or "")):
        return False
    inbound = float(row.get("inbound_wait_s") or 0)
    route = float(row.get("route_delay_s") or 0)
    if inbound < HOT_INBOUND_S and route < HOT_ROUTE_S:
        return False
    wlen = _window_len_s(row)
    stall = float(row.get("blocked_time_s") or 0) + float(row.get("starved_time_s") or 0)
    return stall >= HOT_INBOUND_STARVE_FRAC * wlen


def _row_is_carrier_stop(row: dict) -> bool:
    """Gantry / AGV actually STOP this window (freeze observed on the resource)."""
    if not _is_carrier(str(row.get("resource_type") or "")):
        return False
    wlen = _window_len_s(row)
    stopped = float(row.get("stop_time_s") or 0)
    unav = float(row.get("unavailable_pct_s") or 0)
    return unav >= HOT_DOWNTIME_UNAVAIL or stopped >= HOT_DOWNTIME_UNAVAIL * wlen


def _row_is_process_downtime(row: dict) -> bool:
    """Station actually STOP this window while an L2 pulse is injected.

    Resource log STOP is the process observation. disturbance_active only
    distinguishes L2 from L0-disabled ws1. The event is still event_source=score
    with empty disturbance_type.
    """
    if str(row.get("resource_type") or "") != "machine":
        return False
    if float(row.get("disturbance_active_s") or 0) < 0.5:
        return False
    return float(row.get("unavailable_pct_s") or 0) >= HOT_DOWNTIME_UNAVAIL


def kit_assemble_times(material_rows: list[dict]) -> dict[str, float]:
    """First time each kit piece is actually assembled onto a job.

    Warehouse hide also logs ``event=consume`` (storage → disappear) while
    ``finished_task`` is still ``none``. Only kitting pickup counts.
    """
    assembled: dict[str, float] = {}
    for r in material_rows:
        sku = (r.get("material_type") or r.get("submaterial") or "").strip()
        if sku not in MATERIAL_SHORTAGE_TASKS:
            continue
        if (r.get("event") or "").strip() != "consume":
            continue
        if (r.get("finished_task") or "").strip() != "logistic_for_batch_spot_welding":
            continue
        mid = (r.get("material_id") or "").strip()
        if not mid or mid in assembled:
            continue
        assembled[mid] = _f(r.get("logic_time_s"), _f(r.get("time_step")))
    return assembled


def shortage_fraction_by_consumer(
    material_rows: list[dict],
    w0: float,
    w1: float,
    *,
    assembled_at: dict[str, float] | None = None,
) -> dict[str, float]:
    """Fraction of pre-assembly kitting-stage kit rows that are short.

    Warehouse snapshots of hidden stock are excluded. Snapshots after the
    job actually picks the kit (nominal consume) are also excluded — those
    still have storage=disappear and would false-positive on none.
    If several SKUs map to the same station, take the max.
    """
    if assembled_at is None:
        assembled_at = kit_assemble_times(material_rows)
    by_c_sku: dict[tuple[str, str], list[int]] = defaultdict(list)
    for r in material_rows:
        t = _f(r.get("logic_time_s"), _f(r.get("time_step")))
        if not (w0 <= t < w1):
            continue
        sku = (r.get("material_type") or r.get("submaterial") or "").strip()
        consumers = MATERIAL_CONSUMERS.get(sku)
        if not consumers:
            one = MATERIAL_CONSUMER.get(sku)
            consumers = (one,) if one else ()
        if not consumers:
            continue
        tasks = MATERIAL_SHORTAGE_TASKS.get(sku)
        if tasks is not None:
            finished = (r.get("finished_task") or "").strip()
            if finished not in tasks:
                continue
            mid = (r.get("material_id") or "").strip()
            t_asm = assembled_at.get(mid)
            if t_asm is not None and t >= t_asm:
                continue
        flagged = str(r.get("shortage_flag")) in ("1", "True", "true")
        for consumer in consumers:
            if consumer:
                by_c_sku[(consumer, sku)].append(1 if flagged else 0)
    by_c: dict[str, list[float]] = defaultdict(list)
    for (consumer, _sku), flags in by_c_sku.items():
        by_c[consumer].append(sum(flags) / len(flags))
    return {c: max(vals) for c, vals in by_c.items()}


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
    assembled_at = kit_assemble_times(material_rows)

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

    # Transport waits overlapping the window — include still-open tasks so a
    # frozen AGV/gantry shows inbound delay, not only completed trips.
    known_ids = list(timelines.keys())

    def _latest_transport_by_task() -> dict[str, dict]:
        latest: dict[str, dict] = {}
        for r in transport_rows:
            tid = (r.get("task_id") or "").strip()
            if tid:
                latest[tid] = r
        return latest

    def _latest_transport_by_task_carrier() -> dict[tuple[str, str], dict]:
        """Keep the last snapshot per (task, carrier). Latest-per-task drops AGV rows."""
        latest: dict[tuple[str, str], dict] = {}
        for r in transport_rows:
            tid = (r.get("task_id") or "").strip()
            cid = (r.get("carrier_id") or "").strip()
            if tid and cid:
                latest[(tid, cid)] = r
        return latest

    def _task_complete_end() -> dict[str, float]:
        done: dict[str, float] = {}
        for r in transport_rows:
            tid = (r.get("task_id") or "").strip()
            if not tid:
                continue
            end = _f(r.get("transport_end_time_step"), _f(r.get("dropoff_time_step"), default=-1.0))
            if end >= 0:
                done[tid] = max(done.get(tid, end), end)
        return done

    task_complete_end = _task_complete_end()

    def _opt_time(r: dict, *keys: str) -> float:
        for k in keys:
            v = r.get(k)
            if v is None or v == "":
                continue
            try:
                return float(v)
            except (TypeError, ValueError):
                continue
        return -1.0

    def _task_window_wait(r: dict, w0: float, w1: float) -> float | None:
        """Inbound wait at dest: request → arrival. Includes still-open tasks."""
        req = _f(r.get("request_time_step"), _f(r.get("transport_start_time_step")))
        end = _f(r.get("transport_end_time_step"), _f(r.get("dropoff_time_step"), default=-1.0))
        status = (r.get("status") or "").strip()
        if end < 0 or status in ("requested", "in_progress", "delayed"):
            tid = (r.get("task_id") or "").strip()
            cap = task_complete_end.get(tid, w1)
            end = min(float(cap), w1)
        if end < w0 or req >= w1:
            return None
        return max(min(end, w1) - max(req, w0), 0.0)

    def _carrier_delay_span(r: dict) -> tuple[float, float] | None:
        """Wait-for-carrier [request, pickup), not travel after pickup.

        Completed trips with no pickup are travel-only. Open delayed rows
        may still reach task_complete; the window value is clipped to
        STARVED/WAITING on that carrier so driving windows stay 0.
        """
        status = (r.get("status") or "").strip()
        req = _opt_time(r, "request_time_step")
        if req < 0:
            req = _opt_time(r, "transport_start_time_step")
        if req < 0:
            return None
        pickup = _opt_time(r, "pickup_time_step")
        if pickup < 0 and status == "in_progress":
            pickup = _opt_time(r, "transport_start_time_step")
        if pickup >= 0 and pickup < req:
            pickup = -1.0
        if pickup >= 0:
            if pickup <= req:
                return None
            return req, pickup
        if status == "completed":
            return None
        if status not in ("requested", "delayed", "in_progress", ""):
            return None
        tid = (r.get("task_id") or "").strip()
        cap = task_complete_end.get(tid, -1.0)
        log_t = _opt_time(r, "time_step")
        end = cap if cap >= 0 else log_t
        if end < 0:
            return req, float("inf")
        if end <= req:
            return None
        return req, end

    def route_delay_by_carrier(w0: float, w1: float) -> dict[str, float]:
        by_c: dict[str, float] = defaultdict(float)
        for r in _latest_transport_by_task_carrier().values():
            span = _carrier_delay_span(r)
            if span is None:
                continue
            a, b = span
            if b == float("inf"):
                b = w1
            if b < w0 or a >= w1:
                continue
            raw = max(min(b, w1) - max(a, w0), 0.0)
            if raw <= 0:
                continue
            cid = (r.get("carrier_id") or "").strip()
            if not cid:
                continue
            tl = timelines.get(cid)
            if tl is None:
                by_c[cid] = max(by_c[cid], raw)
                continue
            stall = _overlap_duration(
                tl.intervals, max(a, w0), min(b, w1), STARVED_STATES
            )
            by_c[cid] = max(by_c[cid], min(raw, stall))
        return dict(by_c)

    def inbound_wait_by_dest(w0: float, w1: float) -> dict[str, float]:
        by_d: dict[str, list[float]] = defaultdict(list)
        for r in _latest_transport_by_task().values():
            wait = _task_window_wait(r, w0, w1)
            if wait is None:
                continue
            dest = resolve_feature_resource_id((r.get("to_node") or "").strip(), known_ids)
            if dest:
                by_d[dest].append(wait)
        raw = {k: statistics.mean(v) for k, v in by_d.items()}
        spread = dict(raw)
        for dest, val in raw.items():
            machines = _affiliated_machines(dest)
            if not machines:
                continue
            for mid in machines:
                tl = timelines.get(mid)
                if tl is None:
                    continue
                starved = _overlap_duration(tl.intervals, w0, w1, STARVED_STATES)
                if starved > 0.0:
                    spread[mid] = max(spread.get(mid, 0.0), val)
        return spread

    def shortage_by_consumer(w0: float, w1: float) -> dict[str, float]:
        return shortage_fraction_by_consumer(
            material_rows, w0, w1, assembled_at=assembled_at
        )

    dist_ivs = disturbance_intervals or []

    def disturbance_active(rid: str, w0: float, w1: float) -> float:
        for iv in dist_ivs:
            mapped = set(resolve_feature_resource_ids(iv.get("resource_id") or "", known_ids))
            if rid not in mapped:
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

        human_unav = max(
            (
                float(r.get("unavailable_pct_s") or 0)
                for r in window_rows
                if r.get("resource_type") == "human"
            ),
            default=0.0,
        )
        for r in window_rows:
            r["operator_absent_s"] = round(human_unav, 6)
            r["labor_saturated_s"] = 0.0

        rows_out.extend(window_rows)
    _stamp_labor_saturated(rows_out)
    return rows_out


def _stamp_labor_saturated(rows_out: list[dict]) -> None:
    """Mark windows where every present, on-duty human is busy (not unused slots)."""
    present = {
        str(r.get("resource_id") or "")
        for r in rows_out
        if r.get("resource_type") == "human"
        and (
            float(r.get("active_pct_s") or 0) > HOT_HUMAN_PRESENT_EPS
            or float(r.get("unavailable_pct_s") or 0) > HOT_HUMAN_PRESENT_EPS
        )
    }
    by_window: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows_out:
        key = (r.get("run_id"), r.get("env_id"), r.get("window_size_s"), r.get("window_index"))
        by_window[key].append(r)
    for group in by_window.values():
        on_duty = [
            r
            for r in group
            if r.get("resource_type") == "human"
            and str(r.get("resource_id") or "") in present
            and float(r.get("unavailable_pct_s") or 0) < HOT_DOWNTIME_UNAVAIL
        ]
        sat = (
            1.0
            if on_duty and all(float(r.get("active_pct_s") or 0) >= HOT_LABOR_ACTIVE for r in on_duty)
            else 0.0
        )
        for r in group:
            r["labor_saturated_s"] = sat


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
                + W_STOP * float(r.get("unavailable_pct_s") or 0.0)  # 0: downtime is X, not y
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
