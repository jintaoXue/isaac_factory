"""Hot-window process events and PDFormer will/mark/cause labels.

Disturbance intervals are environment context (see ``disturbance_active_s`` in
features). They are not bottleneck onsets and do not drive will/mark/cause.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from .constants import DISTURBANCE_L2_TYPES
from .features import row_is_hot
from .io_util import _f, _i


def _map_disturbance_resource_type(resource_id: str, raw_type: str) -> str:
    """Map disturbance_log target type onto feature-table resource_type."""
    rid = (resource_id or "").strip()
    rt = (raw_type or "").strip().lower()
    if rid.startswith("gantry_") or rt == "gantry":
        return "gantry"
    if rid.startswith("robot_") or rid.startswith("agv_") or rt == "transport_robot":
        return "transport_robot"
    if rt == "logistics":
        if rid.startswith("robot_") or rid.startswith("agv_"):
            return "transport_robot"
        return "gantry"
    if rid.startswith("human_") or rt == "human":
        return "human"
    if rid.startswith("material_") or rt == "material":
        return "material"
    if rt in ("machine", "gantry", "human", "transport_robot", "buffer", "material"):
        return rt
    return raw_type or "machine"


def parse_disturbance_l2_intervals(
    disturbance_rows: list[dict[str, str]],
    *,
    open_end_s: float | None = None,
) -> list[dict[str, Any]]:
    """Extract L2 intervals from disturbance_log.csv for *input* features.

    Collector writes a start row (no end) and an end row (with end_time_step).
    Completed intervals are always kept. If ``open_end_s`` is set, still-open
    disturbances are truncated there so live windows can see ``disturbance_active_s``.
    These intervals never become STGNPP events or will/mark labels.
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
            if open_end_s is None:
                continue
            ev = dict(ev)
            ev["end_s"] = float(open_end_s)
            ev["open"] = True
        if float(ev["end_s"]) <= float(ev["start_s"]):
            continue
        out.append(ev)
    out.sort(key=lambda e: (e["start_s"], e["resource_id"]))
    return out


def _assign_event_ids(events: list[dict]) -> list[dict]:
    merged = [dict(e) for e in events]
    for e in merged:
        e.setdefault("event_source", "score")
        e.setdefault("disturbance_id", "")
        e.setdefault("disturbance_type", "")
    merged.sort(key=lambda e: (e["window_size_s"], e["start_s"], e["resource_id"]))
    for i, ev in enumerate(merged):
        ev["event_id"] = i
    return merged


def _coalesce_per_node_score_events(
    windows: dict[int, list[dict]],
    window_meta: dict[int, dict],
    score_threshold: float,
    min_event_windows: int,
) -> list[dict]:
    """Merge consecutive windows where this resource is a turning-point (sparse)."""
    rid_type: dict[str, str] = {}
    for rs in windows.values():
        for r in rs:
            rid_type.setdefault(r["resource_id"], r["resource_type"])

    score_events: list[dict] = []
    wis = sorted(windows)
    for rid, rtype in rid_type.items():
        cur: dict | None = None
        for wi in wis:
            row = next((r for r in windows[wi] if r["resource_id"] == rid), None)
            meta = window_meta[wi]
            hot = row_is_hot(row, score_threshold)
            if hot:
                score = float(row["bottleneck_score_s"])
                if cur is not None and wi == cur["end_window_index"] + 1:
                    cur["end_window_index"] = wi
                    cur["end_s"] = meta["window_end_s"]
                    cur["duration_s"] = cur["end_s"] - cur["start_s"]
                    cur["max_score"] = max(cur["max_score"], score)
                    cur["n_windows"] += 1
                else:
                    if cur is not None and cur["n_windows"] >= min_event_windows:
                        score_events.append(cur)
                    cur = {
                        "run_id": meta["run_id"],
                        "env_id": meta["env_id"],
                        "window_size_s": meta["window_size_s"],
                        "resource_id": rid,
                        "resource_type": rtype,
                        "start_window_index": wi,
                        "end_window_index": wi,
                        "start_s": meta["window_start_s"],
                        "end_s": meta["window_end_s"],
                        "duration_s": meta["window_end_s"] - meta["window_start_s"],
                        "max_score": score,
                        "n_windows": 1,
                        "event_source": "score",
                        "disturbance_id": "",
                        "disturbance_type": "",
                    }
            else:
                if cur is not None and cur["n_windows"] >= min_event_windows:
                    score_events.append(cur)
                cur = None
        if cur is not None and cur["n_windows"] >= min_event_windows:
            score_events.append(cur)
    return score_events


def _process_root_cause(nr: dict) -> str:
    """Process-side cause of a hot window. Never copies L2 disturbance_type."""
    if nr["blocked_time_s"] >= nr["starved_time_s"] and nr["blocked_time_s"] > 0:
        return "blocked_downstream"
    if nr["starved_time_s"] > 0:
        return "starved_upstream"
    if nr["queue_length_s"] > 0 or nr["avg_waiting_time_s"] > 0:
        return "queue_buildup"
    if nr["active_pct_s"] >= 0.8:
        return "high_utilization"
    return "score_threshold"


def build_labels_and_events(
    feature_rows: list[dict],
    horizon: float,
    score_threshold: float,
    min_event_windows: int,
    disturbance_rows: list[dict[str, str]] | None = None,
    as_of_s: float | None = None,
) -> tuple[list[dict], list[dict]]:
    """Window-level labels + process bottleneck events for PDFormer / STGNPP.

    Events are consecutive TPM turning-point windows only. Injected L2 is not
    an onset: the model conditions on ``disturbance_active_s`` (and episode
    dim/intensity) to predict process phenomena that arise under that config.

    ``bottleneck_node_t`` prefers TP-hot, then momentary, else score argmax.
    ``will_bottleneck`` / ``future_bottleneck_object_id`` look at process events
    in (t, t+H]. ``root_cause_reason`` is a process heuristic (queue / stall),
    never ``machine_failure`` / ``human_unavailable`` / etc.
    """
    del disturbance_rows  # kept on the signature; L2 is features-only
    by_ws: dict[float, list[dict]] = defaultdict(list)
    for r in feature_rows:
        by_ws[r["window_size_s"]].append(r)

    label_rows: list[dict] = []
    event_rows: list[dict] = []

    for window_size, rows in by_ws.items():
        windows: dict[int, list[dict]] = defaultdict(list)
        for r in rows:
            windows[r["window_index"]].append(r)

        window_meta: dict[int, dict] = {}
        for wi, rs in sorted(windows.items()):
            hot = [r for r in rs if row_is_hot(r, score_threshold)]
            turning = [r for r in rs if int(r.get("is_turning_point") or 0)]
            momentary = [r for r in rs if int(r.get("is_momentary_bn") or 0)]
            if hot:
                best = max(hot, key=lambda x: x["bottleneck_score_s"])
            elif turning:
                best = max(turning, key=lambda x: x["bottleneck_score_s"])
            elif momentary:
                best = max(momentary, key=lambda x: x["current_active_duration_s"])
            else:
                best = max(rs, key=lambda x: x["bottleneck_score_s"])
            hot_ids = {r["resource_id"] for r in hot}
            n_hot_nodes = len(hot_ids)
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
                "is_bottleneck_window": int(n_hot_nodes > 0),
                "n_hot_nodes": n_hot_nodes,
            }

        score_events = _coalesce_per_node_score_events(
            windows, window_meta, score_threshold, min_event_windows
        )
        events = _assign_event_ids(score_events)
        event_rows.extend(events)

        for wi, meta in sorted(window_meta.items()):
            t = meta["window_start_s"]
            horizon_ready = as_of_s is None or (t + horizon) <= float(as_of_s) + 1e-6
            future = [
                ev
                for ev in events
                if ev["start_s"] > t and ev["start_s"] <= t + horizon
            ]
            if horizon_ready and future:
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

            reason = ""
            if meta["is_bottleneck_window"]:
                node_rows = [
                    r
                    for r in windows[wi]
                    if r["resource_id"] == meta["bottleneck_node_t"]
                ]
                if node_rows:
                    reason = _process_root_cause(node_rows[0])

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
                    "label_horizon_ready": int(horizon_ready),
                }
            )

    return label_rows, event_rows
