"""Resource state timelines from resource_event_log.jsonl."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .constants import HOT_ABSENT_RAW_STATES

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


def _event_to_state(event: dict[str, Any]) -> str:
    """Prefer logged to_state; remap freeze/leave raw states that were stored as PROCESSING."""
    raw = str(event.get("raw_to_state") or "")
    to_state = str(event.get("to_state") or "")
    if raw in HOT_ABSENT_RAW_STATES:
        return "STOP"
    return to_state


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
            state = _event_to_state(e) or state
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


def _intervals_overlap(a0: float, a1: float, b0: float, b1: float) -> bool:
    return a0 < b1 and b0 < a1
