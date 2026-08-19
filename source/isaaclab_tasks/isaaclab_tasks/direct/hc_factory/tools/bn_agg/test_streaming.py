"""Closed-window streaming vs full-episode aggregation."""

from __future__ import annotations

import sys
from pathlib import Path

_TOOLS = Path(__file__).resolve().parent.parent
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

from bn_agg.features import compute_window_features
from bn_agg.labels import build_labels_and_events, parse_disturbance_l2_intervals
from bn_agg.timelines import Interval, ResourceTimeline


def test_closed_windows_skip_partial() -> None:
    timelines = {
        "m0": ResourceTimeline(
            "m0",
            "machine",
            [Interval(0.0, 200.0, "PROCESSING")],
        )
    }
    closed = compute_window_features(
        timelines=dict(timelines),
        job_rows=[],
        buffer_rows=[],
        transport_rows=[],
        material_rows=[],
        window_size=60.0,
        episode_end=200.0,
        run_id="t",
        env_id=0,
        closed_windows_only=True,
    )
    full = compute_window_features(
        timelines=dict(timelines),
        job_rows=[],
        buffer_rows=[],
        transport_rows=[],
        material_rows=[],
        window_size=60.0,
        episode_end=200.0,
        run_id="t",
        env_id=0,
        closed_windows_only=False,
    )
    assert {r["window_index"] for r in closed} == {0, 1, 2}
    assert all(abs(r["window_end_s"] - r["window_start_s"] - 60.0) < 1e-6 for r in closed)
    assert {r["window_index"] for r in full} == {0, 1, 2, 3}
    last = [r for r in full if r["window_index"] == 3][0]
    assert abs(last["window_end_s"] - last["window_start_s"] - 20.0) < 1e-6


def test_open_disturbance_truncated() -> None:
    rows = [
        {
            "disturbance_id": "d1",
            "disturbance_type": "machine_failure",
            "target_resource_id": "m0",
            "target_resource_type": "machine",
            "start_logic_time_s": "10",
            "end_logic_time_s": "",
            "start_time_step": "10",
            "end_time_step": "-1",
        }
    ]
    done = parse_disturbance_l2_intervals(rows)
    assert done == []
    live = parse_disturbance_l2_intervals(rows, open_end_s=90.0)
    assert len(live) == 1
    assert live[0]["open"] is True
    assert live[0]["end_s"] == 90.0


def _feat(
    *,
    wi: int,
    rid: str,
    turning: int = 0,
    dist: float = 0.0,
    score: float = 0.1,
    blocked: float = 0.0,
    starved: float = 0.0,
) -> dict:
    return {
        "run_id": "t",
        "env_id": 0,
        "window_index": wi,
        "window_start_s": float(wi * 60),
        "window_end_s": float((wi + 1) * 60),
        "window_size_s": 60.0,
        "resource_id": rid,
        "resource_type": "machine",
        "queue_length_s": 0.0,
        "avg_waiting_time_s": 0.0,
        "active_pct_s": 0.2,
        "current_active_duration_s": 0.0,
        "blocked_time_s": blocked,
        "starved_time_s": starved,
        "unavailable_pct_s": dist,
        "disturbance_active_s": dist,
        "is_turning_point": turning,
        "is_momentary_bn": 0,
        "bottleneck_score_s": score,
    }


def test_l2_is_context_not_event() -> None:
    """Injected L2 must not mint STGNPP events or will/cause labels."""
    rows = [
        _feat(wi=0, rid="m0", dist=1.0, score=0.9),
        _feat(wi=0, rid="m1", dist=0.0, score=0.1),
        _feat(wi=1, rid="m0", dist=1.0, score=0.9),
        _feat(wi=1, rid="m1", dist=0.0, score=0.1),
    ]
    dist_rows = [
        {
            "disturbance_id": "d1",
            "disturbance_type": "machine_failure",
            "target_resource_id": "m0",
            "target_resource_type": "machine",
            "start_logic_time_s": "10",
            "end_logic_time_s": "90",
            "start_time_step": "10",
            "end_time_step": "90",
        }
    ]
    labels, events = build_labels_and_events(
        rows, horizon=180.0, score_threshold=0.55, min_event_windows=1,
        disturbance_rows=dist_rows,
    )
    assert events == []
    assert all(l["will_bottleneck"] == 0 for l in labels)
    assert all(l["is_bottleneck_window"] == 0 for l in labels)
    assert all(l["root_cause_reason"] == "" for l in labels)


def test_turning_point_is_process_event() -> None:
    rows = [
        _feat(wi=0, rid="m0", turning=0, score=0.2, starved=0.0),
        _feat(wi=0, rid="m1", turning=0, score=0.1),
        _feat(wi=1, rid="m0", turning=1, score=0.4, starved=40.0),
        _feat(wi=1, rid="m1", turning=0, score=0.1),
        _feat(wi=2, rid="m0", turning=0, score=0.2),
        _feat(wi=2, rid="m1", turning=0, score=0.1),
    ]
    labels, events = build_labels_and_events(
        rows, horizon=180.0, score_threshold=0.55, min_event_windows=1,
        disturbance_rows=[{"disturbance_type": "machine_failure", "disturbance_id": "x",
                           "target_resource_id": "m1", "start_logic_time_s": "0",
                           "end_logic_time_s": "200", "start_time_step": "0", "end_time_step": "200"}],
    )
    assert len(events) == 1
    assert events[0]["event_source"] == "score"
    assert events[0]["resource_id"] == "m0"
    assert events[0]["disturbance_type"] == ""
    hot = [l for l in labels if l["is_bottleneck_window"]]
    assert len(hot) == 1
    assert hot[0]["window_index"] == 1
    assert hot[0]["root_cause_reason"] == "starved_upstream"
    assert labels[0]["will_bottleneck"] == 1
    assert labels[0]["future_bottleneck_object_id"] == "m0"


if __name__ == "__main__":
    test_closed_windows_skip_partial()
    test_open_disturbance_truncated()
    test_l2_is_context_not_event()
    test_turning_point_is_process_event()
    print("ok")
