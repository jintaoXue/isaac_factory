"""Closed-window streaming vs full-episode aggregation."""

from __future__ import annotations

import sys
from pathlib import Path

_TOOLS = Path(__file__).resolve().parent.parent
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

from bn_agg.features import compute_window_features
from bn_agg.labels import parse_disturbance_l2_intervals
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


if __name__ == "__main__":
    test_closed_windows_skip_partial()
    test_open_disturbance_truncated()
    print("ok")
