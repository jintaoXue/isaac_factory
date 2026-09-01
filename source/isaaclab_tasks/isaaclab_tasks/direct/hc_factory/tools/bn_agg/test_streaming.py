"""Closed-window streaming vs full-episode aggregation."""

from __future__ import annotations

import sys
from pathlib import Path

_TOOLS = Path(__file__).resolve().parent.parent
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

from bn_agg.features import _stamp_labor_saturated, compute_window_features
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
    unavailable: float | None = None,
    score: float = 0.1,
    blocked: float = 0.0,
    starved: float = 0.0,
    queue: float = 0.0,
    shortage: float = 0.0,
    inbound: float = 0.0,
    route: float = 0.0,
    resource_type: str = "machine",
    upstream: float = 0.0,
    downstream: float = 0.0,
) -> dict:
    unav = dist if unavailable is None else unavailable
    return {
        "run_id": "t",
        "env_id": 0,
        "window_index": wi,
        "window_start_s": float(wi * 60),
        "window_end_s": float((wi + 1) * 60),
        "window_size_s": 60.0,
        "resource_id": rid,
        "resource_type": resource_type,
        "queue_length_s": queue,
        "avg_waiting_time_s": 0.0,
        "active_pct_s": 0.2,
        "current_active_duration_s": 0.0,
        "blocked_time_s": blocked,
        "starved_time_s": starved,
        "unavailable_pct_s": unav,
        "upstream_blocked_ratio_s": upstream,
        "downstream_starved_ratio_s": downstream,
        "disturbance_active_s": dist,
        "material_shortage_propagation_s": shortage,
        "inbound_wait_s": inbound,
        "route_delay_s": route,
        "affiliated_buffer_occ_s": 0.0,
        "operator_absent_s": 0.0,
        "labor_saturated_s": 0.0,
        "is_turning_point": turning,
        "is_momentary_bn": 0,
        "bottleneck_score_s": score,
    }


def test_l2_is_context_not_event() -> None:
    """Scheduled L2 without actual STOP must not mint STGNPP events."""
    rows = [
        _feat(wi=0, rid="m0", dist=1.0, unavailable=0.0, score=0.9),
        _feat(wi=0, rid="m1", dist=0.0, score=0.1),
        _feat(wi=1, rid="m0", dist=1.0, unavailable=0.0, score=0.9),
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


def test_cause_from_process_features_not_l2() -> None:
    starved = _feat(wi=0, rid="m0", turning=1, score=0.4, starved=40.0, shortage=0.5)
    idle = _feat(wi=0, rid="m1", turning=0, score=0.1)
    labels, _ = build_labels_and_events(
        [starved, idle], horizon=180.0, score_threshold=0.55, min_event_windows=1
    )
    assert labels[0]["root_cause_reason"] == "material_shortage"

    inbound = _feat(wi=0, rid="m0", turning=1, score=0.4, starved=40.0, inbound=40.0)
    labels, _ = build_labels_and_events(
        [inbound, idle], horizon=180.0, score_threshold=0.55, min_event_windows=1
    )
    assert labels[0]["root_cause_reason"] == "transport_delay"

    piled = _feat(wi=0, rid="m0", turning=0, score=0.4, starved=10.0, queue=2.0)
    labels, events = build_labels_and_events(
        [piled, idle], horizon=180.0, score_threshold=0.55, min_event_windows=1
    )
    assert len(events) == 1
    assert labels[0]["root_cause_reason"] == "queue_buildup"


def test_material_starve_is_process_event() -> None:
    """Sustained kitting WAITING + high shortage_propagation is an STGNPP event.

    Must not require a TPM turning-point (kit shortage starves grooving too).
    Must not fire from disturbance_active_s alone.
    """
    idle = _feat(wi=0, rid="m1", turning=0, score=0.1)
    waiting = _feat(
        wi=0, rid="num08_workbench_ws0", turning=0, score=0.18,
        starved=50.0, shortage=0.8,
    )
    labels, events = build_labels_and_events(
        [waiting, idle], horizon=180.0, score_threshold=0.55, min_event_windows=1
    )
    assert len(events) == 1
    assert events[0]["resource_id"] == "num08_workbench_ws0"
    assert events[0]["event_source"] == "score"
    hot = [l for l in labels if l["is_bottleneck_window"]]
    assert len(hot) == 1
    assert hot[0]["root_cause_reason"] == "material_shortage"

    dist_only = _feat(
        wi=0, rid="num08_workbench_ws0", turning=0, score=0.9, dist=1.0, unavailable=0.0
    )
    labels, events = build_labels_and_events(
        [dist_only, idle], horizon=180.0, score_threshold=0.55, min_event_windows=1
    )
    assert events == []
    assert all(l["is_bottleneck_window"] == 0 for l in labels)


def test_inbound_starve_and_blocked_are_process_events() -> None:
    idle = _feat(wi=0, rid="m1", turning=0, score=0.1)
    inbound = _feat(wi=0, rid="m0", turning=0, score=0.2, starved=30.0, inbound=40.0)
    labels, events = build_labels_and_events(
        [inbound, idle], horizon=180.0, score_threshold=0.55, min_event_windows=1
    )
    assert len(events) == 1
    assert events[0]["resource_id"] == "m0"
    assert labels[0]["root_cause_reason"] == "transport_delay"

    blocked = _feat(wi=0, rid="m0", turning=0, score=0.2, blocked=30.0, queue=1.0)
    labels, events = build_labels_and_events(
        [blocked, idle], horizon=180.0, score_threshold=0.55, min_event_windows=1
    )
    assert len(events) == 1
    assert labels[0]["root_cause_reason"] == "queue_buildup"


def test_shortage_fraction_ignores_warehouse_snapshots() -> None:
    from bn_agg.features import shortage_fraction_by_consumer

    wb = "num08_workbench_ws0"
    rows = [
        {
            "logic_time_s": "10",
            "material_type": "product_00_flange",
            "finished_task": "none",
            "shortage_flag": "0",
        },
        {
            "logic_time_s": "11",
            "material_type": "product_00_flange",
            "finished_task": "none",
            "shortage_flag": "0",
        },
        {
            "logic_time_s": "12",
            "material_type": "product_00_flange",
            "finished_task": "logistic_for_batch_spot_welding",
            "shortage_flag": "1",
        },
        {
            "logic_time_s": "13",
            "material_type": "product_00_elbow",
            "finished_task": "logistic_for_batch_spot_welding",
            "shortage_flag": "0",
        },
        {
            "logic_time_s": "40",
            "material_id": "material_0_product_00_flange",
            "material_type": "product_00_flange",
            "finished_task": "logistic_for_batch_spot_welding",
            "event": "consume",
            "shortage_flag": "1",
        },
        {
            "logic_time_s": "50",
            "material_id": "material_0_product_00_flange",
            "material_type": "product_00_flange",
            "finished_task": "logistic_for_batch_spot_welding",
            "event": "snapshot",
            "shortage_flag": "1",
        },
    ]
    frac = shortage_fraction_by_consumer(rows, 0.0, 60.0)
    assert frac[wb] >= 0.99
    assert frac["num08_workbench_ws1"] >= 0.99
    # Post-assemble snapshots in a later window must not look like shortage.
    frac_after = shortage_fraction_by_consumer(rows, 45.0, 60.0)
    assert frac_after.get(wb, 0.0) == 0.0


def test_process_downtime_and_gantry_not_event() -> None:
    idle = _feat(wi=0, rid="m1", turning=0, score=0.1)
    down = _feat(wi=0, rid="m0", dist=1.0, unavailable=1.0, score=0.2)
    labels, events = build_labels_and_events(
        [down, idle], horizon=180.0, score_threshold=0.55, min_event_windows=1
    )
    assert len(events) == 1
    assert events[0]["resource_id"] == "m0"
    assert events[0]["disturbance_type"] == ""
    assert labels[0]["is_bottleneck_window"] == 1

    gantry = _feat(
        wi=0, rid="gantry_0", resource_type="gantry",
        starved=40.0, inbound=40.0, score=0.4,
    )
    labels, events = build_labels_and_events(
        [gantry, idle], horizon=180.0, score_threshold=0.55, min_event_windows=1
    )
    assert len(events) == 1
    assert events[0]["resource_id"] == "gantry_0"
    assert labels[0]["is_bottleneck_window"] == 1

    gantry_starve = _feat(
        wi=0, rid="gantry_1", resource_type="gantry", starved=40.0, score=0.4,
    )
    labels, events = build_labels_and_events(
        [gantry_starve, idle], horizon=180.0, score_threshold=0.55, min_event_windows=1
    )
    assert events == []
    assert all(l["is_bottleneck_window"] == 0 for l in labels)

    gantry_l2 = _feat(
        wi=0, rid="gantry_2", resource_type="gantry",
        dist=1.0, unavailable=1.0, score=0.4,
    )
    labels, events = build_labels_and_events(
        [gantry_l2, idle], horizon=180.0, score_threshold=0.55, min_event_windows=1
    )
    assert len(events) == 1
    assert events[0]["resource_id"] == "gantry_2"
    assert labels[0]["is_bottleneck_window"] == 1

    agv_stop = _feat(
        wi=0, rid="robot_0", resource_type="transport_robot",
        unavailable=1.0, score=0.2,
    )
    labels, events = build_labels_and_events(
        [agv_stop, idle], horizon=180.0, score_threshold=0.55, min_event_windows=1
    )
    assert len(events) == 1
    assert events[0]["resource_id"] == "robot_0"


def test_coupled_stall_is_process_event() -> None:
    idle = _feat(wi=0, rid="m1", turning=0, score=0.1)
    stall = _feat(wi=0, rid="m0", starved=30.0, upstream=0.2, score=0.2)
    labels, events = build_labels_and_events(
        [stall, idle], horizon=180.0, score_threshold=0.55, min_event_windows=1
    )
    assert len(events) == 1
    assert events[0]["resource_id"] == "m0"


def test_robot_delay_not_dropped_by_gantry_complete() -> None:
    timelines = {
        "robot_0": ResourceTimeline(
            "robot_0",
            "transport_robot",
            [Interval(0.0, 60.0, "STARVED")],
        ),
        "gantry_1": ResourceTimeline(
            "gantry_1",
            "gantry",
            [Interval(0.0, 60.0, "PROCESSING")],
        ),
    }
    transport = [
        {
            "task_id": "0_logistic_for_pipe_cutting",
            "carrier_id": "robot_0",
            "carrier_type": "robot",
            "status": "delayed",
            "request_time_step": "0",
            "transport_start_time_step": "0",
            "transport_end_time_step": "",
            "dropoff_time_step": "",
            "to_node": "num02_rollerbedCNCPipeIntersectionCuttingMachine_ws0",
        },
        {
            "task_id": "0_logistic_for_pipe_cutting",
            "carrier_id": "gantry_1",
            "carrier_type": "gantry",
            "status": "completed",
            "request_time_step": "0",
            "transport_start_time_step": "0",
            "transport_end_time_step": "50",
            "dropoff_time_step": "50",
            "to_node": "num02_rollerbedCNCPipeIntersectionCuttingMachine_ws0",
        },
    ]
    rows = compute_window_features(
        timelines=timelines,
        job_rows=[],
        buffer_rows=[],
        transport_rows=transport,
        material_rows=[],
        window_size=60.0,
        episode_end=60.0,
        run_id="t",
        env_id=0,
        closed_windows_only=True,
    )
    robot = next(r for r in rows if r["resource_id"] == "robot_0")
    gantry = next(r for r in rows if r["resource_id"] == "gantry_1")
    assert float(robot["route_delay_s"]) >= 20.0
    # Gantry completed the trip while PROCESSING: travel is not delay.
    assert float(gantry["route_delay_s"]) < 20.0


def test_robot_travel_is_not_route_delay() -> None:
    """Open delayed task + driving (PROCESSING) must not paint the whole trip."""
    timelines = {
        "robot_0": ResourceTimeline(
            "robot_0",
            "transport_robot",
            [Interval(0.0, 60.0, "PROCESSING")],
        ),
    }
    transport = [
        {
            "task_id": "0_logistic_for_pipe_cutting",
            "carrier_id": "robot_0",
            "carrier_type": "robot",
            "status": "delayed",
            "delay_reason": "finding_free_gantry",
            "request_time_step": "0",
            "pickup_time_step": "",
            "transport_end_time_step": "50",
            "dropoff_time_step": "50",
            "to_node": "num02_rollerbedCNCPipeIntersectionCuttingMachine_ws0",
        },
    ]
    rows = compute_window_features(
        timelines=timelines,
        job_rows=[],
        buffer_rows=[],
        transport_rows=transport,
        material_rows=[],
        window_size=60.0,
        episode_end=60.0,
        run_id="t",
        env_id=0,
        closed_windows_only=True,
    )
    robot = next(r for r in rows if r["resource_id"] == "robot_0")
    assert float(robot["route_delay_s"]) < 5.0


def test_robot_delay_clipped_to_wait_not_trip() -> None:
    """Only STARVED/WAITING seconds count, not request→dropoff."""
    timelines = {
        "robot_0": ResourceTimeline(
            "robot_0",
            "transport_robot",
            [Interval(0.0, 20.0, "STARVED"), Interval(20.0, 60.0, "PROCESSING")],
        ),
    }
    transport = [
        {
            "task_id": "0_logistic_for_pipe_cutting",
            "carrier_id": "robot_0",
            "carrier_type": "robot",
            "status": "delayed",
            "delay_reason": "finding_free_gantry",
            "request_time_step": "0",
            "pickup_time_step": "",
            "transport_end_time_step": "60",
            "dropoff_time_step": "60",
            "to_node": "num02_rollerbedCNCPipeIntersectionCuttingMachine_ws0",
        },
    ]
    rows = compute_window_features(
        timelines=timelines,
        job_rows=[],
        buffer_rows=[],
        transport_rows=transport,
        material_rows=[],
        window_size=60.0,
        episode_end=60.0,
        run_id="t",
        env_id=0,
        closed_windows_only=True,
    )
    robot = next(r for r in rows if r["resource_id"] == "robot_0")
    delay = float(robot["route_delay_s"])
    assert 15.0 <= delay <= 25.0


def test_operator_wait_is_process_event() -> None:
    idle = _feat(wi=0, rid="m1", score=0.1)
    stalled = _feat(wi=0, rid="m0", starved=40.0, score=0.2)
    stalled["operator_absent_s"] = 1.0
    labels, events = build_labels_and_events(
        [stalled, idle], horizon=180.0, score_threshold=0.55, min_event_windows=1
    )
    assert len(events) == 1
    assert events[0]["resource_id"] == "m0"
    starve_only = _feat(wi=0, rid="m0", starved=40.0, score=0.2)
    labels2, events2 = build_labels_and_events(
        [starve_only, idle], horizon=180.0, score_threshold=0.55, min_event_windows=1
    )
    assert events2 == []
    assert labels2[0]["is_bottleneck_window"] == 0


def test_labor_saturated_operator_wait_is_process_event() -> None:
    idle = _feat(wi=0, rid="m1", score=0.1)
    stalled = _feat(wi=0, rid="m0", starved=40.0, score=0.2)
    stalled["labor_saturated_s"] = 1.0
    labels, events = build_labels_and_events(
        [stalled, idle], horizon=180.0, score_threshold=0.55, min_event_windows=1
    )
    assert len(events) == 1
    assert events[0]["resource_id"] == "m0"
    idle_human = _feat(wi=0, rid="m0", starved=40.0, score=0.2)
    idle_human["labor_saturated_s"] = 0.0
    labels2, events2 = build_labels_and_events(
        [idle_human, idle], horizon=180.0, score_threshold=0.55, min_event_windows=1
    )
    assert events2 == []
    assert labels2[0]["is_bottleneck_window"] == 0


def test_stamp_labor_saturated_ignores_unused_idle_slots() -> None:
    rows = [
        {
            "run_id": "t",
            "env_id": 0,
            "window_size_s": 60,
            "window_index": 0,
            "resource_id": "human_0",
            "resource_type": "human",
            "active_pct_s": 0.9,
            "unavailable_pct_s": 0.0,
        },
        {
            "run_id": "t",
            "env_id": 0,
            "window_size_s": 60,
            "window_index": 0,
            "resource_id": "human_3",
            "resource_type": "human",
            "active_pct_s": 0.0,
            "unavailable_pct_s": 0.0,
        },
        {
            "run_id": "t",
            "env_id": 0,
            "window_size_s": 60,
            "window_index": 0,
            "resource_id": "m0",
            "resource_type": "machine",
            "active_pct_s": 0.0,
            "unavailable_pct_s": 0.0,
        },
    ]
    _stamp_labor_saturated(rows)
    assert rows[2]["labor_saturated_s"] == 1.0
    rows[0]["active_pct_s"] = 0.1
    rows[0]["labor_saturated_s"] = 0.0
    _stamp_labor_saturated(rows)
    assert rows[2]["labor_saturated_s"] == 0.0


def test_absent_raw_state_is_stop_on_timeline() -> None:
    from bn_agg.timelines import build_timelines

    events = [
        {
            "resource_id": "robot_0",
            "resource_type": "transport_robot",
            "time_step": 10,
            "logic_time_s": 10.0,
            "to_state": "PROCESSING",
            "raw_to_state": "working_disturbance_absent",
        },
        {
            "resource_id": "robot_0",
            "resource_type": "transport_robot",
            "time_step": 40,
            "logic_time_s": 40.0,
            "to_state": "IDLE",
            "raw_to_state": "free",
        },
    ]
    tls = build_timelines(events, episode_end=60.0)
    ivs = tls["robot_0"].intervals
    states = [(iv.start, iv.end, iv.state) for iv in ivs]
    assert any(s == "STOP" and a <= 10.0 and b >= 40.0 for a, b, s in states)


if __name__ == "__main__":
    test_closed_windows_skip_partial()
    test_open_disturbance_truncated()
    test_l2_is_context_not_event()
    test_turning_point_is_process_event()
    test_cause_from_process_features_not_l2()
    test_material_starve_is_process_event()
    test_inbound_starve_and_blocked_are_process_events()
    test_shortage_fraction_ignores_warehouse_snapshots()
    test_process_downtime_and_gantry_not_event()
    test_coupled_stall_is_process_event()
    test_robot_delay_not_dropped_by_gantry_complete()
    test_robot_travel_is_not_route_delay()
    test_robot_delay_clipped_to_wait_not_trip()
    test_operator_wait_is_process_event()
    test_labor_saturated_operator_wait_is_process_event()
    test_stamp_labor_saturated_ignores_unused_idle_slots()
    test_absent_raw_state_is_stop_on_timeline()
    print("ok")
