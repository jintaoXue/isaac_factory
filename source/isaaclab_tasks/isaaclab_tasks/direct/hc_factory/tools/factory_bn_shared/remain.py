"""Canonical A.1 target and metric definitions from the main experiment."""

from . import canonical as _canonical  # noqa: F401
from factory_bn.remain import (  # noqa: E402,F401
    ensure_labor_saturated_feature,
    first_done_index,
    gaussian_start_soft_labels,
    jobs_remaining_series,
    node_event_targets,
    occupancy_event_metrics,
    occupancy_node_mask,
    occupancy_to_events,
    ops_hot_mask,
    pack_remain_target,
    smooth_occupancy_runs,
    station_report_metrics,
)

__all__ = [
    "ensure_labor_saturated_feature",
    "first_done_index",
    "gaussian_start_soft_labels",
    "jobs_remaining_series",
    "node_event_targets",
    "occupancy_event_metrics",
    "occupancy_node_mask",
    "occupancy_to_events",
    "ops_hot_mask",
    "pack_remain_target",
    "smooth_occupancy_runs",
    "station_report_metrics",
]
