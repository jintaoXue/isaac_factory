"""Stable feature and tensor schema for the BSTAN baseline."""

from __future__ import annotations


DATASET_VERSION = "bstan_dataset_v1"
LABEL_VERSION = "bstan_weak_v2_1"
SUPPORTED_LABEL_VERSIONS = frozenset(
    {"bstan_weak_v1", "bstan_weak_v2", "bstan_weak_v2_1"}
)

CONTINUOUS_FEATURES = (
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

GLOBAL_FEATURES = (
    "total_WIP",
    "throughput_rolling",
    "num_busy_resources",
    "num_blocked_resources",
    "num_starved_resources",
    "disturbance_flag",
)

BUFFER_ONLY_FEATURES = frozenset(
    {
        "occupancy_ratio_s",
        "queue_growth_rate_s",
    }
)

PROCESS_ONLY_FEATURES = frozenset(
    {
        "active_pct_s",
        "blocked_ratio_s",
        "starved_ratio_s",
        "current_active_duration_s",
        "output_rate_s",
    }
)


def is_buffer(resource_id: str, resource_type: str) -> bool:
    return resource_type == "buffer" or resource_id.startswith("storage_")


def feature_is_applicable(
    feature_name: str, resource_id: str, resource_type: str
) -> bool:
    buffer_node = is_buffer(resource_id, resource_type)
    if feature_name in BUFFER_ONLY_FEATURES:
        return buffer_node
    if feature_name in PROCESS_ONLY_FEATURES:
        return not buffer_node
    return True
