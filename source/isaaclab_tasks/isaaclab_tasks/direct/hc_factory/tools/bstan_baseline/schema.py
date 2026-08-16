"""Stable feature and tensor schema for the BSTAN baseline."""

from __future__ import annotations

from canonical_factory_bn.contract import (
    CANONICAL_CONTRACT_VERSION,
    CANONICAL_LABEL_VERSION,
    RAW_COLLECTOR_VERSION,
)

DATASET_VERSION = "bstan_canonical_dataset_v1"
LABEL_VERSION = CANONICAL_LABEL_VERSION
COLLECTOR_VERSION = RAW_COLLECTOR_VERSION
DATASET_CONTRACT = CANONICAL_CONTRACT_VERSION
TARGET_NODE_CATEGORY = "process"

CONTINUOUS_FEATURES = (
    "observation_available_s",
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
    "operation_throughput_rolling",
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
