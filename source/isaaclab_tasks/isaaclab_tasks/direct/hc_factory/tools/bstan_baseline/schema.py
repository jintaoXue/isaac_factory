"""Stable feature and tensor schema for the BSTAN baseline."""

from __future__ import annotations

from factory_bn_shared.contract import (
    DERIVED_CONTRACT_VERSION,
    RAW_COLLECTOR_VERSION,
    SHARED_LABEL_VERSION,
)

DATASET_VERSION = "bstan_tyxbn_dataset_v1"
LABEL_VERSION = SHARED_LABEL_VERSION
COLLECTOR_VERSION = RAW_COLLECTOR_VERSION
DATASET_CONTRACT = DERIVED_CONTRACT_VERSION
TARGET_NODE_CATEGORY = "process"

CONTINUOUS_FEATURES = (
    "queue_length_s",
    "avg_waiting_time_s",
    "occupancy_ratio_s",
    "queue_growth_rate_s",
    "active_pct_s",
    "current_active_duration_s",
    "blocked_time_s",
    "starved_time_s",
    "stop_time_s",
    "unavailable_pct_s",
    "inter_departure_var_s",
    "upstream_blocked_ratio_s",
    "downstream_starved_ratio_s",
    "route_delay_s",
    "inbound_wait_s",
    "material_shortage_propagation_s",
    "affiliated_buffer_occ_s",
    "tb_minus_ts_s",
    "disturbance_active_s",
    "is_turning_point",
    "is_momentary_bn",
)

GLOBAL_FEATURES: tuple[str, ...] = ()


def is_buffer(resource_id: str, resource_type: str) -> bool:
    return resource_type == "buffer" or resource_id.startswith("storage_")


def feature_is_applicable(
    feature_name: str, resource_id: str, resource_type: str
) -> bool:
    del feature_name, resource_id, resource_type
    return True
