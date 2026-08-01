"""Static manufacturing-prior graph construction."""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

import torch

from .schema import is_buffer


def _json_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _canonical_buffer_id(value: Any) -> str:
    node_id = str(value or "").strip()
    if node_id.startswith("storage_"):
        return node_id
    return f"storage_{node_id}" if node_id else ""


def _matching_machine_nodes(machine: str, active_nodes: set[str]) -> list[str]:
    return sorted(
        node_id
        for node_id in active_nodes
        if node_id == machine or node_id.startswith(f"{machine}_ws")
    )


def _material_aliases(value: Any) -> set[str]:
    material = str(value or "").strip().lower()
    aliases = {material}
    parts = material.split("_")
    if len(parts) >= 3 and parts[0] == "product" and parts[1].isdigit():
        aliases.add("_".join(parts[2:]))
    if material.endswith("_semi") or material.endswith("_maded"):
        aliases.add("product_water_pipe")
    return {alias for alias in aliases if alias}


def build_static_graph(
    node_ids: list[str],
    node_types: dict[str, str],
    active_nodes: set[str],
    episode_config: dict[str, Any],
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    """Build one episode graph using only static configuration and node types."""
    node_index = {node_id: index for index, node_id in enumerate(node_ids)}
    edge_types: dict[tuple[str, str], set[str]] = defaultdict(set)

    def add_edge(
        source: str, target: str, edge_type: str, bidirectional: bool = True
    ) -> None:
        if source not in active_nodes or target not in active_nodes:
            return
        edge_types[(source, target)].add(edge_type)
        if bidirectional and source != target:
            edge_types[(target, source)].add(edge_type)

    for node_id in active_nodes:
        add_edge(node_id, node_id, "self_loop", bidirectional=False)

    process_cfg = _json_dict(episode_config.get("process_time_config"))
    buffer_cfg = _json_dict(episode_config.get("buffer_capacity_config"))

    required_by_machine: dict[str, set[str]] = defaultdict(set)
    for product_steps in process_cfg.values():
        if not isinstance(product_steps, dict):
            continue
        product_sequence: list[tuple[list[str], set[str]]] = []
        for step_config in product_steps.values():
            if not isinstance(step_config, dict):
                continue
            machine = str(step_config.get("machine") or "")
            machines = _matching_machine_nodes(machine, active_nodes)
            required_raw = step_config.get("required_materials") or {}
            required_values = (
                required_raw.keys() if isinstance(required_raw, dict) else required_raw
            )
            required = {
                alias for item in required_values for alias in _material_aliases(item)
            }
            product_sequence.append((machines, required))
            for machine_node in machines:
                required_by_machine[machine_node].update(str(item) for item in required)
        for current, following in zip(product_sequence, product_sequence[1:]):
            for source in current[0]:
                for target in following[0]:
                    add_edge(source, target, "process_flow")
    for raw_buffer_id, config in buffer_cfg.items():
        if not isinstance(config, dict):
            continue
        buffer_id = _canonical_buffer_id(raw_buffer_id)
        supported_raw = config.get("supporting_materials") or []
        supported_values = (
            [supported_raw] if isinstance(supported_raw, str) else supported_raw
        )
        supported = {
            alias for item in supported_values for alias in _material_aliases(item)
        }
        for machine_node, required in required_by_machine.items():
            if supported.intersection(required):
                add_edge(buffer_id, machine_node, "buffer_supply")

    machine_nodes = {
        node_id for node_id in active_nodes if node_types.get(node_id) == "machine"
    }
    buffer_nodes = {
        node_id
        for node_id in active_nodes
        if is_buffer(node_id, node_types.get(node_id, "unknown"))
    }
    human_nodes = {
        node_id for node_id in active_nodes if node_types.get(node_id) == "human"
    }
    robot_nodes = {
        node_id
        for node_id in active_nodes
        if node_types.get(node_id) in {"robot", "transport_robot", "agv"}
    }
    gantry_nodes = {
        node_id for node_id in active_nodes if node_types.get(node_id) == "gantry"
    }

    for human in human_nodes:
        for machine in machine_nodes:
            add_edge(human, machine, "human_capability")
    for robot in robot_nodes:
        for target in machine_nodes | buffer_nodes:
            add_edge(robot, target, "robot_capability")
    for gantry in gantry_nodes:
        for target in machine_nodes | buffer_nodes:
            add_edge(gantry, target, "gantry_capability")

    adjacency = torch.zeros((len(node_ids), len(node_ids)), dtype=torch.bool)
    rows: list[dict[str, Any]] = []
    for (source, target), types in sorted(edge_types.items()):
        adjacency[node_index[source], node_index[target]] = True
        for edge_type in sorted(types):
            rows.append(
                {
                    "source_node_id": source,
                    "source_node_index": node_index[source],
                    "target_node_id": target,
                    "target_node_index": node_index[target],
                    "edge_type": edge_type,
                }
            )
    return adjacency, rows
