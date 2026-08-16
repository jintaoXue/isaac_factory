"""Static graph matching dev_tyx PDFormer's factory graph prior."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch

from bn_agg.constants import BUFFER_MACHINE_AFFINITY, PROCESS_CHAIN


def build_static_graph(
    node_ids: list[str],
    node_types: dict[str, str],
    active_nodes: set[str],
    episode_config: dict[str, Any],
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    del episode_config
    index = {node_id: i for i, node_id in enumerate(node_ids)}
    edges: dict[tuple[str, str], tuple[str, float]] = {}

    def add(source: str, target: str, edge_type: str, weight: float) -> None:
        if source not in active_nodes or target not in active_nodes:
            return
        for pair in ((source, target), (target, source)):
            previous = edges.get(pair)
            if previous is None or weight > previous[1]:
                edges[pair] = (edge_type, weight)

    chain = [node_id for node_id in PROCESS_CHAIN if node_id in active_nodes]
    for source, target in zip(chain, chain[1:]):
        add(source, target, "process_flow", 1.0)

    families: dict[str, list[str]] = defaultdict(list)
    for node_id in active_nodes:
        if "_ws" in node_id:
            families[node_id.rsplit("_ws", 1)[0]].append(node_id)
    for members in families.values():
        for i, source in enumerate(members):
            for target in members[i + 1 :]:
                add(source, target, "machine_sibling", 0.8)

    for buffer_id in active_nodes:
        if not buffer_id.startswith("storage_"):
            continue
        for key, machines in BUFFER_MACHINE_AFFINITY.items():
            if key in buffer_id:
                for machine in machines:
                    add(buffer_id, machine, "buffer_affinity", 0.6)
                break

    machines = [
        node_id
        for node_id in active_nodes
        if node_types.get(node_id) == "machine" or "_ws" in node_id
    ]
    agents = [
        node_id
        for node_id in active_nodes
        if node_types.get(node_id) in {"gantry", "human", "transport_robot"}
        or node_id.startswith(("gantry_", "human_", "robot_"))
    ]
    for agent in agents:
        for machine in machines:
            add(agent, machine, "agent_machine", 0.5)

    for resource_type in ("gantry", "transport_robot"):
        members = [
            node_id for node_id in active_nodes if node_types.get(node_id) == resource_type
        ]
        for i, source in enumerate(members):
            for target in members[i + 1 :]:
                add(source, target, "agent_same_type", 0.3)

    for node_id in active_nodes:
        edges[(node_id, node_id)] = ("self_loop", 1.0)

    adjacency = torch.zeros((len(node_ids), len(node_ids)), dtype=torch.bool)
    rows = []
    for (source, target), (edge_type, weight) in sorted(edges.items()):
        adjacency[index[source], index[target]] = True
        rows.append(
            {
                "source_node_id": source,
                "source_node_index": index[source],
                "target_node_id": target,
                "target_node_index": index[target],
                "edge_type": edge_type,
                "edge_weight": weight,
            }
        )
    return adjacency, rows
