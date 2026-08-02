"""Build a heterogeneous process/logistics graph for HC Factory resources."""

from __future__ import annotations

from typing import Iterable

import numpy as np


# ProductWaterPipe primary process chain (workstation-level resource_ids).
PROCESS_CHAIN: list[str] = [
    "num02_rollerbedCNCPipeIntersectionCuttingMachine_ws0",
    "num04_groovingMachineLarge_ws0",
    "num08_workbench_ws0",
    "num08_workbench_ws1",
    "num01_weldingRobot_ws0",
    "num00_rotaryPipeAutomaticWeldingMachine_ws0",
    "num00_rotaryPipeAutomaticWeldingMachine_ws1",
]

# Logical buffer ↔ machine affinities (name heuristics).
BUFFER_MACHINE_AFFINITY: dict[str, list[str]] = {
    "BlackStorage": [
        "num02_rollerbedCNCPipeIntersectionCuttingMachine_ws0",
        "num04_groovingMachineLarge_ws0",
    ],
    "YellowStorage": [
        "num08_workbench_ws0",
        "num08_workbench_ws1",
        "num01_weldingRobot_ws0",
        "num00_rotaryPipeAutomaticWeldingMachine_ws0",
        "num00_rotaryPipeAutomaticWeldingMachine_ws1",
    ],
    "GroundStorage": [
        "num02_rollerbedCNCPipeIntersectionCuttingMachine_ws0",
        "num04_groovingMachineLarge_ws0",
        "num08_workbench_ws0",
    ],
}

RESOURCE_TYPES = ("machine", "gantry", "human", "transport_robot", "buffer")


def type_onehot(resource_type: str) -> list[float]:
    vec = [0.0] * len(RESOURCE_TYPES)
    if resource_type in RESOURCE_TYPES:
        vec[RESOURCE_TYPES.index(resource_type)] = 1.0
    return vec


def _add_undirected(adj: np.ndarray, i: int, j: int, w: float = 1.0) -> None:
    if i == j:
        return
    adj[i, j] = max(adj[i, j], w)
    adj[j, i] = max(adj[j, i], w)


def build_factory_adjacency(
    resource_ids: list[str],
    resource_types: list[str] | None = None,
) -> np.ndarray:
    """Return symmetric adjacency ``(N, N)`` for factory resources.

    Edge families
    -------------
    1. process flow along ``PROCESS_CHAIN``
    2. buffer ↔ affiliated machines
    3. transport agents (gantry / human / robot) ↔ all machines
    4. same-type soft coupling among machines / gantries / robots
    """
    n = len(resource_ids)
    idx = {rid: i for i, rid in enumerate(resource_ids)}
    types = resource_types or [""] * n
    adj = np.zeros((n, n), dtype=np.float64)

    # 1) process chain
    chain = [rid for rid in PROCESS_CHAIN if rid in idx]
    for a, b in zip(chain, chain[1:]):
        _add_undirected(adj, idx[a], idx[b], 1.0)

    # sibling workstations of the same machine family
    family_groups: dict[str, list[str]] = {}
    for rid in resource_ids:
        if "_ws" in rid:
            family = rid.rsplit("_ws", 1)[0]
            family_groups.setdefault(family, []).append(rid)
    for members in family_groups.values():
        for a in members:
            for b in members:
                if a != b and a in idx and b in idx:
                    _add_undirected(adj, idx[a], idx[b], 0.8)

    # 2) buffer affinities
    for rid, i in idx.items():
        if not rid.startswith("storage_"):
            continue
        for key, machines in BUFFER_MACHINE_AFFINITY.items():
            if key in rid:
                for m in machines:
                    if m in idx:
                        _add_undirected(adj, i, idx[m], 0.6)
                break

    # 3) logistics agents ↔ machines
    machine_ids = [rid for rid, t in zip(resource_ids, types) if t == "machine" or "_ws" in rid]
    agent_ids = [
        rid
        for rid, t in zip(resource_ids, types)
        if t in ("gantry", "human", "transport_robot")
        or rid.startswith(("gantry_", "human_", "robot_"))
    ]
    for agent in agent_ids:
        for m in machine_ids:
            _add_undirected(adj, idx[agent], idx[m], 0.5)

    # 4) same-type soft clique for agents
    for t_name in ("gantry", "transport_robot"):
        members = [rid for rid, t in zip(resource_ids, types) if t == t_name]
        for a in members:
            for b in members:
                if a != b:
                    _add_undirected(adj, idx[a], idx[b], 0.3)

    # self-loops help LapPE stability
    np.fill_diagonal(adj, 1.0)
    return adj


def hop_distance_matrix(adj: np.ndarray, unreachable: int = 511) -> np.ndarray:
    """Floyd–Warshall hop distances on a weighted adjacency (edge if >0)."""
    n = adj.shape[0]
    hops = np.full((n, n), unreachable, dtype=np.float64)
    hops[adj > 0] = 1.0
    np.fill_diagonal(hops, 0.0)
    for k in range(n):
        for i in range(n):
            for j in range(n):
                hops[i, j] = min(hops[i, j], hops[i, k] + hops[k, j])
    return hops


def semantic_distance_matrix(series: np.ndarray) -> np.ndarray:
    """Pairwise correlation distance on ``(T, N, F)`` mean feature trajectories.

    Used as a DTW substitute for short factory episodes (no calendar days).
    """
    # mean over feature channels → (T, N)
    x = series.mean(axis=-1)
    n = x.shape[1]
    dist = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i, n):
            a, b = x[:, i], x[:, j]
            if a.std() < 1e-8 or b.std() < 1e-8:
                d = 1.0
            else:
                corr = np.corrcoef(a, b)[0, 1]
                d = 1.0 - float(corr if np.isfinite(corr) else 0.0)
            dist[i, j] = dist[j, i] = d
    return dist


def iter_edge_list(adj: np.ndarray, resource_ids: Iterable[str]) -> list[tuple[int, int, float]]:
    ids = list(resource_ids)
    edges = []
    n = adj.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] > 0 and i != j:
                edges.append((i, j, float(adj[i, j])))
    return edges
