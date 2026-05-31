from ..env_asset_cfg.cfg_route.cfg_route import CfgRoute, OptionalInitPointIds
import heapq
import json
import math
from pathlib import Path
import torch


class _RoadmapGraph:
    """Parser and route generator for one map_routes_*.json file."""

    def __init__(self, routes_data: dict, default_points_path: str, agent_label: str):
        self.routes_data = routes_data
        self.agent_label = agent_label
        self.map_path = routes_data.get("map_path")
        self.points_path = routes_data.get("points_path", default_points_path)
        self.num_nodes = int(routes_data.get("num_nodes", 0))
        self.coordinate_frame = routes_data.get("coordinate_frame", "isaac_sim")

        self.edges: list[dict] = routes_data.get("edges", [])
        self.paths: dict = routes_data.get("paths", {})
        self.edge_by_uv = self._index_edges_by_uv(self.edges)
        self.graph_adjacency = self._build_adjacency_from_edges(self.edges)
        self.graph_node_ids = self._collect_graph_node_ids()
        self.point_xy_by_id = self._load_point_xy_by_id(self.points_path)

    @staticmethod
    def _index_edges_by_uv(edges: list[dict]) -> dict[tuple[int, int], dict]:
        edge_by_uv: dict[tuple[int, int], dict] = {}
        for edge in edges:
            if not isinstance(edge, dict):
                continue
            samples = edge.get("samples")
            if not isinstance(samples, list) or not samples:
                continue
            edge_by_uv[(int(edge["u"]), int(edge["v"]))] = edge
        return edge_by_uv

    @staticmethod
    def _build_adjacency_from_edges(edges: list[dict]) -> dict[int, list[tuple[int, float]]]:
        adjacency: dict[int, list[tuple[int, float]]] = {}
        for edge in edges:
            if not isinstance(edge, dict):
                continue
            node_u = int(edge["u"])
            node_v = int(edge["v"])
            edge_length = float(edge.get("length", 1.0))
            adjacency.setdefault(node_u, []).append((node_v, edge_length))
            adjacency.setdefault(node_v, []).append((node_u, edge_length))
        return adjacency

    def _collect_graph_node_ids(self) -> set[int]:
        node_ids: set[int] = set()
        for edge in self.edges:
            node_ids.add(int(edge["u"]))
            node_ids.add(int(edge["v"]))
        for src_id in self.paths.keys():
            node_ids.add(int(src_id))
            for dst_id in self.paths[src_id].keys():
                node_ids.add(int(dst_id))
        return node_ids

    @staticmethod
    def _load_point_xy_by_id(points_path: str) -> dict[int, tuple[float, float]]:
        points_file = Path(points_path).expanduser()
        points_data = json.loads(points_file.read_text(encoding="utf-8"))
        points = points_data.get("points", points_data)
        return {int(point["id"]): (float(point["x"]), float(point["y"])) for point in points}

    def generate_route(self, start_id: int, end_id: int) -> dict:
        start_node_id = self._resolve_graph_node_id(int(start_id))
        end_node_id = self._resolve_graph_node_id(int(end_id))

        if start_node_id == end_node_id:
            return self._build_route_result(
                start_id=start_id,
                end_id=end_id,
                start_node_id=start_node_id,
                end_node_id=end_node_id,
                node_ids=[start_node_id],
                route=[self._pose_at_node(start_node_id)],
                path_cost=0.0,
            )

        node_ids, path_cost = self._find_shortest_path_nodes(start_node_id, end_node_id)
        route = self._stitch_route_from_node_ids(node_ids)
        return self._build_route_result(
            start_id=start_id,
            end_id=end_id,
            start_node_id=start_node_id,
            end_node_id=end_node_id,
            node_ids=node_ids,
            route=route,
            path_cost=path_cost,
        )

    def _resolve_graph_node_id(self, area_id: int) -> int:
        if area_id in self.graph_node_ids:
            return area_id
        return self._find_nearest_graph_node_id(area_id)

    def _find_nearest_graph_node_id(self, area_id: int) -> int:
        query_xy = self.point_xy_by_id.get(area_id)
        if query_xy is None:
            raise ValueError(f"Unknown {self.agent_label} map point id: {area_id}")

        best_node_id = None
        best_dist_sq = math.inf
        query_x, query_y = query_xy
        for node_id in self.graph_node_ids:
            node_xy = self.point_xy_by_id.get(node_id)
            if node_xy is None:
                continue
            dist_sq = (node_xy[0] - query_x) ** 2 + (node_xy[1] - query_y) ** 2
            if dist_sq < best_dist_sq:
                best_dist_sq = dist_sq
                best_node_id = node_id

        if best_node_id is None:
            raise ValueError(f"No {self.agent_label} graph node found near map point id: {area_id}")
        return best_node_id

    def _find_shortest_path_nodes(self, start_node_id: int, end_node_id: int) -> tuple[list[int], float]:
        path_info = self.paths.get(str(start_node_id), {}).get(str(end_node_id))
        if path_info is not None:
            node_ids = path_info.get("nodes")
            if isinstance(node_ids, list) and node_ids:
                return [int(node_id) for node_id in node_ids], float(path_info.get("cost", 0.0))

        node_ids = self._dijkstra_path_nodes(start_node_id, end_node_id)
        if not node_ids:
            raise ValueError(f"No {self.agent_label} route found: {start_node_id} -> {end_node_id}")
        return node_ids, self._estimate_path_cost(node_ids)

    def _dijkstra_path_nodes(self, start_node_id: int, end_node_id: int) -> list[int]:
        if start_node_id == end_node_id:
            return [start_node_id]

        distances: dict[int, float] = {start_node_id: 0.0}
        previous: dict[int, int] = {}
        priority_queue: list[tuple[float, int]] = [(0.0, start_node_id)]

        while priority_queue:
            current_dist, current_node = heapq.heappop(priority_queue)
            if current_dist > distances.get(current_node, math.inf):
                continue
            if current_node == end_node_id:
                break
            for neighbor_node, edge_length in self.graph_adjacency.get(current_node, []):
                next_dist = current_dist + edge_length
                if next_dist < distances.get(neighbor_node, math.inf):
                    distances[neighbor_node] = next_dist
                    previous[neighbor_node] = current_node
                    heapq.heappush(priority_queue, (next_dist, neighbor_node))

        if end_node_id not in distances:
            return []

        path_nodes = [end_node_id]
        cursor = end_node_id
        while cursor != start_node_id:
            cursor = previous.get(cursor)
            if cursor is None:
                return []
            path_nodes.append(cursor)
        path_nodes.reverse()
        return path_nodes

    def _estimate_path_cost(self, node_ids: list[int]) -> float:
        path_cost = 0.0
        for node_from, node_to in zip(node_ids[:-1], node_ids[1:]):
            edge, _ = self._get_edge(node_from, node_to)
            if edge is not None:
                path_cost += float(edge.get("length", 0.0))
        return path_cost

    def _pose_at_node(self, node_id: int) -> dict[str, float]:
        node_xy = self.point_xy_by_id.get(node_id)
        if node_xy is not None:
            return {"x": node_xy[0], "y": node_xy[1], "yaw": 0.0}

        for edge in self.edges:
            if int(edge["u"]) == node_id:
                return self._sample_to_pose(edge["samples"][0], forward=True)
            if int(edge["v"]) == node_id:
                return self._sample_to_pose(edge["samples"][-1], forward=False)
        raise ValueError(f"Unknown {self.agent_label} map node id: {node_id}")

    def _stitch_route_from_node_ids(self, node_ids: list[int]) -> list[dict[str, float]]:
        route: list[dict[str, float]] = []
        for segment_idx in range(len(node_ids) - 1):
            node_from = node_ids[segment_idx]
            node_to = node_ids[segment_idx + 1]
            edge, forward = self._get_edge(node_from, node_to)
            if edge is None:
                raise ValueError(f"Missing {self.agent_label} edge samples: {node_from} -> {node_to}")

            samples = edge["samples"] if forward else list(reversed(edge["samples"]))
            for sample_idx, sample in enumerate(samples):
                if segment_idx > 0 and sample_idx == 0:
                    continue
                route.append(self._sample_to_pose(sample, forward=forward))

        if not route:
            route.append(self._pose_at_node(node_ids[-1]))
        return route

    def _get_edge(self, node_from: int, node_to: int) -> tuple[dict | None, bool]:
        if (node_from, node_to) in self.edge_by_uv:
            return self.edge_by_uv[(node_from, node_to)], True
        if (node_to, node_from) in self.edge_by_uv:
            return self.edge_by_uv[(node_to, node_from)], False
        return None, True

    @staticmethod
    def _sample_to_pose(sample: dict, forward: bool) -> dict[str, float]:
        yaw = float(sample.get("yaw", 0.0))
        if not forward:
            yaw += math.pi
            if yaw > math.pi:
                yaw -= 2 * math.pi
            elif yaw < -math.pi:
                yaw += 2 * math.pi
        return {
            "x": float(sample["x"]),
            "y": float(sample["y"]),
            "yaw": yaw,
        }

    @staticmethod
    def _build_route_result(
        start_id: int,
        end_id: int,
        start_node_id: int,
        end_node_id: int,
        node_ids: list[int],
        route: list[dict[str, float]],
        path_cost: float,
    ) -> dict:
        return {
            "start_id": start_id,
            "end_id": end_id,
            "start_node_id": start_node_id,
            "end_node_id": end_node_id,
            "node_ids": node_ids,
            "path_cost": path_cost,
            "route": route,
        }


class RouteManagerVectorEnv:
    """Route manager for human and robot using map_routes_*.json precomputed roadmap data."""

    def __init__(self, cuda_device: torch.device):
        self.cuda_device = cuda_device
        self.cfg_route = CfgRoute

        routes_human = json.load(Path(self.cfg_route["routes_path_human"]).expanduser().open("r", encoding="utf-8"))
        routes_robot = json.load(Path(self.cfg_route["routes_path_robot"]).expanduser().open("r", encoding="utf-8"))

        self.human_roadmap = _RoadmapGraph(
            routes_human,
            default_points_path=self.cfg_route["points_path_human"],
            agent_label="human",
        )
        self.robot_roadmap = _RoadmapGraph(
            routes_robot,
            default_points_path=self.cfg_route["points_path_robot"],
            agent_label="robot",
        )
        self.default_z_human = OptionalInitPointIds["human_z"]
        self.default_z_robot = OptionalInitPointIds["robot_z"]

    def step(self, env_state_action_dict: dict) -> dict:
        self._update_agent_routes(env_state_action_dict["human"], self.generate_human_route)
        self._update_agent_routes(env_state_action_dict["robot"], self.generate_robot_route)
        self._step_next_pose(env_state_action_dict["human"])
        self._step_next_pose(env_state_action_dict["robot"])
        
        return env_state_action_dict


    def _step_next_pose(self, agent_states: dict) -> dict:
        for agent_name, agent_state in agent_states.items():
            route_index = agent_state["route_index"]
            route_length = agent_state["route_length"]
            if route_index >= route_length:
                #arrived at the target area
                agent_state["current_area_id"] = agent_state["target_area_id"]
                return
            route = agent_state["generated_route"]
            xy_yaw = route[route_index]
            agent_state["position"] = torch.tensor([xy_yaw[0], xy_yaw[1], self.default_z_human], dtype=torch.float32, device=self.cuda_device)
            agent_state["orientation"] = torch.tensor([0, 0, xy_yaw[2]], dtype=torch.float32, device=self.cuda_device)
            agent_state["route_index"] += 1

    @staticmethod
    def _update_agent_routes(agent_states: dict, route_generator) -> None:
        for agent_state in agent_states.values():
            state = agent_state["state"]
            if state["ongoing_task_record_index"] is None:
                continue
            if (
                state["current_area_id"] != state["target_area_id"]
                and state["target_area_id"] is not None
                and len(state["generated_route"]) == 0
            ):
                route_info = route_generator(state["current_area_id"], state["target_area_id"])
                state["generated_route"] = route_info["route"]
                state["route_index"] = 0
                state["route_length"] = len(route_info["route"])

    def generate_human_route(self, start_id: int, end_id: int) -> dict:
        return self.human_roadmap.generate_route(start_id, end_id)

    def generate_robot_route(self, start_id: int, end_id: int) -> dict:
        return self.robot_roadmap.generate_route(start_id, end_id)
