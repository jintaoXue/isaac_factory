from ..env_asset_cfg.cfg_route.cfg_route import CfgRoute, OptionalInitPointIds
from .utils import quat_multiply_wxyz, quaternion_wxyz_to_yaw, yaw_to_quaternion_wxyz
from . import route_collision as rc
import heapq
import json
import math
from pathlib import Path
import torch
from ..env_asset_cfg.cfg_human import CfgHuman
from ..env_asset_cfg.cfg_machine import CfgMachine

_CPU_DEVICE = torch.device("cpu")


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

    def get_xy_at_area_id(self, area_id: int) -> tuple[float, float]:
        """Map point id -> Isaac Sim (x, y). Uses exact id first, then nearest graph node."""
        area_id = int(area_id)
        point_xy = self.point_xy_by_id.get(area_id)
        if point_xy is not None:
            return point_xy
        nearest_node_id = self._find_nearest_graph_node_id(area_id)
        node_xy = self.point_xy_by_id.get(nearest_node_id)
        if node_xy is not None:
            return node_xy
        raise ValueError(f"Unknown {self.agent_label} map point id: {area_id}")

    def find_nearest_area_id(self, x: float, y: float) -> int:
        best_area_id = None
        best_dist_sq = math.inf
        for area_id, point_xy in self.point_xy_by_id.items():
            dist_sq = (point_xy[0] - x) ** 2 + (point_xy[1] - y) ** 2
            if dist_sq < best_dist_sq:
                best_dist_sq = dist_sq
                best_area_id = int(area_id)
        if best_area_id is None:
            raise ValueError(f"No {self.agent_label} map point found near ({x}, {y})")
        return best_area_id

    def _pose_at_node(self, node_id: int) -> dict:
        node_xy = self.point_xy_by_id.get(node_id)
        if node_xy is not None:
            return {
                "x": node_xy[0],
                "y": node_xy[1],
                "orientation": yaw_to_quaternion_wxyz(0.0, _CPU_DEVICE),
            }

        for edge in self.edges:
            if int(edge["u"]) == node_id:
                return self._sample_to_pose(edge["samples"][0], forward=True)
            if int(edge["v"]) == node_id:
                return self._sample_to_pose(edge["samples"][-1], forward=False)
        raise ValueError(f"Unknown {self.agent_label} map node id: {node_id}")

    def _stitch_route_from_node_ids(self, node_ids: list[int]) -> list[dict]:
        route: list[dict] = []
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

    def _sample_to_pose(self, sample: dict, forward: bool) -> dict:
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
            "orientation": yaw_to_quaternion_wxyz(yaw, _CPU_DEVICE),
        }

    @staticmethod
    def _build_route_result(
        start_id: int,
        end_id: int,
        start_node_id: int,
        end_node_id: int,
        node_ids: list[int],
        route: list[dict],
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
        self.collision_cfg = self.cfg_route["collision_avoidance"]
        self.human_route_orientation_offset = CfgHuman["NormalHuman"]["human_route_orientation_offset"]["orientation"].to(_CPU_DEVICE)
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
        free_threshold = int(self.collision_cfg["occupancy_free_threshold"])
        png_coords = self.cfg_route["png_image_coordinates"]
        isaac_coords = self.cfg_route["isaac_sim_coordinates"]
        self._occupancy_human = rc.OccupancyMap(
            self.cfg_route["map_path_human"], png_coords, isaac_coords, free_threshold
        )
        self._occupancy_robot = rc.OccupancyMap(
            self.cfg_route["map_path_robot"], png_coords, isaac_coords, free_threshold
        )

    def step(self, env_state_action_dict: dict) -> dict:
        self._generate_working_agent_routes(env_state_action_dict["human"], self.generate_working_human_route)
        self._generate_working_agent_routes(env_state_action_dict["robot"], self.generate_working_robot_route)
        self.agents_collision_check(env_state_action_dict)
        self._step_next_pose(
            env_state_action_dict["rigid_prims"], env_state_action_dict["human"], self.default_z_human
        )
        self._step_next_pose(
            env_state_action_dict["rigid_prims"], env_state_action_dict["robot"], self.default_z_robot
        )
        self._step_gantry(env_state_action_dict)
        return env_state_action_dict

    def _step_gantry(self, env_state_action_dict: dict) -> None:
        """Resolve per-gantry target_area_xy from map point id."""
        gantry_state = env_state_action_dict["machine"]["num07_gantry_group"]
        active_indices = CfgMachine["num07_gantry_group"]["active_gantry_indices"]
        for gantry_index in active_indices:
            target_area_id = gantry_state["target_area_id"][gantry_index]
            if target_area_id is None or gantry_state["target_area_xy"][gantry_index] is not None:
                continue

            area_id = int(target_area_id)
            x, y = self.human_roadmap.get_xy_at_area_id(area_id)
            gantry_state["target_area_xy"][gantry_index] = torch.tensor(
                [x, y],
                dtype=torch.float32,
                device=self.cuda_device,
            )

    def reset(self, env_state_action_dict: dict) -> dict:
        pass

    def _step_next_pose(self, agent_prims_dict: dict, agent_states: dict, default_z: float) -> dict:
        """Write ``generated_route[route_index]`` to rigid_prims and increment route_index.

        Called after ``agents_collision_check`` adjusts routes.
        """
        for agent_name, agent_state in agent_states.items():
            route_index = agent_state["route_index"]
            route_length = agent_state["route_length"]
            if route_length == 0:
                continue
            if route_index >= route_length:
                #arrived at the target area
                assert agent_state["current_area_id"] is not None, "The current area id should be set before the agent arrives at the target area"
                agent_state["current_area_id"] = agent_state["target_area_id"]
                continue
            route = agent_state["generated_route"]
            waypoint = route[route_index]
            agent_prims_dict[agent_name]["position"] = torch.tensor(
                [waypoint["x"], waypoint["y"], default_z],
                dtype=torch.float32,
                device=self.cuda_device,
            ).unsqueeze(0)
            orientation = waypoint["orientation"]
            if orientation.dim() == 1:
                orientation = orientation.unsqueeze(0)
            agent_prims_dict[agent_name]["orientation"] = orientation
            agent_state["route_index"] += 1

    def _move_route_to_device(self, route: list[dict]) -> list[dict]:
        gpu_route: list[dict] = []
        for waypoint in route:
            orientation = waypoint["orientation"]
            if not isinstance(orientation, torch.Tensor):
                orientation = torch.tensor(orientation, dtype=torch.float32, device=_CPU_DEVICE)
            gpu_route.append(
                {
                    "x": waypoint["x"],
                    "y": waypoint["y"],
                    "orientation": orientation.to(self.cuda_device),
                }
            )
        return gpu_route

    def _generate_working_agent_routes(self, agent_states: dict, route_generator) -> None:
        """Generate initial working-agent routes from the map graph only (no occupancy check).

        When ``current_area_id != target_area_id`` and ``generated_route`` is empty, call
        ``route_generator`` to fill ``generated_route`` / ``route_index`` / ``route_length``.
        Detour and yield waypoints are not created in this stage.
        """
        for state in agent_states.values():
            if state["ongoing_task_record_index"] is None:
                continue
            if (
                state["current_area_id"] != state["target_area_id"]
                and state["target_area_id"] is not None
                and len(state["generated_route"]) == 0
            ):
                route_info = route_generator(state["current_area_id"], state["target_area_id"])
                state["generated_route"] = self._move_route_to_device(route_info["route"])
                state["route_index"] = 0
                state["route_length"] = len(route_info["route"])
                self._clear_detour_state(state)

    def generate_working_human_route(self, start_id: int, end_id: int) -> dict:
        """Build the initial working human route from the map graph (no live occupancy check)."""
        route_info = self.human_roadmap.generate_route(start_id, end_id)
        offset_orientation = self.human_route_orientation_offset
        updated_route: list[dict] = []
        for waypoint in route_info["route"]:
            # path heading first, then apply human model orientation offset in local frame
            composed_orientation = quat_multiply_wxyz(waypoint["orientation"], offset_orientation)
            updated_route.append(
                {
                    "x": waypoint["x"],
                    "y": waypoint["y"],
                    "orientation": composed_orientation,
                }
            )
        route_info["route"] = updated_route
        return route_info

    def generate_working_robot_route(self, start_id: int, end_id: int) -> dict:
        """Build the initial working robot route from the map graph (no live occupancy check)."""
        return self.robot_roadmap.generate_route(start_id, end_id)

    # ------------------------------------------------------------------
    # Collision avoidance pipeline (called before _step_next_pose)
    # ------------------------------------------------------------------

    def agents_collision_check(self, env_state_action_dict: dict) -> None:
        """Orchestrate dynamic collision avoidance before pose stepping; does not modify rigid_prims.

        Called from ``step`` after ``_generate_working_agent_routes`` and before
        ``_step_next_pose``. All detour/yield waypoints are written to ``generated_route`` only;
        ``map_routes_*.json`` is never modified.

        Two-phase resolution (working first, free last):
        1. Resolve detour/wait among working agents only (free agents are excluded).
        2. After working routes are updated, predictively yield free agents when they
           approach a working route (distance threshold), then advance immediately on
           yield-route creation.
        3. Finalize completed yield routes, then separate idle free agents whose
           passing ranges overlap (priority order, lateral offset, stay at new pose).
        """
        working_snapshot = self._collect_agent_collision_snapshot(
            env_state_action_dict, include_free=False
        )
        self._resolve_working_mover_conflicts(env_state_action_dict, working_snapshot)

        full_snapshot = self._collect_agent_collision_snapshot(
            env_state_action_dict,
            include_free=True,
            lookahead_override=int(self.collision_cfg["free_yield_lookahead_waypoints"]),
        )
        self._resolve_free_agent_yields(env_state_action_dict, full_snapshot)
        self._finalize_free_agent_yields(env_state_action_dict)

    @staticmethod
    def _is_free_agent(agent_state: dict) -> bool:
        return agent_state["state"] == "free"

    @staticmethod
    def _is_working_agent(agent_state: dict) -> bool:
        return agent_state["state"] != "free"

    def _lookahead_for_agent_type(self, agent_type: str) -> int:
        return int(self.collision_cfg["lookahead_waypoints"][agent_type])

    def _occupancy_for_agent_type(self, agent_type: str) -> rc.OccupancyMap:
        if agent_type == "human":
            return self._occupancy_human
        if agent_type == "robot":
            return self._occupancy_robot
        raise ValueError(f"Unknown agent type for occupancy map: {agent_type}")

    def _roadmap_for_agent_type(self, agent_type: str) -> _RoadmapGraph:
        if agent_type == "human":
            return self.human_roadmap
        if agent_type == "robot":
            return self.robot_roadmap
        raise ValueError(f"Unknown agent type for roadmap: {agent_type}")

    @staticmethod
    def _pose_from_rigid_prim(prim_entry: dict) -> dict:
        position = prim_entry["position"]
        orientation = prim_entry["orientation"]
        if position.dim() > 1:
            position = position.reshape(-1, 3)[0]
        if orientation.dim() > 1:
            orientation = orientation.reshape(-1, 4)[0]
        return {
            "x": float(position[0].item()),
            "y": float(position[1].item()),
            "orientation": orientation,
        }

    @staticmethod
    def _is_agent_moving(agent_state: dict) -> bool:
        return agent_state["route_length"] > 0 and agent_state["route_index"] < agent_state["route_length"]

    def _yaw_from_orientation(self, orientation: torch.Tensor) -> float:
        if orientation.dim() > 1:
            orientation = orientation.reshape(-1, 4)[0]
        return quaternion_wxyz_to_yaw(orientation.to(_CPU_DEVICE))

    def _make_waypoint(self, x: float, y: float, yaw: float) -> dict:
        return {
            "x": float(x),
            "y": float(y),
            "orientation": yaw_to_quaternion_wxyz(yaw, self.cuda_device),
        }

    def _human_footprint_at_xy(self, x: float, y: float) -> dict:
        """Human point footprint: circle/capsule with ``human_safety_diameter``, centered at (x, y)."""
        diameter = float(self.collision_cfg["human_safety_diameter"])
        return rc.circle_footprint(x, y, diameter)

    def _robot_footprint_at_pose(self, x: float, y: float, orientation: torch.Tensor) -> dict:
        """Robot point footprint: yaw-oriented rectangle.

        Local origin at one end of the vehicle; length and width from
        ``collision_cfg["robot_footprint_local_bounds"]``. Determined by (x, y) and orientation yaw.
        """
        yaw = self._yaw_from_orientation(orientation)
        return rc.rect_footprint(x, y, yaw, self.collision_cfg["robot_footprint_local_bounds"])

    def _robot_sweep_footprint(
        self,
        pose_from: dict,
        pose_to: dict,
    ) -> dict:
        """Robot motion occupancy: rectangle cluster covering the sweep from current to next pose."""
        step = float(self.collision_cfg["robot_sweep_step_m"])
        x0, y0 = pose_from["x"], pose_from["y"]
        x1, y1 = pose_to["x"], pose_to["y"]
        yaw = rc.yaw_from_xy_delta(x1 - x0, y1 - y0)
        primitives = [self._robot_footprint_at_pose(x0, y0, pose_from["orientation"])]
        for x, y in rc.densify_segment(x0, y0, x1, y1, step)[1:]:
            primitives.append(rc.rect_footprint(x, y, yaw, self.collision_cfg["robot_footprint_local_bounds"]))
        return {"primitives": primitives}

    def _footprint_for_pose(self, agent_type: str, pose: dict) -> dict:
        if agent_type == "human":
            return self._human_footprint_at_xy(pose["x"], pose["y"])
        return self._robot_footprint_at_pose(pose["x"], pose["y"], pose["orientation"])

    def _compute_passing_range(
        self,
        agent_type: str,
        agent_state: dict,
        current_pose: dict,
        *,
        lookahead_override: int | None = None,
    ) -> dict:
        """Compute passing range for the current step plus lookahead waypoints."""
        route = agent_state["generated_route"]
        route_index = int(agent_state["route_index"])
        route_length = int(agent_state["route_length"])
        if lookahead_override is not None:
            lookahead_count = lookahead_override
        else:
            lookahead_count = self._lookahead_for_agent_type(agent_type)
        primitives: list[dict] = []

        current_fp = self._footprint_for_pose(agent_type, current_pose)
        primitives.append(current_fp)

        for offset in range(lookahead_count):
            waypoint_index = route_index + offset
            if waypoint_index >= route_length:
                break
            waypoint = route[waypoint_index]
            waypoint_pose = {
                "x": float(waypoint["x"]),
                "y": float(waypoint["y"]),
                "orientation": waypoint["orientation"],
            }
            primitives.append(self._footprint_for_pose(agent_type, waypoint_pose))

        if (
            lookahead_override is None
            and agent_type == "robot"
            and route_index < route_length
        ):
            next_pose = {
                "x": float(route[route_index]["x"]),
                "y": float(route[route_index]["y"]),
                "orientation": route[route_index]["orientation"],
            }
            sweep = self._robot_sweep_footprint(current_pose, next_pose)
            primitives.extend(sweep["primitives"])

        return {"primitives": primitives}

    def _build_snapshot_entry(
        self,
        agent_key: str,
        agent_type: str,
        agent_state: dict,
        rigid_prims: dict,
        *,
        lookahead_override: int | None = None,
    ) -> dict:
        current_pose = self._pose_from_rigid_prim(rigid_prims[agent_key])
        passing_range = self._compute_passing_range(
            agent_type, agent_state, current_pose, lookahead_override=lookahead_override
        )
        return {
            "agent_key": agent_key,
            "agent_type": agent_type,
            "is_working": self._is_working_agent(agent_state),
            "is_moving": self._is_agent_moving(agent_state),
            "current_pose": current_pose,
            "passing_range": passing_range,
            "agent_state": agent_state,
        }

    def _collect_agent_collision_snapshot(
        self,
        env_state_action_dict: dict,
        *,
        include_free: bool = True,
        lookahead_override: int | None = None,
    ) -> dict:
        """Build a collision snapshot: passing_range per agent from rigid_prims and state."""
        entries: list[dict] = []
        rigid_prims = env_state_action_dict["rigid_prims"]
        for agent_type in ("human", "robot"):
            for agent_key, agent_state in env_state_action_dict[agent_type].items():
                if not include_free and self._is_free_agent(agent_state):
                    continue
                if agent_key not in rigid_prims:
                    continue
                entries.append(
                    self._build_snapshot_entry(
                        agent_key,
                        agent_type,
                        agent_state,
                        rigid_prims,
                        lookahead_override=lookahead_override,
                    )
                )
        return {
            "agents": entries,
            "by_key": {entry["agent_key"]: entry for entry in entries},
        }

    def _refresh_snapshot_entry(self, entry: dict, rigid_prims: dict) -> None:
        entry["current_pose"] = self._pose_from_rigid_prim(rigid_prims[entry["agent_key"]])
        entry["is_moving"] = self._is_agent_moving(entry["agent_state"])
        entry["passing_range"] = self._compute_passing_range(
            entry["agent_type"], entry["agent_state"], entry["current_pose"]
        )

    @staticmethod
    def _clear_detour_state(agent_state: dict) -> None:
        agent_state["detour_active"] = False
        agent_state["detour_blocker_key"] = None
        agent_state["detour_until_route_index"] = None

    def _activate_detour_state(
        self,
        agent_state: dict,
        blocker_key: str,
        until_route_index: int,
    ) -> None:
        agent_state["detour_active"] = True
        agent_state["detour_blocker_key"] = blocker_key
        agent_state["detour_until_route_index"] = int(until_route_index)

    def _should_skip_detour(self, agent_state: dict, blocker_key: str) -> bool:
        """Return True while the agent is still executing a detour for the same blocker."""
        if not agent_state.get("detour_active"):
            return False
        if agent_state.get("detour_blocker_key") != blocker_key:
            return False
        until_idx = agent_state.get("detour_until_route_index")
        if until_idx is None:
            return False
        return int(agent_state["route_index"]) < int(until_idx)

    def _sync_detour_state(self, mover_entry: dict) -> None:
        """Clear detour cooldown only after all modified conflicting waypoints are passed."""
        agent_state = mover_entry["agent_state"]
        if not agent_state.get("detour_active"):
            return

        until_idx = agent_state.get("detour_until_route_index")
        if until_idx is not None and int(agent_state["route_index"]) >= int(until_idx):
            self._clear_detour_state(agent_state)
            return

        if int(agent_state["route_length"]) == 0:
            self._clear_detour_state(agent_state)

    def _find_conflicting_waypoint_offsets(
        self,
        agent_type: str,
        route: list[dict],
        route_index: int,
        blocker_passing_range: dict,
    ) -> list[int]:
        """Return offsets (relative to route_index) whose footprints overlap the blocker."""
        scan_limit = min(
            int(self.collision_cfg["detour_conflict_scan_waypoints"]),
            len(route) - route_index,
        )
        conflicting_offsets: list[int] = []
        for offset in range(scan_limit):
            waypoint = route[route_index + offset]
            pose = {
                "x": float(waypoint["x"]),
                "y": float(waypoint["y"]),
                "orientation": waypoint["orientation"],
            }
            footprint = self._footprint_for_pose(agent_type, pose)
            if self._check_passing_range_overlap(
                {"primitives": [footprint]}, blocker_passing_range
            ):
                conflicting_offsets.append(offset)
        return conflicting_offsets

    def _build_detour_modification_weights(
        self,
        route: list[dict],
        route_index: int,
        conflicting_offsets: list[int],
    ) -> dict[int, float]:
        """Map global route indices to lateral offset weights for conflict smoothing."""
        smooth_radius = int(self.collision_cfg["detour_smooth_neighbor_waypoints"])
        route_len = len(route)
        weights: dict[int, float] = {}
        for offset in conflicting_offsets:
            conflict_global = route_index + offset
            for delta in range(-smooth_radius, smooth_radius + 1):
                global_idx = conflict_global + delta
                if global_idx <= 0 or global_idx >= route_len - 1:
                    continue
                weight = max(0.0, 1.0 - abs(delta) / (smooth_radius + 1))
                weights[global_idx] = max(weights.get(global_idx, 0.0), weight)
        return weights

    def _nearest_conflict_global_idx(
        self,
        global_idx: int,
        route_index: int,
        conflicting_offsets: list[int],
    ) -> int:
        conflict_globals = [route_index + offset for offset in conflicting_offsets]
        return min(conflict_globals, key=lambda conflict_idx: abs(conflict_idx - global_idx))

    def _path_tangent_at(
        self,
        route: list[dict],
        global_idx: int,
        route_index: int,
        current_pose: dict,
    ) -> tuple[float, float]:
        """Unit tangent of the original route polyline at global_idx."""
        if route_index < global_idx < len(route) - 1:
            dx = float(route[global_idx + 1]["x"]) - float(route[global_idx - 1]["x"])
            dy = float(route[global_idx + 1]["y"]) - float(route[global_idx - 1]["y"])
        elif global_idx + 1 < len(route):
            dx = float(route[global_idx + 1]["x"]) - float(route[global_idx]["x"])
            dy = float(route[global_idx + 1]["y"]) - float(route[global_idx]["y"])
        elif global_idx > 0:
            dx = float(route[global_idx]["x"]) - float(route[global_idx - 1]["x"])
            dy = float(route[global_idx]["y"]) - float(route[global_idx - 1]["y"])
        else:
            dx = float(route[global_idx]["x"]) - float(current_pose["x"])
            dy = float(route[global_idx]["y"]) - float(current_pose["y"])

        length = math.hypot(dx, dy)
        if length < 1e-9:
            return 1.0, 0.0
        return dx / length, dy / length

    @staticmethod
    def _perpendicular_away_from_blocker(
        tangent: tuple[float, float],
        waypoint_xy: tuple[float, float],
        blocker_xy: tuple[float, float],
    ) -> tuple[float, float]:
        """Unit perpendicular pointing away from the blocker."""
        tx, ty = tangent
        perp_x, perp_y = -ty, tx
        to_blocker_x = blocker_xy[0] - waypoint_xy[0]
        to_blocker_y = blocker_xy[1] - waypoint_xy[1]
        if perp_x * to_blocker_x + perp_y * to_blocker_y > 0.0:
            perp_x, perp_y = -perp_x, -perp_y
        return perp_x, perp_y

    def _yaw_along_route_at(
        self,
        route: list[dict],
        global_idx: int,
        current_pose: dict,
    ) -> float:
        """Yaw aligned with the route polyline at global_idx (after xy updates)."""
        if global_idx + 1 < len(route):
            dx = float(route[global_idx + 1]["x"]) - float(route[global_idx]["x"])
            dy = float(route[global_idx + 1]["y"]) - float(route[global_idx]["y"])
        elif global_idx > 0:
            dx = float(route[global_idx]["x"]) - float(route[global_idx - 1]["x"])
            dy = float(route[global_idx]["y"]) - float(route[global_idx - 1]["y"])
        else:
            dx = float(route[global_idx]["x"]) - float(current_pose["x"])
            dy = float(route[global_idx]["y"]) - float(current_pose["y"])
        return rc.yaw_from_xy_delta(dx, dy)

    def _set_waypoint_orientation(self, waypoint: dict, yaw: float, agent_type: str) -> None:
        orientation = yaw_to_quaternion_wxyz(yaw, self.cuda_device)
        if agent_type == "human":
            orientation = quat_multiply_wxyz(orientation, self.human_route_orientation_offset.to(self.cuda_device))
        waypoint["orientation"] = orientation

    def _resolve_working_mover_conflicts(self, env_state_action_dict: dict, snapshot: dict) -> None:
        """Phase 1: resolve passing_range conflicts among working agents (free excluded).

        Stationary working agents never move. Moving working agents detour around stationary
        blockers, or wait when blocked by a higher-priority moving working agent.
        """
        rigid_prims = env_state_action_dict["rigid_prims"]
        working_agents = [entry for entry in snapshot["agents"] if entry["is_working"]]
        moving_agents = [entry for entry in working_agents if entry["is_moving"]]

        for mover in moving_agents:
            self._sync_detour_state(mover)
            for other in working_agents:
                if mover["agent_key"] == other["agent_key"]:
                    continue
                if not self._check_passing_range_overlap(mover["passing_range"], other["passing_range"]):
                    continue
                if not other["is_moving"]:
                    if self._should_skip_detour(mover["agent_state"], other["agent_key"]):
                        break
                    if self._apply_detour(
                        mover["agent_state"],
                        mover["agent_type"],
                        {"blocker": other, "current_pose": mover["current_pose"]},
                    ):
                        self._refresh_snapshot_entry(mover, rigid_prims)
                    break
                if rc.should_agent_wait(mover, other):
                    self._clear_detour_state(mover["agent_state"])
                    self._apply_wait(mover["agent_state"], mover["current_pose"])
                    self._refresh_snapshot_entry(mover, rigid_prims)
                    break

    def _apply_detour(
        self,
        agent_state: dict,
        agent_type: str,
        block_info: dict,
    ) -> bool:
        """Detour: offset conflicting waypoints perpendicular to the route; no new waypoints."""
        blocker_key = block_info["blocker"]["agent_key"]
        if self._should_skip_detour(agent_state, blocker_key):
            return False

        current_pose = block_info["current_pose"]
        blocker_passing_range = block_info["blocker"]["passing_range"]
        blocker_xy = (
            float(block_info["blocker"]["current_pose"]["x"]),
            float(block_info["blocker"]["current_pose"]["y"]),
        )
        route = agent_state["generated_route"]
        route_index = int(agent_state["route_index"])
        if route_index >= len(route):
            return False

        conflicting_offsets = self._find_conflicting_waypoint_offsets(
            agent_type, route, route_index, blocker_passing_range
        )
        if not conflicting_offsets:
            return False

        modification_weights = self._build_detour_modification_weights(
            route, route_index, conflicting_offsets
        )
        if not modification_weights:
            return False

        lateral_base = float(self.collision_cfg["detour_lateral_offset_m"])
        max_attempts = int(self.collision_cfg["detour_max_attempts"])

        for attempt in range(max_attempts):
            sign = 1.0 if attempt % 2 == 0 else -1.0
            magnitude = lateral_base * (attempt // 2 + 1)
            trial_positions: dict[int, tuple[float, float]] = {}
            trial_footprints: list[dict] = []

            for global_idx, weight in modification_weights.items():
                anchor_global_idx = self._nearest_conflict_global_idx(
                    global_idx, route_index, conflicting_offsets
                )
                anchor_waypoint = route[anchor_global_idx]
                anchor_xy = (float(anchor_waypoint["x"]), float(anchor_waypoint["y"]))
                waypoint = route[global_idx]
                waypoint_xy = (float(waypoint["x"]), float(waypoint["y"]))
                tangent = self._path_tangent_at(route, anchor_global_idx, route_index, current_pose)
                perp_x, perp_y = self._perpendicular_away_from_blocker(
                    tangent, anchor_xy, blocker_xy
                )
                effective_magnitude = magnitude * weight
                new_x = waypoint_xy[0] + perp_x * sign * effective_magnitude
                new_y = waypoint_xy[1] + perp_y * sign * effective_magnitude
                trial_positions[global_idx] = (new_x, new_y)
                trial_pose = {
                    "x": new_x,
                    "y": new_y,
                    "orientation": waypoint["orientation"],
                }
                trial_footprints.append(self._footprint_for_pose(agent_type, trial_pose))

            if self._check_static_map_collision({"primitives": trial_footprints}, agent_type):
                continue

            for global_idx, (new_x, new_y) in trial_positions.items():
                route[global_idx]["x"] = new_x
                route[global_idx]["y"] = new_y
                yaw = self._yaw_along_route_at(route, global_idx, current_pose)
                self._set_waypoint_orientation(route[global_idx], yaw, agent_type)

            until_route_index = max(modification_weights.keys()) + 1
            self._activate_detour_state(agent_state, blocker_key, until_route_index)
            return True

        return False

    def _apply_wait(self, agent_state: dict, current_pose: dict) -> None:
        """Wait: insert a hold waypoint at the current (x, y, yaw)."""
        route = agent_state["generated_route"]
        route_index = int(agent_state["route_index"])
        yaw = self._yaw_from_orientation(current_pose["orientation"])
        hold_waypoint = self._make_waypoint(current_pose["x"], current_pose["y"], yaw)
        route.insert(route_index, hold_waypoint)
        agent_state["route_length"] = len(route)

    def _check_static_map_collision(self, passing_range: dict, agent_type: str) -> bool:
        """Return whether passing_range collides with static occupancy-map obstacles."""
        sample_step = float(self.collision_cfg["detour_densify_step_m"])
        return self._occupancy_for_agent_type(agent_type).passing_range_collides(passing_range, sample_step)

    def _check_passing_range_overlap(self, range_a: dict, range_b: dict) -> bool:
        """Return whether two passing ranges overlap in space."""
        return rc.passing_ranges_overlap(range_a, range_b)

    def _footprint_with_yield_clearance(self, agent_type: str, pose: dict) -> dict:
        """Footprint enlarged by yield clearance margin for safer stand-off distance."""
        margin = float(self.collision_cfg["yield_clearance_margin_m"])
        if agent_type == "human":
            diameter = float(self.collision_cfg["human_safety_diameter"]) + 2.0 * margin
            return rc.circle_footprint(pose["x"], pose["y"], diameter)
        yaw = self._yaw_from_orientation(pose["orientation"])
        footprint = rc.rect_footprint(
            pose["x"], pose["y"], yaw, self.collision_cfg["robot_footprint_local_bounds"]
        )
        return footprint

    @staticmethod
    def _clear_yield_state(agent_state: dict) -> None:
        agent_state["yield_active"] = False
        agent_state["yield_blocker_key"] = None

    def _activate_yield_state(self, agent_state: dict, blocker_key: str) -> None:
        agent_state["yield_active"] = True
        agent_state["yield_blocker_key"] = blocker_key

    def _is_executing_yield_route(self, agent_state: dict) -> bool:
        """Return True while a yield route is still being followed."""
        if not agent_state.get("yield_active"):
            return False
        return (
            int(agent_state["route_length"]) > 0
            and int(agent_state["route_index"]) < int(agent_state["route_length"])
        )

    def _predictive_yield_conflicts(
        self,
        from_xy: tuple[float, float],
        working_agents: list[dict],
    ) -> list[dict]:
        """Return working agents whose route is within the predictive trigger distance."""
        trigger_distance = float(self.collision_cfg["yield_predictive_trigger_distance_m"])
        return [
            other
            for other in working_agents
            if self._distance_xy_to_working_route(from_xy, other) <= trigger_distance
        ]

    def _find_yield_route_start_index(self, route: list[dict], current_pose: dict) -> int:
        """Skip densified prefix waypoints that duplicate the current pose."""
        threshold = float(self.collision_cfg["yield_route_step_m"]) * 0.5
        current_x = float(current_pose["x"])
        current_y = float(current_pose["y"])
        for idx, waypoint in enumerate(route):
            dist = math.hypot(float(waypoint["x"]) - current_x, float(waypoint["y"]) - current_y)
            if dist > threshold:
                return idx
        return len(route)

    def _write_agent_waypoint_to_prim(
        self,
        agent_key: str,
        waypoint: dict,
        rigid_prims: dict,
        default_z: float,
    ) -> None:
        rigid_prims[agent_key]["position"] = torch.tensor(
            [waypoint["x"], waypoint["y"], default_z],
            dtype=torch.float32,
            device=self.cuda_device,
        ).unsqueeze(0)
        orientation = waypoint["orientation"]
        if orientation.dim() == 1:
            orientation = orientation.unsqueeze(0)
        rigid_prims[agent_key]["orientation"] = orientation

    def _apply_yield_route_and_advance(
        self,
        entry: dict,
        yield_route: list[dict],
        goal_xy: tuple[float, float],
        blocker_key: str,
        env_state_action_dict: dict,
    ) -> None:
        """Assign a yield route, lock yield state, and advance one actionable waypoint immediately."""
        agent_state = entry["agent_state"]
        agent_type = entry["agent_type"]
        current_pose = entry["current_pose"]
        default_z = self.default_z_human if agent_type == "human" else self.default_z_robot

        gpu_route = self._move_route_to_device(yield_route)
        start_index = self._find_yield_route_start_index(gpu_route, current_pose)
        agent_state["generated_route"] = gpu_route
        agent_state["route_length"] = len(gpu_route)
        agent_state["route_index"] = start_index
        roadmap = self._roadmap_for_agent_type(agent_type)
        agent_state["target_area_id"] = roadmap.find_nearest_area_id(goal_xy[0], goal_xy[1])
        self._activate_yield_state(agent_state, blocker_key)

        if start_index < len(gpu_route):
            self._write_agent_waypoint_to_prim(
                entry["agent_key"],
                gpu_route[start_index],
                env_state_action_dict["rigid_prims"],
                default_z,
            )
            agent_state["route_index"] = start_index + 1

    def _resolve_free_agent_yields(self, env_state_action_dict: dict, snapshot: dict) -> None:
        """Phase 2: predictively yield free agents and advance immediately on route creation."""
        working_agents = [entry for entry in snapshot["agents"] if entry["is_working"]]
        for entry in snapshot["agents"]:
            if entry["is_working"]:
                continue

            agent_state = entry["agent_state"]
            if self._is_executing_yield_route(agent_state):
                continue

            current_pose = entry["current_pose"]
            from_xy = (current_pose["x"], current_pose["y"])
            conflicts = self._predictive_yield_conflicts(from_xy, working_agents)
            if not conflicts:
                continue

            agent_type = entry["agent_type"]
            primary_conflict = self._select_primary_yield_conflict(from_xy, conflicts)
            goal_xy = self._sample_yield_standable_xy(
                agent_type, from_xy, working_agents, primary_conflict
            )
            goal_yaw = rc.yaw_from_xy_delta(goal_xy[0] - from_xy[0], goal_xy[1] - from_xy[1])
            goal_pose = {
                "x": goal_xy[0],
                "y": goal_xy[1],
                "orientation": yaw_to_quaternion_wxyz(goal_yaw, self.cuda_device),
            }
            yield_route = self._build_yield_route(current_pose, goal_pose, agent_type)
            self._apply_yield_route_and_advance(
                entry,
                yield_route,
                goal_xy,
                primary_conflict["agent_key"],
                env_state_action_dict,
            )

    def _distance_xy_to_working_route(self, query_xy: tuple[float, float], working_entry: dict) -> float:
        agent_state = working_entry["agent_state"]
        route = agent_state["generated_route"]
        route_index = int(agent_state["route_index"])
        route_length = int(agent_state["route_length"])
        if route_length == 0 or route_index >= route_length:
            pose = working_entry["current_pose"]
            return math.hypot(query_xy[0] - pose["x"], query_xy[1] - pose["y"])

        lookahead = int(self.collision_cfg["free_yield_lookahead_waypoints"])
        best_dist_sq = math.inf
        for idx in range(route_index, min(route_length, route_index + lookahead)):
            wx = float(route[idx]["x"])
            wy = float(route[idx]["y"])
            dist_sq = (wx - query_xy[0]) ** 2 + (wy - query_xy[1]) ** 2
            best_dist_sq = min(best_dist_sq, dist_sq)
        return math.sqrt(best_dist_sq)

    def _select_primary_yield_conflict(
        self,
        from_xy: tuple[float, float],
        conflicts: list[dict],
    ) -> dict:
        """Pick the working agent whose route is closest to the free agent."""
        return min(conflicts, key=lambda entry: self._distance_xy_to_working_route(from_xy, entry))

    def _working_route_tangent_near_xy(
        self,
        working_entry: dict,
        query_xy: tuple[float, float],
    ) -> tuple[float, float]:
        """Unit tangent of the working route polyline near query_xy."""
        agent_state = working_entry["agent_state"]
        route = agent_state["generated_route"]
        route_index = int(agent_state["route_index"])
        route_length = int(agent_state["route_length"])
        current_pose = working_entry["current_pose"]

        if route_length == 0 or route_index >= route_length:
            yaw = self._yaw_from_orientation(current_pose["orientation"])
            return math.cos(yaw), math.sin(yaw)

        lookahead = int(self.collision_cfg["free_yield_lookahead_waypoints"])
        nearest_idx = route_index
        best_dist_sq = math.inf
        for idx in range(route_index, min(route_length, route_index + lookahead)):
            wx = float(route[idx]["x"])
            wy = float(route[idx]["y"])
            dist_sq = (wx - query_xy[0]) ** 2 + (wy - query_xy[1]) ** 2
            if dist_sq < best_dist_sq:
                best_dist_sq = dist_sq
                nearest_idx = idx

        return self._path_tangent_at(route, nearest_idx, route_index, current_pose)

    @staticmethod
    def _perpendicular_from_route_tangent(tangent: tuple[float, float]) -> tuple[float, float]:
        tx, ty = tangent
        return -ty, tx

    def _sample_yield_standable_xy(
        self,
        agent_type: str,
        from_xy: tuple[float, float],
        working_agents: list[dict],
        conflict_working: dict,
    ) -> tuple[float, float]:
        """Sample a standable point by stepping perpendicular to the conflicting working route."""
        search_radius = float(self.collision_cfg["yield_search_radius_m"])
        search_step = float(self.collision_cfg["yield_search_step_m"])
        occupancy = self._occupancy_for_agent_type(agent_type)

        tangent = self._working_route_tangent_near_xy(conflict_working, from_xy)
        perp_x, perp_y = self._perpendicular_from_route_tangent(tangent)

        nearest_route_xy = self._nearest_working_route_xy(conflict_working, from_xy)
        away_x = from_xy[0] - nearest_route_xy[0]
        away_y = from_xy[1] - nearest_route_xy[1]
        if perp_x * away_x + perp_y * away_y < 0.0:
            perp_x, perp_y = -perp_x, -perp_y

        step_count = max(1, int(math.ceil(search_radius / search_step)))
        for step_idx in range(1, step_count + 1):
            distance = step_idx * search_step
            for sign in (1.0, -1.0):
                candidate = (
                    from_xy[0] + perp_x * sign * distance,
                    from_xy[1] + perp_y * sign * distance,
                )
                if self._is_standable_xy(agent_type, candidate, working_agents, occupancy):
                    return candidate

        if self._is_standable_xy(agent_type, from_xy, working_agents, occupancy):
            return from_xy
        return from_xy

    def _nearest_working_route_xy(
        self,
        working_entry: dict,
        query_xy: tuple[float, float],
    ) -> tuple[float, float]:
        agent_state = working_entry["agent_state"]
        route = agent_state["generated_route"]
        route_index = int(agent_state["route_index"])
        route_length = int(agent_state["route_length"])
        if route_length == 0 or route_index >= route_length:
            pose = working_entry["current_pose"]
            return pose["x"], pose["y"]

        lookahead = int(self.collision_cfg["free_yield_lookahead_waypoints"])
        nearest_xy = (float(route[route_index]["x"]), float(route[route_index]["y"]))
        best_dist_sq = math.inf
        for idx in range(route_index, min(route_length, route_index + lookahead)):
            wx = float(route[idx]["x"])
            wy = float(route[idx]["y"])
            dist_sq = (wx - query_xy[0]) ** 2 + (wy - query_xy[1]) ** 2
            if dist_sq < best_dist_sq:
                best_dist_sq = dist_sq
                nearest_xy = (wx, wy)
        return nearest_xy

    def _is_standable_xy(
        self,
        agent_type: str,
        xy: tuple[float, float],
        working_agents: list[dict],
        occupancy: rc.OccupancyMap,
        *,
        free_agents: list[dict] | None = None,
    ) -> bool:
        pose = {
            "x": xy[0],
            "y": xy[1],
            "orientation": yaw_to_quaternion_wxyz(0.0, self.cuda_device),
        }
        footprint = self._footprint_with_yield_clearance(agent_type, pose)
        if occupancy.passing_range_collides(
            {"primitives": [footprint]}, float(self.collision_cfg["yield_search_step_m"])
        ):
            return False
        for other in working_agents:
            if self._check_passing_range_overlap({"primitives": [footprint]}, other["passing_range"]):
                return False
        if free_agents is not None:
            for other in free_agents:
                if self._check_passing_range_overlap({"primitives": [footprint]}, other["passing_range"]):
                    return False
        return True

    def _build_yield_route(
        self,
        start_pose: dict,
        goal_pose: dict,
        agent_type: str,
    ) -> list[dict]:
        """Build a smooth yield route with limited per-step yaw change."""
        step = float(self.collision_cfg["yield_route_step_m"])
        max_yaw_step = float(self.collision_cfg["yield_max_yaw_step_rad"])
        start_yaw = self._yaw_from_orientation(start_pose["orientation"])
        goal_yaw = self._yaw_from_orientation(goal_pose["orientation"])

        path_points = rc.densify_segment(
            start_pose["x"], start_pose["y"], goal_pose["x"], goal_pose["y"], step
        )
        if not path_points:
            path_points = [(start_pose["x"], start_pose["y"]), (goal_pose["x"], goal_pose["y"])]

        route: list[dict] = []
        previous_yaw = start_yaw
        for point_idx, (x, y) in enumerate(path_points):
            if point_idx + 1 < len(path_points):
                next_x, next_y = path_points[point_idx + 1]
                target_yaw = rc.yaw_from_xy_delta(next_x - x, next_y - y)
            else:
                target_yaw = goal_yaw
            yaw = rc.clamp_yaw_step(previous_yaw, target_yaw, max_yaw_step)
            previous_yaw = yaw
            route.append(
                {
                    "x": float(x),
                    "y": float(y),
                    "orientation": yaw_to_quaternion_wxyz(yaw, _CPU_DEVICE),
                }
            )

        if agent_type == "human":
            for waypoint in route:
                waypoint["orientation"] = quat_multiply_wxyz(
                    waypoint["orientation"].to(_CPU_DEVICE),
                    self.human_route_orientation_offset,
                )
        else:
            route[-1]["orientation"] = yaw_to_quaternion_wxyz(goal_yaw, _CPU_DEVICE)

        return route

    def _collect_idle_free_agent_entries(self, env_state_action_dict: dict) -> list[dict]:
        """Build snapshot entries for idle free agents (current pose passing range only)."""
        entries: list[dict] = []
        rigid_prims = env_state_action_dict["rigid_prims"]
        for agent_type in ("human", "robot"):
            for agent_key, agent_state in env_state_action_dict[agent_type].items():
                if not self._is_free_agent(agent_state):
                    continue
                if self._is_executing_yield_route(agent_state):
                    continue
                if int(agent_state["route_length"]) > 0:
                    continue
                if agent_key not in rigid_prims:
                    continue
                entries.append(
                    self._build_snapshot_entry(
                        agent_key,
                        agent_type,
                        agent_state,
                        rigid_prims,
                        lookahead_override=0,
                    )
                )
        return entries

    def _free_agent_tangent_near_xy(
        self,
        entry: dict,
        query_xy: tuple[float, float],
    ) -> tuple[float, float]:
        """Unit tangent from the free agent's own route direction or current orientation."""
        agent_state = entry["agent_state"]
        route = agent_state["generated_route"]
        route_index = int(agent_state["route_index"])
        route_length = int(agent_state["route_length"])
        current_pose = entry["current_pose"]

        if route_length > 0 and route_index < route_length:
            nearest_idx = route_index
            best_dist_sq = math.inf
            for idx in range(route_index, route_length):
                wx = float(route[idx]["x"])
                wy = float(route[idx]["y"])
                dist_sq = (wx - query_xy[0]) ** 2 + (wy - query_xy[1]) ** 2
                if dist_sq < best_dist_sq:
                    best_dist_sq = dist_sq
                    nearest_idx = idx
            return self._path_tangent_at(route, nearest_idx, route_index, current_pose)

        yaw = self._yaw_from_orientation(current_pose["orientation"])
        return math.cos(yaw), math.sin(yaw)

    def _distance_xy_to_free_agent(self, query_xy: tuple[float, float], free_entry: dict) -> float:
        pose = free_entry["current_pose"]
        return math.hypot(query_xy[0] - pose["x"], query_xy[1] - pose["y"])

    def _sample_free_separation_standable_xy(
        self,
        agent_type: str,
        from_xy: tuple[float, float],
        entry: dict,
        working_agents: list[dict],
        committed_free_agents: list[dict],
        conflict_free: dict,
    ) -> tuple[float, float]:
        """Sample a standable point perpendicular to the free agent's own heading."""
        search_radius = float(self.collision_cfg["yield_search_radius_m"])
        search_step = float(self.collision_cfg["yield_search_step_m"])
        occupancy = self._occupancy_for_agent_type(agent_type)

        tangent = self._free_agent_tangent_near_xy(entry, from_xy)
        perp_x, perp_y = self._perpendicular_from_route_tangent(tangent)

        conflict_pose = conflict_free["current_pose"]
        away_x = from_xy[0] - conflict_pose["x"]
        away_y = from_xy[1] - conflict_pose["y"]
        if perp_x * away_x + perp_y * away_y < 0.0:
            perp_x, perp_y = -perp_x, -perp_y

        step_count = max(1, int(math.ceil(search_radius / search_step)))
        for step_idx in range(1, step_count + 1):
            distance = step_idx * search_step
            for sign in (1.0, -1.0):
                candidate = (
                    from_xy[0] + perp_x * sign * distance,
                    from_xy[1] + perp_y * sign * distance,
                )
                if self._is_standable_xy(
                    agent_type,
                    candidate,
                    working_agents,
                    occupancy,
                    free_agents=committed_free_agents,
                ):
                    return candidate

        if self._is_standable_xy(
            agent_type,
            from_xy,
            working_agents,
            occupancy,
            free_agents=committed_free_agents,
        ):
            return from_xy
        return from_xy

    def _resolve_free_free_separations(
        self,
        env_state_action_dict: dict,
        working_agents: list[dict],
    ) -> None:
        """Separate overlapping idle free agents; lower-priority agents move aside and stay."""
        free_entries = self._collect_idle_free_agent_entries(env_state_action_dict)
        free_entries.sort(
            key=lambda entry: rc.agent_priority_key(entry["agent_type"], entry["agent_key"])
        )

        committed_entries: list[dict] = []
        for entry in free_entries:
            conflicts = [
                other
                for other in committed_entries
                if self._check_passing_range_overlap(entry["passing_range"], other["passing_range"])
            ]
            if not conflicts:
                committed_entries.append(entry)
                continue

            agent_type = entry["agent_type"]
            current_pose = entry["current_pose"]
            from_xy = (current_pose["x"], current_pose["y"])
            primary_conflict = min(
                conflicts,
                key=lambda other: self._distance_xy_to_free_agent(from_xy, other),
            )
            goal_xy = self._sample_free_separation_standable_xy(
                agent_type,
                from_xy,
                entry,
                working_agents,
                committed_entries,
                primary_conflict,
            )
            if goal_xy == from_xy:
                committed_entries.append(entry)
                continue

            goal_yaw = rc.yaw_from_xy_delta(goal_xy[0] - from_xy[0], goal_xy[1] - from_xy[1])
            goal_pose = {
                "x": goal_xy[0],
                "y": goal_xy[1],
                "orientation": yaw_to_quaternion_wxyz(goal_yaw, self.cuda_device),
            }
            separation_route = self._build_yield_route(current_pose, goal_pose, agent_type)
            self._apply_yield_route_and_advance(
                entry,
                separation_route,
                goal_xy,
                primary_conflict["agent_key"],
                env_state_action_dict,
            )

    def _finalize_free_agent_yields(self, env_state_action_dict: dict) -> None:
        """Finalize completed yield/separation routes, then resolve free-free overlaps."""
        for agent_type in ("human", "robot"):
            for agent_state in env_state_action_dict[agent_type].values():
                if not self._is_free_agent(agent_state):
                    continue
                if agent_state["route_length"] == 0:
                    continue
                if agent_state["route_index"] < agent_state["route_length"]:
                    continue
                agent_state["current_area_id"] = agent_state["target_area_id"]
                agent_state["generated_route"] = []
                agent_state["route_index"] = 0
                agent_state["route_length"] = 0
                agent_state["target_area_id"] = None
                self._clear_yield_state(agent_state)

        working_snapshot = self._collect_agent_collision_snapshot(
            env_state_action_dict,
            include_free=False,
            lookahead_override=int(self.collision_cfg["free_yield_lookahead_waypoints"]),
        )
        working_agents = [entry for entry in working_snapshot["agents"] if entry["is_working"]]
        self._resolve_free_free_separations(env_state_action_dict, working_agents)

