import json
import os
import torch

CfgRoute = {
    "map_path_human": "~/work/isaac_factory/source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/env_asset_cfg/route/occupancy_map4human.png",
    "points_path_human": "~/work/isaac_factory/source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/env_asset_cfg/route/map_points_human.json",
    "routes_path_human": "~/work/Dataset/HC_data/map_data/map_routes_human.json",
    "map_path_robot": "~/work/isaac_factory/source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/env_asset_cfg/route/occupancy_map4robot.png",
    "points_path_robot": "~/work/isaac_factory/source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/env_asset_cfg/route/map_points_robot.json",
    "routes_path_robot": "~/work/Dataset/HC_data/map_data/map_routes_robot.json",   
    "png_image_coordinates": {
        "top_left": [0, 0],
        "bottom_right": [2202, 1645],
        "x_bound": [0, 2202],
        "y_bound": [0, 1645]
    },
    "isaac_sim_coordinates": {
        "top_left": [55.02995, -6.10027],
        "bottom_right": [-55.0706, 76.16652],
        "x_bound": [55.02995, -55.0706],
        "y_bound": [76.16652, -6.10027]
    },
    "collision_avoidance": {
        # Phase 1 working conflict: waypoints ahead included in passing_range
        "lookahead_waypoints": {
            "human": 3,
            "robot": 3,
        },
        # Human collision footprint diameter (m); circle centered at current pose
        "human_safety_diameter": 0.3,
        # Robot footprint local bounds (m); origin at one end, rotated by yaw
        "robot_footprint_local_bounds": {
            "min_x": 0.0,
            "max_x": 1.8,
            "min_y": 0.0,
            "max_y": 0.8,
        },
        # Occupancy map: pixel grayscale >= threshold is walkable (yield/detour static checks)
        "occupancy_free_threshold": 250,
        # Predictive yield trigger: free agent within this distance of a working route (m)
        "yield_predictive_trigger_distance_m": 2.0,
        # Max search radius along perpendicular for yield/separation stand points (m)
        "yield_search_radius_m": 3.5,
        # Yield/separation sample step (m); also used for occupancy collision sampling
        "yield_search_step_m": 0.2,
        # Yield/separation route densify step (m); spacing between waypoints
        "yield_route_step_m": 0.2,
        # Extra clearance margin on yield stand footprint (m)
        "yield_clearance_margin_m": 0.4,
        # Max yaw change between adjacent yield waypoints (rad)
        "yield_max_yaw_step_rad": 0.35,
        # Phase 2: waypoints ahead for working passing_range in free yield / free-free separation
        "free_yield_lookahead_waypoints": 15,
        # Phase 1 detour: lateral offset along route normal at conflicting waypoints (m)
        "detour_lateral_offset_m": 0.5,
        # Sample step for static occupancy collision checks (m)
        "detour_densify_step_m": 0.3,
        # Detour lateral attempts (alternating +/- normal direction)
        "detour_max_attempts": 6,
        # Waypoints ahead to scan for detour conflicts with a blocker
        "detour_conflict_scan_waypoints": 8,
        # Neighbor waypoints on each side of conflict for smooth detour weights (linear decay)
        "detour_smooth_neighbor_waypoints": 3,
        # Robot sweep sample step from current pose to next waypoint in passing_range (m)
        "robot_sweep_step_m": 0.3,
    },
}

# Optional init point ids from map points
OptionalInitPointIds = {
    "human": [190, 191, 192, 193, 194, 195, 202, 203, 204, 205, 206, 207, 214, 215, 216, 217, 218, 219],
    "robot": [236, 272],
    "human_z": 0.13395,
    "robot_z": 0.13395,
}


def _expand_user_path(path: str) -> str:
    return os.path.expanduser(path)


def _load_points(points_path: str) -> list[dict]:
    with open(_expand_user_path(points_path), "r", encoding="utf-8") as f:
        return json.load(f)


def _extract_xy_points_by_ids(points: list[dict], ids: list[int]) -> list[list[float]]:
    """Extract [[x, y], ...] for given point ids from loaded map points."""
    id_set = set(ids)
    xy: list[list[float]] = []
    for p in points:
        if int(p.get("id")) not in id_set:
            continue
        xy.append([float(p["x"]), float(p["y"])])
    return xy


def _build_agent_init_xyz(points_path: str, point_ids: list[int], z: float) -> torch.Tensor:
    xy = _extract_xy_points_by_ids(_load_points(points_path)["points"], point_ids)
    return torch.tensor([[p[0], p[1], z] for p in xy], dtype=torch.float32)


RouteOptionalInitPointsInMap: dict = {
    "human_xyz": _build_agent_init_xyz(
        CfgRoute["points_path_human"], OptionalInitPointIds["human"], OptionalInitPointIds["human_z"]
    ),
    "robot_xyz": _build_agent_init_xyz(
        CfgRoute["points_path_robot"], OptionalInitPointIds["robot"], OptionalInitPointIds["robot_z"]
    ),
}
