import json
import os
import torch

CfgRoute = {
    "map_path_human": "~/work/isaac_factory/source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/env_asset_cfg/cfg_route/occupancy_map4human.png",
    "points_path_human": "~/work/isaac_factory/source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/env_asset_cfg/cfg_route/map_points_human.json",
    "routes_path_human": "~/work/Dataset/HC_data/map_data/map_routes_human.json",
    "map_path_robot": "~/work/isaac_factory/source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/env_asset_cfg/cfg_route/occupancy_map4robot.png",
    "points_path_robot": "~/work/isaac_factory/source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/env_asset_cfg/cfg_route/map_points_robot.json",
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
        "lookahead_waypoints": {
            "human": 3,
            "robot": 3,
        },
        "human_safety_diameter": 0.1,
        "robot_footprint_local_bounds": {
            "min_x": 0.0,
            "max_x": 1.8,
            "min_y": 0.0,
            "max_y": 0.8,
        },
        "occupancy_free_threshold": 250,
        "yield_search_radius_m": 3.5,
        "yield_search_step_m": 0.2,
        "yield_route_step_m": 0.2,
        "yield_clearance_margin_m": 0.4,
        "yield_max_yaw_step_rad": 0.35,
        "free_yield_lookahead_waypoints": 15,
        "detour_lateral_offset_m": 0.5,
        "detour_densify_step_m": 0.3,
        "detour_max_attempts": 6,
        "detour_conflict_scan_waypoints": 8,
        "detour_smooth_neighbor_waypoints": 3,
        "robot_sweep_step_m": 0.3,
    },
}

# 可选初始化点集合 in map points
OptionalInitPointIds = {
    "human": [190, 191, 192, 193, 194, 195, 202, 203, 204, 205, 206, 207, 214, 215, 216, 217, 218, 219],
    "robot": [236, 272],
    "human_z": 0.1779,
    "robot_z": 0.24478,
}
# 可选初始化点集合 in map points (template refined with correct structure and type hints)
RouteOptionalInitPointsInMap: dict = {
    # Lists of initialization points (xyz[x,y,z])
    "human_xyz": torch.tensor([], dtype=torch.float32),
    "robot_xyz": torch.tensor([], dtype=torch.float32),
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


# Build init points (xy) for human/robot from point json.
_points_human = _load_points(CfgRoute["points_path_human"])
_points_robot = _load_points(CfgRoute["points_path_robot"])

RouteOptionalInitPointsInMap["human_xy"] = _extract_xy_points_by_ids(
    _points_human["points"], OptionalInitPointIds["human"]
)
RouteOptionalInitPointsInMap["robot_xy"] = _extract_xy_points_by_ids(
    _points_robot["points"], OptionalInitPointIds["robot"]
)

# Also expose xyz (xy + fixed z) for convenience.
RouteOptionalInitPointsInMap["human_xyz"] = torch.tensor(
    [
        [xy[0], xy[1], float(OptionalInitPointIds["human_z"])]
        for xy in RouteOptionalInitPointsInMap["human_xy"]
    ],
    dtype=torch.float32,
)
RouteOptionalInitPointsInMap["robot_xyz"] = torch.tensor(
    [
        [xy[0], xy[1], float(OptionalInitPointIds["robot_z"])]
        for xy in RouteOptionalInitPointsInMap["robot_xy"]
    ],
    dtype=torch.float32,
)