
from .cfg_hc_env import HcVectorEnvCfg
import torch


def robot_central_point_offset_from_footprint_bounds(
    local_bounds: dict[str, float],
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Geometric center of the robot footprint in local (x, y), as offset from pose origin.

    Local footprint is defined relative to the robot pose origin. For AGV bounds
    with origin at one corner (``min_*=0``), this returns mid-box ``(cx, cy, 0)``.
    """
    cx = 0.5 * (float(local_bounds["min_x"]) + float(local_bounds["max_x"]))
    cy = 0.5 * (float(local_bounds["min_y"]) + float(local_bounds["max_y"]))
    return torch.tensor([cx, cy, 0.0], dtype=torch.float32, device=device)


_AGV_FOOTPRINT_LOCAL_BOUNDS = {
    "min_x": 0.0,
    "max_x": 1.8,
    "min_y": 0.0,
    "max_y": 0.8,
}

CfgRobot = {
    "NumUpperBound": HcVectorEnvCfg().robot_upper_bound,
    "AGV": {
        "type_id": 00,
        "type_name": "AGV",
        "meta_registeration_info": {
            "prim_paths_expr": "/World/envs/env_{i}/obj/HC_factory/human_robot_group/robot_AGV_{idx}",
            "name": "robot_00_{idx}",
        },
        "reset_state": {
            "key_variables": {
                "type_name": None,
                "idx": None,
            },
            #states: free, working_taskName + done
            "state": "free",
            "ongoing_task_record_index": None,
            "current_area_id": None,
            "target_area_id": None,
            "generated_route": [],
            "route_index": 0,
            "route_length": 0,
            "detour_active": False,
            "detour_blocker_key": None,
            "detour_until_route_index": None,
            "yield_active": False,
            "yield_blocker_key": None,
        },
        "offset_for_material_placement": {
            "position": torch.tensor([-1, -0.4, 0.2]),
        },
        "robot_footprint_local_bounds": _AGV_FOOTPRINT_LOCAL_BOUNDS,
        "robot_central_point_offset": robot_central_point_offset_from_footprint_bounds(
            _AGV_FOOTPRINT_LOCAL_BOUNDS
        ),
    }
}


CfgRobotRegistrationInfos = {
    "AGV": 2, #idx: 00-01
}

RobotIdAppearance = {
    "type_id": 00,
    "type_name": "AGV",
    "appearance": {
        "num_00": {
            "color": "yellow",
            "rgb_hint": (0.95, 0.85, 0.15),
        },
        "num_01": {
            "color": "purple",
            "rgb_hint": (0.50, 0.10, 0.50),
        },
    }
}
