"""Camera poses, sensor intrinsics, and manual ground visibility regions."""

from __future__ import annotations

from typing import Sequence

_CAMERA_SENSOR = {
    "width": 1080,
    "height": 720,
    "update_period": 0.0,
    "data_types": ["rgb"],
    "spawn": {
        "focal_length": 22.0,
        "focus_distance": 400.0,
        "horizontal_aperture": 30.0,
        "clipping_range": (0.1, 1.0e5),
    },
}

_CAMERA_RESET_STATE = {
    "key_variables": {"type_name": None, "idx": None, "machine_name": None},
    "rgb": None,
    "is_initialized": False,
}

# machine_name -> camera_name -> pose + optional manual ground footprint
#
# ground_footprint_xy: env-local (x, y) on factory floor (meters).
# Four corners in image order: top_left, top_right, bottom_right, bottom_left.
# Set to None until calibrated from sim screenshots; fill manually per camera.
#
# detect_human_id: if False, skip "which humans are in this image" labeling for
# this camera (still capture RGB). Highrise cameras are overview-only for now.
CAMERA_POSES = {
    "highrise_for_env": {
        "camera_num00_highrise_for_env": {
            "eye": (51, 9, 9),
            "lookat": (40, 9, 3.0),
            "detect_human_id": False,
            "ground_footprint_xy": None,
        },
        "camera_num01_highrise_for_env": {
            "eye": (-40, 9, 8.5),
            "lookat": (-30, 9, 3.0),
            "detect_human_id": False,
            "ground_footprint_xy": None,
        },
    },
    "storage_area": {
        "camera_num00_storage_area": {
            "eye": (-9, -3, 9),
            "lookat": (-9, -1, 4),
            "ground_footprint_xy": [[-13.49862, -2.98048], [-4.34776, -2.98048], [-15.0, 4.0], [-3.0, 4.0]],
        },
    },
    "num00_rotaryPipeAutomaticWeldingMachine": {
        "camera_num00_rotaryPipeAutomaticWeldingMachine_part_01_station": {
            "eye": (40, 20, 12),
            "lookat": (40, 14.5, 1.0),
            "ground_footprint_xy": [[30.39993, 7.7], [49.7, 7.7], [33.7, 19.2], [46.3, 19.2]],
        },
        "camera_num00_rotaryPipeAutomaticWeldingMachine_part_02_station": {
            "eye": (38, 20, 12),
            "lookat": (38, 14.5, 1.0),
            "ground_footprint_xy": [[29.0, 7.7], [47.5, 7.7], [32.0, 19.2], [44.0, 19.2]],
        },
    },
    "num01_weldingRobot": {
        "camera_num01_weldingRobot_part02_robot_arm_and_base": {
            "eye": (24.75003, 20, 12),
            "lookat": (24.75003, 14.5, 1.0),
            "ground_footprint_xy": [[16.0, 7.7], [34.0, 7.7], [18.52796, 19.2], [30.88481, 19.2]],
        },
    },
    "num02_rollerbedCNCPipeIntersectionCuttingMachine": {
        "camera_num02_rollerbedCNCPipeIntersectionCuttingMachine_part01_station": {
            "eye": (12, 20.5, 12),
            "lookat": (12, 15, 1.0),
            "ground_footprint_xy": [[3.0432, 8.2], [20.5, 8.2], [6.0, 19.7], [17.91964, 19.7]],
        },
    },
    "num04_groovingMachineLarge": {
        "camera_num04_groovingMachineLarge_part01_large_fixed_base": {
            "eye": (-7.65483, 20, 12),
            "lookat": (-7.65483, 14.5, 1.0),
            "ground_footprint_xy": [[-16.2, 7.7], [1.2, 7.7], [-13.38389, 19.2], [-1.59264, 19.2]],
        },
    },
    "num08_workbench": {
        "camera_num08_workbench": {
            "eye": (26, -3, 12),
            "lookat": (26, -1, 1.0),
            "ground_footprint_xy": [[19.59869, -3.7], [32.5, -3.7], [18.75174, 3.7], [33.3215, 3.7]],
        },
    },
}


def point_in_polygon(x: float, y: float, polygon_xy: Sequence[Sequence[float]]) -> bool:
    """Ray-casting point-in-polygon test."""
    n = len(polygon_xy)
    if n < 3:
        return False
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon_xy[i]
        xj, yj = polygon_xy[j]
        denom = yj - yi
        if (yi > y) != (yj > y):
            x_intersect = (xj - xi) * (y - yi) / (denom if abs(denom) > 1e-12 else 1e-12) + xi
            if x < x_intersect:
                inside = not inside
        j = i
    return inside


def visible_human_ids_in_camera(
    human_xy_by_id: dict[str, tuple[float, float]],
    ground_footprint_xy: Sequence[Sequence[float]] | None,
    *,
    detect_human_id: bool = True,
) -> list[str]:
    """Return human ids whose (x, y) lies inside the camera ground footprint."""
    if not detect_human_id or not ground_footprint_xy:
        return []
    visible: list[str] = []
    for human_id, (x, y) in human_xy_by_id.items():
        if point_in_polygon(x, y, ground_footprint_xy):
            visible.append(human_id)
    return sorted(visible)


def _iter_camera_pose_entries(
    camera_poses: dict,
) -> list[tuple[str, dict, str]]:
    entries: list[tuple[str, dict, str]] = []
    for machine_name, cameras in camera_poses.items():
        for camera_name, pose in cameras.items():
            entries.append((camera_name, pose, machine_name))
    return entries


def _make_machine_camera(
    camera_name: str,
    pose: dict,
    machine_name: str,
) -> dict:
    return {
        "type_name": camera_name,
        "machine_name": machine_name,
        "meta_registeration_info": {
            "prim_paths_expr": f"/World/envs/env_{{i}}/obj/HC_factory/{camera_name}",
            "name": camera_name,
        },
        "eye": pose["eye"],
        "lookat": pose["lookat"],
        "detect_human_id": bool(pose.get("detect_human_id", True)),
        "ground_footprint_xy": pose.get("ground_footprint_xy"),
        "camera_sensor": _CAMERA_SENSOR,
        "reset_state": _CAMERA_RESET_STATE,
        "debug_save_frames": True,
        "debug_max_frames": 50,
    }


CfgCamera = {
    camera_name: _make_machine_camera(camera_name, pose, machine_name)
    for camera_name, pose, machine_name in _iter_camera_pose_entries(CAMERA_POSES)
}

CfgCameraRegistrationInfos = {camera_name: 1 for camera_name in CfgCamera}


def has_registered_cameras() -> bool:
    return any(count > 0 for count in CfgCameraRegistrationInfos.values())


def camera_detects_human_id(camera_name: str) -> bool:
    """Whether this camera participates in human-id / in-frame labeling."""
    return bool(CfgCamera[camera_name].get("detect_human_id", True))


def get_ground_footprint_xy(camera_name: str) -> list[tuple[float, float]] | None:
    """Return manual ground footprint for a camera, or None if not configured / skipped."""
    if not camera_detects_human_id(camera_name):
        return None
    footprint = CfgCamera[camera_name].get("ground_footprint_xy")
    if footprint is None:
        return None
    return [tuple(pt) for pt in footprint]


def visible_human_ids_for_camera(
    camera_name: str,
    human_xy_by_id: dict[str, tuple[float, float]],
) -> list[str]:
    """Convenience: resolve detect flag + footprint, then test human positions."""
    return visible_human_ids_in_camera(
        human_xy_by_id,
        get_ground_footprint_xy(camera_name),
        detect_human_id=camera_detects_human_id(camera_name),
    )
