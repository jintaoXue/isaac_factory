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

# machine_name -> camera_name -> {eye, lookat}
CAMERA_POSES = {
    "highrise_for_env":{
        "camera_num00_highrise_for_env": {
            "eye": (-3, -2.83729, 10),
            "lookat": (14, 8.76496, 1.0),
        },
        "camera_num01_highrise_for_env": {
            "eye": (10, -2.83729, 10),
            "lookat": (-17, 7.76346, 1.0),
        },
    },
    "storage_for_env":{
        "camera_num00_storage_for_env": {
            "eye": (0, 0, 0),
            "lookat": (0, 0, 0),
        },
    },
    "num00_rotaryPipeAutomaticWeldingMachine": {
        "camera_num00_rotaryPipeAutomaticWeldingMachine_part_01_station": {
            "eye": (40, 20, 12),
            "lookat": (40, 14.5, 1.0),
        },
        "camera_num00_rotaryPipeAutomaticWeldingMachine_part_02_station": {
            "eye": (38, 20, 12),
            "lookat": (38, 14.5, 1.0),
        },
    },
    "num01_weldingRobot": {
        "camera_num01_weldingRobot_part02_robot_arm_and_base": {
            "eye": (24.75003, 20, 12),
            "lookat": (24.75003, 14.5, 1.0),
        },
    },
    "num02_rollerbedCNCPipeIntersectionCuttingMachine": {
        "camera_num02_rollerbedCNCPipeIntersectionCuttingMachine_part01_station": {
            "eye": (10, 20, 12),
            "lookat": (10, 14.5, 1.0),
        },
    },
    "num04_groovingMachineLarge": {
        "camera_num04_groovingMachineLarge_part01_large_fixed_base": {
            "eye": (-7.65483, 20, 12),
            "lookat": (-7.65483, 14.5, 1.0),
        },
    },
    "num08_workbench": {
        "camera_num08_workbench": {
            "eye": (26, 1, 12),
            "lookat": (26, -4.5, 1.0),
        },
    },
}


def _iter_camera_pose_entries(
    camera_poses: dict,
) -> list[tuple[str, dict[str, tuple[float, float, float]], str]]:
    """展开 machine -> camera -> pose，返回 (camera_name, pose, machine_name)。"""
    entries: list[tuple[str, dict[str, tuple[float, float, float]], str]] = []
    for machine_name, cameras in camera_poses.items():
        for camera_name, pose in cameras.items():
            entries.append((camera_name, pose, machine_name))
    return entries


def _make_machine_camera(
    camera_name: str,
    eye: tuple[float, float, float],
    lookat: tuple[float, float, float],
    machine_name: str,
) -> dict:
    return {
        "type_name": camera_name,
        "machine_name": machine_name,
        "meta_registeration_info": {
            "prim_paths_expr": f"/World/envs/env_{{i}}/obj/HC_factory/{camera_name}",
            "name": camera_name,
        },
        "eye": eye,
        "lookat": lookat,
        "camera_sensor": _CAMERA_SENSOR,
        "reset_state": _CAMERA_RESET_STATE,
        "debug_save_frames": False,
        "debug_max_frames": 50,
    }


CfgCamera = {
    camera_name: _make_machine_camera(camera_name, pose["eye"], pose["lookat"], machine_name)
    for camera_name, pose, machine_name in _iter_camera_pose_entries(CAMERA_POSES)
}

CfgCameraRegistrationInfos = {camera_name: 1 for camera_name in CfgCamera}


def has_registered_cameras() -> bool:
    return any(count > 0 for count in CfgCameraRegistrationInfos.values())
