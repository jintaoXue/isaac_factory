_CAMERA_SENSOR = {
    "width": 640,
    "height": 480,
    "update_period": 0.0,
    "data_types": ["rgb"],
    "spawn": {
        "focal_length": 22.0,
        "focus_distance": 400.0,
        "horizontal_aperture": 24.0,
        "clipping_range": (0.1, 1.0e5),
    },
}

_CAMERA_RESET_STATE = {
    "key_variables": {"type_name": None, "idx": None, "machine_name": None},
    "rgb": None,
    "is_initialized": False,
}

# 每台 machine 一个相机，type_name 均为 camera_for_machine；eye/lookat 请按机器自行调整
CAMERA_FOR_MACHINE_CFG = {
    "camera_num00_rotaryPipeAutomaticWeldingMachine": {
        "eye": (38, 20, 12),
        "lookat": (38, 14.5, 1.0),
    },
    "camera_num01_weldingRobot": {
        "eye": (38, 20, 12),
        "lookat": (38, 14.5, 1.0),
    },
    "camera_num02_rollerbedCNCPipeIntersectionCuttingMachine": {
        "eye": (38, 20, 12),
        "lookat": (38, 14.5, 1.0),
    },
    "camera_num03_laserCuttingMachine": {
        "eye": (38, 20, 12),
        "lookat": (38, 14.5, 1.0),
    },
    "camera_num04_groovingMachineLarge": {
        "eye": (38, 20, 12),
        "lookat": (38, 14.5, 1.0),
    },
    "camera_num05_groovingMachineSmall": {
        "eye": (38, 20, 12),
        "lookat": (38, 14.5, 1.0),
    },
    "camera_num06_highPressureFoamingMachine": {
        "eye": (38, 20, 12),
        "lookat": (38, 14.5, 1.0),
    },
    "camera_num07_gantry_group": {
        "eye": (38, 20, 12),
        "lookat": (38, 14.5, 1.0),
    },
    "camera_num08_workbench": {
        "eye": (38, 20, 12),
        "lookat": (38, 14.5, 1.0),
    }
}

def _make_machine_camera(camera_name: str, eye: tuple[float, float, float], lookat: tuple[float, float, float]) -> dict:
    return {
        "type_name": camera_name,
        "machine_name": camera_name,
        "meta_registeration_info": {
            "prim_paths_expr": f"/World/envs/env_{{i}}/obj/HC_factory/{camera_name}",
            "name": camera_name,
        },
        "eye": eye,
        "lookat": lookat,
        "camera_sensor": _CAMERA_SENSOR,
        "reset_state": _CAMERA_RESET_STATE,
        "debug_save_frames": True,
        "debug_max_frames": 10,
    }


CfgCamera = {
    camera_name: _make_machine_camera(camera_name, pose["eye"], pose["lookat"])
    for camera_name, pose in CAMERA_FOR_MACHINE_CFG.items()
}

CfgCameraRegistrationInfos = {camera_name: 1 for camera_name in CAMERA_FOR_MACHINE_CFG}


def has_registered_cameras() -> bool:
    return any(count > 0 for count in CfgCameraRegistrationInfos.values())
