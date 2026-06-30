CfgCamera = {
    "TestCamera": {
        "type_id": 0,
        "type_name": "TestCamera",
        "meta_registeration_info": {
            "prim_paths_expr": "/World/envs/env_{i}/test_camera_{idx}",
            "name": "test_camera_{idx}",
        },
        # overview (bird's-eye)
        "eye": (0.0, 30.0, 100.0),
        "lookat": (0.0, 10.0, 10.0),
        "camera_sensor": {
            "width": 640,
            "height": 480,
            "update_period": 0.0,
            "data_types": ["rgb"],
            "spawn": {
                "focal_length": 24.0,
                "focus_distance": 400.0,
                "horizontal_aperture": 20.955,
                "clipping_range": (0.1, 1.0e5),
            },
        },
        "reset_state": {
            "key_variables": {
                "type_name": None,
                "idx": None,
            },
            "rgb": None,
            "is_initialized": False,
        },
        "debug_save_frames": True,
        "debug_max_frames": 10,
    },
    "DetailCamera": {
        "type_id": 1,
        "type_name": "DetailCamera",
        "meta_registeration_info": {
            "prim_paths_expr": "/World/envs/env_{i}/detail_camera_{idx}",
            "name": "detail_camera_{idx}",
        },
        "eye": (40.0, 21.0, 12.0),
        "lookat": (40.0, 16.0, 2.0),
        "camera_sensor": {
            "width": 640,
            "height": 480,
            "update_period": 0.0,
            "data_types": ["rgb"],
            "spawn": {
                "focal_length": 24.0,
                "focus_distance": 400.0,
                "horizontal_aperture": 20.955,
                "clipping_range": (0.1, 1.0e5),
            },
        },
        "reset_state": {
            "key_variables": {
                "type_name": None,
                "idx": None,
            },
            "rgb": None,
            "is_initialized": False,
        },
        "debug_save_frames": True,
        "debug_max_frames": 10,
    },
}

CfgCameraRegistrationInfos = {
    "TestCamera": 0,
    "DetailCamera": 0,
}


def has_registered_cameras() -> bool:
    return any(count > 0 for count in CfgCameraRegistrationInfos.values())