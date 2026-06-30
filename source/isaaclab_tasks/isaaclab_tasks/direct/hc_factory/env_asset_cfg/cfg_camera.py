CfgCamera = {
    "TestCamera": {
        "type_id": 0,
        "type_name": "TestCamera",
        "meta_registeration_info": {
            "prim_paths_expr": "/World/envs/env_{i}/test_camera_{idx}",
            "name": "test_camera_{idx}",
        },
        # env-local 坐标（相对 /World/envs/env_{i}，vector env clone 后自动对齐）
        # 俯视鸟瞰：eye/lookat 保持相同 X,Y
        "eye": (40.0, 15.0, 100.0),
        "lookat": (40.0, 15.0, 0.5),
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
        "eye": (35, 15.5, 16.5),
        "lookat": (38, 17.5, 2.0),
        "camera_sensor": {
            # 分辨率范围: width, height 可设置为 320~1920, 240~1080
            "width": 1920,    # 推荐范围：320~1920
            "height": 1080,   # 推荐范围：240~1080
     
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
    "TestCamera": 1,
    "DetailCamera": 1,
}


def has_registered_cameras() -> bool:
    return any(count > 0 for count in CfgCameraRegistrationInfos.values())