


CfgRobot = {
    
    "AGV": {
        "type_id": 00,
        "type_name": "AGV",
        "meta_registeration_info": {
            "prim_paths_expr": "/World/envs/env_{i}/obj/HC_factory/human_robot_group/robot_AGV_{idx}",
            "name": "robot_00_{idx}",
        },
        "state_gallery": {
            0: "free",
            1: "moving_to_working",
            2: "working",
            3: "resetting",
        },
        "reset_state": {
            "state": [0],
            "current_pose": [None, None],
            "target_pose": [None, None],
        },
    }
}


CfgRobotRegistrationInfos = {
    "AGV": 2, #idx: 00-01
}