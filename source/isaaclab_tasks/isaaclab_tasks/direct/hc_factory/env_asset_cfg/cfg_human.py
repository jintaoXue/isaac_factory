


CfgHuman = {

    "NormalHuman": {
        "type_id": 00,
        "type_name": "NormalHuman",
        "meta_registeration_info": {
            "rigid_prim_paths_expr": "/World/envs/env_{i}/obj/HC_factory/human_robot_group/human_{idx}",
            "skeleton_prim_paths_expr": "/World/envs/env_{i}/obj/HC_factory/human_robot_group/human_{idx}/ManRoot/male_adult_construction_01/male_adult_construction_01/male_adult_construction_01",
            "name": "human_00_{idx}",
        },
        "state_gallery": {
            0: "free",
            1: "moving_to_working",
            2: "working",
            3: "resetting",
        },
        "reset_state": {
            "state": "free",
        },
    }
}



CfgHumanRegistrationInfos = {
    "NormalHuman": 5, #idx 00-09
}