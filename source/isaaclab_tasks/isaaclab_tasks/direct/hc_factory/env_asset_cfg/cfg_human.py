
from .cfg_hc_env import HcVectorEnvCfg
CfgHuman = {
    "NumUpperBound": HcVectorEnvCfg().human_number_upper_bound,
    "NormalHuman": {
        "type_id": 00,
        "type_name": "NormalHuman",
        "meta_registeration_info": {
            "rigid_prim_paths_expr": "/World/envs/env_{i}/obj/HC_factory/human_robot_group/human_{idx}",
            "skeleton_prim_paths_expr": "/World/envs/env_{i}/obj/HC_factory/human_robot_group/human_{idx}/ManRoot/male_adult_construction_01/male_adult_construction_01/male_adult_construction_01",
            "name": "human_00_{idx}",
        },
        "reset_state": {
            "key_variables": {
                "type_name": None,
                "idx": None,
            },
            #states: free, working_task gallery + done
            "state": "free",
            "ongoing_task_record": {},
            "current_area_id": None,
            "target_area_id": None,
            "arrived_target_area": False,
            "subtask_time_counter": 0,
            "generated_route": [],
            "route_index": 0,
            "route_length": 0,
        },
    }
}



CfgHumanRegistrationInfos = {
    "NormalHuman": 5, #idx 00-09
}