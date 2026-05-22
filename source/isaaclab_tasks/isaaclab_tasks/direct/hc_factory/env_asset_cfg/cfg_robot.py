
from .cfg_hc_env import HcVectorEnvCfg

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
            "arrived_target_area": False,
            "generated_route": [],
            "route_index": 0,
            "route_length": 0,
        },
    }
}


CfgRobotRegistrationInfos = {
    "AGV": 2, #idx: 00-01
}