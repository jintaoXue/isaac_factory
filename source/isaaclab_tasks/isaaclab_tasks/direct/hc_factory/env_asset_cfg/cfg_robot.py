
from .cfg_hc_env import HcVectorEnvCfg
import torch
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
        },
        "offset_for_material_placement": {
            "position": torch.tensor([-1, -0.4, 0.2]),
        },
    }
}


CfgRobotRegistrationInfos = {
    "AGV": 2, #idx: 00-01
}