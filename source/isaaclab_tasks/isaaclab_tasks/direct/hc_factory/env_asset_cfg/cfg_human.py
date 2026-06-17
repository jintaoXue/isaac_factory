
from .cfg_hc_env import HcVectorEnvCfg
import torch

_HUMAN_REST_POSE_JOINTS = {
    "upperarm_l": "RL_BoneRoot/Hip/Waist/Spine01/Spine02/L_Clavicle/L_Upperarm",
    "upperarm_r": "RL_BoneRoot/Hip/Waist/Spine01/Spine02/R_Clavicle/R_Upperarm",
    "forearm_l": "RL_BoneRoot/Hip/Waist/Spine01/Spine02/L_Clavicle/L_Upperarm/L_Forearm",
    "forearm_r": "RL_BoneRoot/Hip/Waist/Spine01/Spine02/R_Clavicle/R_Upperarm/R_Forearm",
}

_HUMAN_REST_POSE_LIBRARY = {
    "idle": {
        "upperarm_l": (0.0, 0.0, -70.0),
        "upperarm_r": (0.0, 0.0, 70.0),
        "forearm_l": (0.0, 0.0, -10.0),
        "forearm_r": (0.0, 0.0, 10.0),
    },
    "walk": {
        "upperarm_l": (0.0, 0.0, -35.0),
        "upperarm_r": (0.0, 0.0, 35.0),
        "forearm_l": (0.0, 0.0, -20.0),
        "forearm_r": (0.0, 0.0, 20.0),
    },
    "operate": {
        "upperarm_l": (15.0, -20.0, -45.0),
        "upperarm_r": (15.0, 20.0, 45.0),
        "forearm_l": (0.0, 0.0, -60.0),
        "forearm_r": (0.0, 0.0, 60.0),
    },
}

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
            #states: free, working_task gallery
            "state": "free",
            "ongoing_task_record_index": None,
            "current_area_id": None,
            "target_area_id": None,
            "subtask_time_counter": 0,
            "generated_route": [],
            "route_index": 0,
            "route_length": 0,
        },
        "human_route_orientation_offset": {
            "orientation": torch.tensor([0.7071, 0, 0, 0.7071]),
        },
        "rest_pose_cfg": {
            "joints": _HUMAN_REST_POSE_JOINTS,
            "poses": _HUMAN_REST_POSE_LIBRARY,
        },
    }
}



CfgHumanRegistrationInfos = {
    "NormalHuman": 5, #idx 00-09
}