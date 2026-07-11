
from .cfg_hc_env import HcVectorEnvCfg
import torch

_BONE = "RL_BoneRoot"
_HIP = f"{_BONE}/Hip"
_PELVIS = f"{_HIP}/Pelvis"
_SPINE = f"{_HIP}/Waist/Spine01"
_SPINE02 = f"{_SPINE}/Spine02"

_HUMAN_JOINTS = {
    "waist": f"{_HIP}/Waist",
    "spine01": _SPINE,
    "spine02": _SPINE02,
    "upperarm_l": f"{_SPINE02}/L_Clavicle/L_Upperarm",
    "upperarm_r": f"{_SPINE02}/R_Clavicle/R_Upperarm",
    "forearm_l": f"{_SPINE02}/L_Clavicle/L_Upperarm/L_Forearm",
    "forearm_r": f"{_SPINE02}/R_Clavicle/R_Upperarm/R_Forearm",
    "thigh_l": f"{_PELVIS}/L_Thigh",
    "thigh_r": f"{_PELVIS}/R_Thigh",
    "calf_l": f"{_PELVIS}/L_Thigh/L_Calf",
    "calf_r": f"{_PELVIS}/R_Thigh/R_Calf",
    "foot_l": f"{_PELVIS}/L_Thigh/L_Calf/L_Foot",
    "foot_r": f"{_PELVIS}/R_Thigh/R_Calf/R_Foot",
}

# 上肢自然下垂基准
_ARM_IDLE_L = {"upperarm_l": (0.0, 0.0, -70.0), "forearm_l": (0.0, 0.0, -10.0)}
_ARM_IDLE_R = {"upperarm_r": (0.0, 0.0, 70.0), "forearm_r": (0.0, 0.0, 10.0)}
_LEG_NEUTRAL = {
    "thigh_l": (0.0, 0.0, 0.0),
    "thigh_r": (0.0, 0.0, 0.0),
    "calf_l": (0.0, 0.0, 0.0),
    "calf_r": (0.0, 0.0, 0.0),
    "foot_l": (0.0, 0.0, 0.0),
    "foot_r": (0.0, 0.0, 0.0),
}
_TORSO_NEUTRAL = {"waist": (0.0, 0.0, 0.0), "spine01": (0.0, 0.0, 0.0), "spine02": (0.0, 0.0, 0.0)}

# walk 正弦步态可调参数
_WALK_PARAMS = {
    "cycle_length": 22,          # 步态循环帧数（越大越慢）
    "thigh_swing_deg": 18.0,     # 髋关节前后摆幅（步幅）
    "knee_lift_deg": 20.0,       # 摆动相屈膝抬腿高度
    "knee_stance_deg": 5.0,      # 支撑相膝关节微屈
    "foot_lift_deg": 9.0,        # 摆动相踝关节背屈（抬脚）
    "foot_push_deg": 5.0,        # 支撑相蹬地踝关节伸展
    "arm_swing_deg": 22.0,       # 上臂前后摆幅（沿 X 轴）
    "forearm_follow_deg": 8.0,   # 前臂随摆臂轻微跟随
    "arm_hang_z_l": -70.0,       # 左臂下垂基准（锁定 Z，禁止横摆）
    "arm_hang_z_r": 70.0,        # 右臂下垂基准
    "torso_lean_deg": 5.0,       # 躯干前倾角度
}

# operate: 站立不动，小臂低位前伸，双手拉手风琴式内外小幅往复
_OPERATE_BASE = {
    **_TORSO_NEUTRAL,
    **_LEG_NEUTRAL,
    "waist": (1.0, 0.0, 0.0),
    "spine01": (4.0, 0.0, 0.0),
    "upperarm_l": (10.0, -10.0, -42.0),
    "upperarm_r": (10.0, 10.0, 42.0),
}

_OPERATE_KEYFRAMES = [
    {
        **_OPERATE_BASE,
        "forearm_l": (28.0, -14.0, -30.0),
        "forearm_r": (28.0, 14.0, 30.0),
    },
    {
        **_OPERATE_BASE,
        "forearm_l": (28.0, 6.0, -46.0),
        "forearm_r": (28.0, -6.0, 46.0),
    },
    {
        **_OPERATE_BASE,
        "forearm_l": (28.0, -10.0, -34.0),
        "forearm_r": (28.0, 10.0, 34.0),
    },
    {
        **_OPERATE_BASE,
        "forearm_l": (28.0, 4.0, -42.0),
        "forearm_r": (28.0, -4.0, 42.0),
    },
]

_HUMAN_ANIMATIONS = {
    "idle": {
        "type": "keyframes",
        "cycle_length": 1,
        "keyframes": [{**_TORSO_NEUTRAL, **_LEG_NEUTRAL, **_ARM_IDLE_L, **_ARM_IDLE_R}],
    },
    "walk": {
        "type": "procedural_sine",
        "cycle_length": _WALK_PARAMS["cycle_length"],
        "params": _WALK_PARAMS,
    },
    "operate": {
        "type": "keyframes",
        "cycle_length": 48,
        "keyframes": _OPERATE_KEYFRAMES,
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
            "detour_active": False,
            "detour_blocker_key": None,
            "detour_until_route_index": None,
        },
        "human_route_orientation_offset": {
            "orientation": torch.tensor([0.7071, 0, 0, 0.7071]),
        },
        "animation_cfg": {
            "joints": _HUMAN_JOINTS,
            "animations": _HUMAN_ANIMATIONS,
            "walk_params": _WALK_PARAMS,
        },
    }
}



CfgHumanRegistrationInfos = {
    "NormalHuman": 5, #idx 00-09
}
