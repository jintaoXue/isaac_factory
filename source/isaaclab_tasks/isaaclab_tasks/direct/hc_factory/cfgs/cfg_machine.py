

cfg_machines = {
    "num01_rotaryPipeAutomaticWeldingMachine": {
        "multiple_parts": True,
        "num01_rotaryPipeAutomaticWeldingMachine_part_01_station": {
            "prim_paths_expr": "/World/envs/.*/obj/HC_factory/num01_rotaryPipeAutomaticWeldingMachine/part_01_station/track_for_mobile_base",
            "joint_positions_working": [0.0, 2.0],
            "animation_time": 100,
        },
        "num01_rotaryPipeAutomaticWeldingMachine_part_02_station": {
            "prim_paths_expr": "/World/envs/.*/obj/HC_factory/num01_rotaryPipeAutomaticWeldingMachine/part_02_station/track_for_mobile_base_001",
            "joint_positions_working": [0.0, 0.5],
            "animation_time": 50,
        },
    },

    "num02_weldingRobot": {
        "multiple_parts": True,
        "num02_weldingRobot_part02_robot_arm_and_base": {
            "prim_paths_expr": "/World/envs/.*/obj/HC_factory/num02_weldingRobot/part02_robot_arm_and_base",
            "joint_positions_working": [3.2, -1.5, -0.3, 0.1, 0.2, 0.0],
            "animation_time": 100,
        },
        "num02_weldingRobot_part04_mobile_base_for_material": {
            "prim_paths_expr": "/World/envs/.*/obj/HC_factory/num02_weldingRobot/part04_mobile_base_for_material",
            "joint_positions_working": [-2.0],
            "animation_time": 100,
        },
    },

    "num03_rollerbedCNCPipeIntersectionCuttingMachine": {
        "multiple_parts": True,
        "num03_rollerbedCNCPipeIntersectionCuttingMachine_part01_station": {
            "prim_paths_expr": "/World/envs/.*/obj/HC_factory/num03_rollerbedCNCPipeIntersectionCuttingMachine/part01_station",
            "joint_positions_working": [1.0],
            "animation_time": 100,
        },
        "num03_rollerbedCNCPipeIntersectionCuttingMachine_part05_cutting_machine": {
            "prim_paths_expr": "/World/envs/.*/obj/HC_factory/num03_rollerbedCNCPipeIntersectionCuttingMachine/part05_cutting_machine",
            "joint_positions_working": [-2.0, 0.3, 0.5],
            "animation_time": 100,
        },
    },

    "num04_laserCuttingMachine": {
        "multiple_parts": False,
        "prim_paths_expr": "/World/envs/.*/obj/HC_factory/num04_laserCuttingMachine",
        "joint_positions_working": [-3.5],
        "animation_time": 100,
    },

    "num05_groovingMachineLarge": {
        "multiple_parts": True,
        "num05_groovingMachineLarge_part01_large_fixed_base": {
            "prim_paths_expr": "/World/envs/.*/obj/HC_factory/num05_groovingMachineLarge/part01_large_fixed_base",
            "joint_positions_working": [-0.2, 0.0],
            "animation_time": 100,
        },
        "num05_groovingMachineLarge_part02_large_mobile_base": {
            "prim_paths_expr": "/World/envs/.*/obj/HC_factory/num05_groovingMachineLarge/part02_large_mobile_base",
            "joint_positions_working": [-1.0],
            "animation_time": 100,
        },
    },
    
    "num06_groovingMachineSmall": {
        "multiple_parts": True,
        "num06_groovingMachineSmall_part01_small_fixed_base": {
            "prim_paths_expr": "/World/envs/.*/obj/HC_factory/num06_groovingMachineSmall/part01_small_fixed_base",
            "joint_positions_working": [-0.3, -0.5],
            "animation_time": 100,
        },
        "num06_groovingMachineSmall_part02_small_mobile_handle": {
            "prim_paths_expr": "/World/envs/.*/obj/HC_factory/num06_groovingMachineSmall/part02_small_mobile_handle",
            "joint_positions_working": [-0.3, -0.5],
            "animation_time": 100,
        },
    },

    "num07_highPressureFoamingMachine": {
        "multiple_parts": False,
        "prim_paths_expr": "/World/envs/.*/obj/HC_factory/num07_highPressureFoamingMachine",
        "joint_positions_working": [-0.7],
        "animation_time": 100,
    },
    
    "num08_gantry_group": {
        "multiple_parts": False,
        "prim_paths_expr": "/World/envs/.*/obj/HC_factory/num08_gantry_group/gantry_00",
        "joint_positions_working": [10.0, 10.0, 10.0, 10.0, 5.0, -5.0, 5.0, -5.0],
        "animation_time": 100,
    },
    # # 下面这些如果未来要用，可以把 hc_env_base 里对应的代码打开并补齐 cfg：
    # "num08_gantry_01": {"prim_paths_expr": "...", "reset_xform_properties": False},
    # "num08_gantry_02": {"prim_paths_expr": "...", "reset_xform_properties": False},
    # "num08_gantry_03": {"prim_paths_expr": "...", "reset_xform_properties": False},
}