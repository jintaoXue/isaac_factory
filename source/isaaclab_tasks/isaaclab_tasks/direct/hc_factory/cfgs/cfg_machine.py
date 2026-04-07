

cfg_machines = {
    "num01_rotaryPipeAutomaticWeldingMachine": {
        "registration_type": "articulation",
        "num_workstations": 2,
        "num_registration_parts": 2,
        "registeration_infos": {
            "num01_rotaryPipeAutomaticWeldingMachine_part_01_station": {
                "prim_paths_expr": "/World/envs/.*/obj/HC_factory/num01_rotaryPipeAutomaticWeldingMachine/part_01_station/track_for_mobile_base",
                "joint_positions_working": [0.0, 2.0],
                "animation_time": 100,
                # 只有“工位/工作站（station）”这种可以处理物料的节点，才需要配置 working area ids。
                # human_working_areas_ids 的编号来源：map_data/map_with_points_human.png（图中蓝色编号）
                "human_working_areas_ids": [56],
                # agv/gantry 的停车区编号来源：map_data/map_with_points_robot.png（图中蓝色编号）
                "agv_parking_areas_ids": [229],
                "gantry_parking_areas_ids": [229],
            },
            "num01_rotaryPipeAutomaticWeldingMachine_part_02_station": {
                "prim_paths_expr": "/World/envs/.*/obj/HC_factory/num01_rotaryPipeAutomaticWeldingMachine/part_02_station/track_for_mobile_base_001",
                "joint_positions_working": [0.0, 0.5],
                "animation_time": 50,
                "human_working_areas_ids": [60],
                "agv_parking_areas_ids": [233],
                "gantry_parking_areas_ids": [233],
            },
        },
    },

    "num02_weldingRobot": {
        "registration_type": "articulation",
        "num_workstations": 1,
        "num_registration_parts": 2,
        "registeration_infos": {
            "num02_weldingRobot_part02_robot_arm_and_base": {
                "prim_paths_expr": "/World/envs/.*/obj/HC_factory/num02_weldingRobot/part02_robot_arm_and_base",
                "joint_positions_working": [3.2, -1.5, -0.3, 0.1, 0.2, 0.0],
                "animation_time": 100,
                "human_working_areas_ids": [64, 65, 66],
                "agv_parking_areas_ids": [238],
                "gantry_parking_areas_ids": [238],
            },
            "num02_weldingRobot_part04_mobile_base_for_material": {
                "prim_paths_expr": "/World/envs/.*/obj/HC_factory/num02_weldingRobot/part04_mobile_base_for_material",
                "joint_positions_working": [-2.0],
                "animation_time": 100,
            },
        },
    },

    "num03_rollerbedCNCPipeIntersectionCuttingMachine": {
        "registration_type": "articulation",
        "num_workstations": 1,
        "num_registration_parts": 2,
        "registeration_infos": {
            "num03_rollerbedCNCPipeIntersectionCuttingMachine_part01_station": {
                "prim_paths_expr": "/World/envs/.*/obj/HC_factory/num03_rollerbedCNCPipeIntersectionCuttingMachine/part01_station",
                "joint_positions_working": [1.0],
                "animation_time": 100,
                "human_working_areas_ids": [89, 90],
                "agv_parking_areas_ids": [243, 244],
                "gantry_parking_areas_ids": [243, 244],
            },
            "num03_rollerbedCNCPipeIntersectionCuttingMachine_part05_cutting_machine": {
                "prim_paths_expr": "/World/envs/.*/obj/HC_factory/num03_rollerbedCNCPipeIntersectionCuttingMachine/part05_cutting_machine",
                "joint_positions_working": [-2.0, 0.3, 0.5],
                "animation_time": 100,
            },
        },
    },

    "num04_laserCuttingMachine": {
        "registration_type": "articulation",
        "num_workstations": 1,
        "num_registration_parts": 1,
        "registeration_infos": {
            "num04_laserCuttingMachine": {
                "prim_paths_expr": "/World/envs/.*/obj/HC_factory/num04_laserCuttingMachine",
                "joint_positions_working": [-3.5],
                "animation_time": 100,
                "human_working_areas_ids": [113, 114],
                "agv_parking_areas_ids": [189],
                "gantry_parking_areas_ids": [189],
            },
        },
    },

    "num05_groovingMachineLarge": {
        "registration_type": "articulation",
        "num_workstations": 1,
        "num_registration_parts": 2,
        "registeration_infos": {
            "num05_groovingMachineLarge_part01_large_fixed_base": {
                "prim_paths_expr": "/World/envs/.*/obj/HC_factory/num05_groovingMachineLarge/part01_large_fixed_base",
                "joint_positions_working": [-0.2, 0.0],
                "animation_time": 100,
                "human_working_areas_ids": [341, 342],
                "agv_parking_areas_ids": [96],
                "gantry_parking_areas_ids": [96],
            },
            "num05_groovingMachineLarge_part02_large_mobile_base": {
                "prim_paths_expr": "/World/envs/.*/obj/HC_factory/num05_groovingMachineLarge/part02_large_mobile_base",
                "joint_positions_working": [-1.0],
                "animation_time": 100,
            },
        },
    },
    
    "num06_groovingMachineSmall": {
        "registration_type": "articulation",
        "num_workstations": 1,
        "num_registration_parts": 2,
        "registeration_infos": {
            "num06_groovingMachineSmall_part01_small_fixed_base": {
                "prim_paths_expr": "/World/envs/.*/obj/HC_factory/num06_groovingMachineSmall/part01_small_fixed_base",
                "joint_positions_working": [-0.3, -0.5],
                "animation_time": 100,
                "human_working_areas_ids": [160],
                "agv_parking_areas_ids": [139],
                "gantry_parking_areas_ids": [139],
            },
            "num06_groovingMachineSmall_part02_small_mobile_handle": {
                "prim_paths_expr": "/World/envs/.*/obj/HC_factory/num06_groovingMachineSmall/part02_small_mobile_handle",
                "joint_positions_working": [-0.3, -0.5],
                "animation_time": 100,
            },
        },
    },

    "num07_highPressureFoamingMachine": {
        "registration_type": "articulation",
        "num_workstations": 1,
        "num_registration_parts": 1,
        "registeration_infos": {
            "num07_highPressureFoamingMachine": {
                "prim_paths_expr": "/World/envs/.*/obj/HC_factory/num07_highPressureFoamingMachine",
                "joint_positions_working": [-0.7],
                "animation_time": 100,
                "human_working_areas_ids": [130],
                "agv_parking_areas_ids": [85],
                "gantry_parking_areas_ids": [85],
            },
        },
    },

    "num08_gantry_group": {
        "registration_type": "articulation",
        "num_workstations": 4,
        "num_registration_parts": 1,
        "registeration_infos": {
            "num08_gantry_group": {
                "prim_paths_expr": "/World/envs/.*/obj/HC_factory/num08_gantry_group/gantry_00",
                "joint_positions_working": [10.0, 10.0, 10.0, 10.0, 5.0, -5.0, 5.0, -5.0],
                "animation_time": 100,
                "human_working_areas_ids": [],
                "agv_parking_areas_ids": [],
                "gantry_parking_areas_ids": [],
            },
        },
    },

    "num09_workbench": {
        "registration_type": "articulation",
        "num_workstations": 2,
        "num_registration_parts": 1,
        "registeration_infos": {
            "num09_workbench": {
                "prim_paths_expr": "/World/envs/.*/obj/HC_factory/num09_workbench",
                "joint_positions_working": [0.0],
                "animation_time": 100,
                "human_working_areas_ids": [],
                "agv_parking_areas_ids": [],
                "gantry_parking_areas_ids": [],
            },
        },
    },
    # # 下面这些如果未来要用，可以把 hc_env_base 里对应的代码打开并补齐 cfg：
    # "num08_gantry_01": {"prim_paths_expr": "...", "reset_xform_properties": False},
    # "num08_gantry_02": {"prim_paths_expr": "...", "reset_xform_properties": False},
    # "num08_gantry_03": {"prim_paths_expr": "...", "reset_xform_properties": False},
}