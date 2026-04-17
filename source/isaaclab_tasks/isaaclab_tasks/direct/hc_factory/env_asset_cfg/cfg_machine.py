

CfgMachine = {
    "num00_rotaryPipeAutomaticWeldingMachine": {
        "type_id": "00",
        "type_name": "num00_rotaryPipeAutomaticWeldingMachine",
        "registration_type": "articulation",
        "num_workstations": 2,
        "num_registration_parts": 2,
        "registeration_infos": {
            "num00_rotaryPipeAutomaticWeldingMachine_part_01_station": {
                #prim_paths_expr is the path in .usd file
                "prim_paths_expr": "/World/envs/env_{i}/obj/HC_factory/num00_rotaryPipeAutomaticWeldingMachine/part_01_station/track_for_mobile_base",
                "joint_positions_working": [0.0, 2.0],
                "joint_positions_reset": [0.0, 0.0],
                "animation_time": 100,
                # 只有“工位/工作站（station）”这种可以处理物料的节点，才需要配置 working area ids。
                # human_working_areas_ids 的编号来源：map_data/map_with_points_human.png（图中蓝色编号）
                "human_working_areas_ids": [56],
                # agv/gantry 的停车区编号来源：map_data/map_with_points_robot.png（图中蓝色编号）
                "robot_parking_areas_ids": [229],
                "gantry_parking_areas_ids": [229],
            },
            "num00_rotaryPipeAutomaticWeldingMachine_part_02_station": {
                "prim_paths_expr": "/World/envs/env_{i}/obj/HC_factory/num00_rotaryPipeAutomaticWeldingMachine/part_02_station/track_for_mobile_base_001",
                "joint_positions_working": [0.0, 0.5],
                "joint_positions_reset": [0.0, 0.0],
                "animation_time": 50,
                "human_working_areas_ids": [60],
                "robot_parking_areas_ids": [233],
                "gantry_parking_areas_ids": [233],
            },
        },
    },
    "num01_weldingRobot": {
        "type_id": "01",
        "type_name": "num01_weldingRobot",
        "registration_type": "articulation",
        "num_workstations": 1,
        "num_registration_parts": 2,
        "registeration_infos": {
            "num01_weldingRobot_part02_robot_arm_and_base": {
                "prim_paths_expr": "/World/envs/env_{i}/obj/HC_factory/num01_weldingRobot/part02_robot_arm_and_base",
                "joint_positions_working": [3.2, -1.5, -0.3, 0.1, 0.2, 0.0],
                "joint_positions_reset": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "animation_time": 100,
                "human_working_areas_ids": [64, 65, 66],
                "robot_parking_areas_ids": [238],
                "gantry_parking_areas_ids": [238],
            },
            "num01_weldingRobot_part04_mobile_base_for_material": {
                "prim_paths_expr": "/World/envs/env_{i}/obj/HC_factory/num01_weldingRobot/part04_mobile_base_for_material",
                "joint_positions_working": [-2.0],
                "joint_positions_reset": [0.0],
                "animation_time": 100,
            },
        },
    },

    "num02_rollerbedCNCPipeIntersectionCuttingMachine": {
        "type_id": "02",
        "type_name": "num02_rollerbedCNCPipeIntersectionCuttingMachine",
        "registration_type": "articulation",
        "num_workstations": 1,
        "num_registration_parts": 2,
        "registeration_infos": {
            "num02_rollerbedCNCPipeIntersectionCuttingMachine_part01_station": {
                "prim_paths_expr": "/World/envs/env_{i}/obj/HC_factory/num02_rollerbedCNCPipeIntersectionCuttingMachine/part01_station",
                "joint_positions_working": [1.0],
                "joint_positions_reset": [0.0],
                "animation_time": 100,
                "human_working_areas_ids": [89, 90],
                "robot_parking_areas_ids": [243, 244],
                "gantry_parking_areas_ids": [243, 244],
            },
            "num02_rollerbedCNCPipeIntersectionCuttingMachine_part05_cutting_machine": {
                "prim_paths_expr": "/World/envs/env_{i}/obj/HC_factory/num02_rollerbedCNCPipeIntersectionCuttingMachine/part05_cutting_machine",
                "joint_positions_working": [-2.0, 0.3, 0.5],
                "joint_positions_reset": [0.0, 0.0, 0.0],
                "animation_time": 100,
            },
        },
    },

    "num03_laserCuttingMachine": {
        "type_id": "03",
        "type_name": "num03_laserCuttingMachine",
        "registration_type": "articulation",
        "num_workstations": 1,
        "num_registration_parts": 1,
        "registeration_infos": {
            "num03_laserCuttingMachine": {
                "prim_paths_expr": "/World/envs/env_{i}/obj/HC_factory/num03_laserCuttingMachine",
                "joint_positions_working": [-3.5],
                "joint_positions_reset": [0.0],
                "animation_time": 100,
                "human_working_areas_ids": [113, 114],
                "robot_parking_areas_ids": [189],
                "gantry_parking_areas_ids": [189],
            },
        },
    },

    "num04_groovingMachineLarge": {
        "type_id": "04",
        "type_name": "num04_groovingMachineLarge",
        "registration_type": "articulation",
        "num_workstations": 1,
        "num_registration_parts": 2,
        "registeration_infos": {
            "num04_groovingMachineLarge_part01_large_fixed_base": {
                "prim_paths_expr": "/World/envs/env_{i}/obj/HC_factory/num04_groovingMachineLarge/part01_large_fixed_base",
                "joint_positions_working": [-0.2, 0.0],
                "joint_positions_reset": [0.0, 0.0],
                "animation_time": 100,
                "human_working_areas_ids": [341, 342],
                "robot_parking_areas_ids": [96],
                "gantry_parking_areas_ids": [96],
            },
            "num04_groovingMachineLarge_part02_large_mobile_base": {
                "prim_paths_expr": "/World/envs/env_{i}/obj/HC_factory/num04_groovingMachineLarge/part02_large_mobile_base",
                "joint_positions_working": [-1.0],
                "joint_positions_reset": [0.0],
                "animation_time": 100,
            },
        },
    },
    
    "num05_groovingMachineSmall": {
        "type_id": "05",
        "type_name": "num05_groovingMachineSmall",
        "registration_type": "articulation",
        "num_workstations": 1,
        "num_registration_parts": 2,
        "registeration_infos": {
            "num05_groovingMachineSmall_part01_small_fixed_base": {
                "prim_paths_expr": "/World/envs/env_{i}/obj/HC_factory/num05_groovingMachineSmall/part01_small_fixed_base",
                "joint_positions_working": [-0.3, -0.5],
                "joint_positions_reset": [0.0, 0.0],
                "animation_time": 100,
                "human_working_areas_ids": [160],
                "robot_parking_areas_ids": [139],
                "gantry_parking_areas_ids": [139],
            },
            "num05_groovingMachineSmall_part02_small_mobile_handle": {
                "prim_paths_expr": "/World/envs/env_{i}/obj/HC_factory/num05_groovingMachineSmall/part02_small_mobile_handle",
                "joint_positions_working": [-0.3, -0.5],
                "joint_positions_reset": [0.0, 0.0],
                "animation_time": 100,
            },
        },
    },

    "num06_highPressureFoamingMachine": {
        "type_id": "06",
        "type_name": "num06_highPressureFoamingMachine",
        "registration_type": "articulation",
        "num_workstations": 1,
        "num_registration_parts": 1,
        "registeration_infos": {
            "num06_highPressureFoamingMachine": {
                "prim_paths_expr": "/World/envs/env_{i}/obj/HC_factory/num06_highPressureFoamingMachine",
                "joint_positions_working": [-0.7],
                "joint_positions_reset": [0.0],
                "animation_time": 100,
                "human_working_areas_ids": [130],
                "robot_parking_areas_ids": [85],
                "gantry_parking_areas_ids": [85],
            },
        },
    },

    "num07_gantry_group": {
        "type_id": "07",
        "type_name": "num07_gantry_group",
        "registration_type": "articulation",
        "num_workstations": 4,
        "num_registration_parts": 1,
        "registeration_infos": {
            "num07_gantry_group": {
                "prim_paths_expr": "/World/envs/env_{i}/obj/HC_factory/num07_gantry_group/gantry_00",
                "joint_positions_working": [10.0, 10.0, 10.0, 10.0, 5.0, -5.0, 5.0, -5.0],
                "joint_positions_reset": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "animation_time": 100,
                "human_working_areas_ids": [],
                "robot_parking_areas_ids": [],
                "gantry_parking_areas_ids": [],
            },
        },
    },

    "num08_workbench": {
        "type_id": "08",
        "type_name": "num08_workbench",
        "registration_type": "articulation",
        "num_workstations": 2,
        "num_registration_parts": 1,
        "registeration_infos": {
            "num08_workbench": {
                "prim_paths_expr": "/World/envs/env_{i}/obj/HC_factory/num08_workbench",
                "joint_positions_working": [0.0],
                "joint_positions_reset": [0.0],
                "animation_time": 100,
                # [station_00, station_01]
                "human_working_areas_ids": [45, 49],
                "robot_parking_areas_ids": [9, 12],
                "gantry_parking_areas_ids": [9, 12],
            },
        },
    },
}


