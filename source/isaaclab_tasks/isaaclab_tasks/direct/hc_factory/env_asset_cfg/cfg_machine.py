import torch

CfgMachine = {
    "num00_rotaryPipeAutomaticWeldingMachine": {
        "type_id": "00",
        "type_name": "num00_rotaryPipeAutomaticWeldingMachine",
        "registration_type": "articulation",
        #see cfg_process_task_gallery.py for the definition of process tasks, and the corresponding machine for each process task
        "corresponding_process_task": ["MIG_welding_surface"],
        "corresponding_logistic_task": ["logistic_for_MIG_welding_surface"],
        "num_workstations": 2,
        "num_registration_parts": 2,
        
        "reset_state": {
            ## state: free, materialReadyFor_task_name, working_task_name, waiting_processing_task
            ## for example: state: ["working_MIG_welding_surface", "materialReadyFor_MIG_welding_surface"]
            "state": ["free", "free"],
            "processing_time_step": [0, 0],
            "target_joints_position": [None, None],
            "ongoing_task_record_index": [None, None],
            "key_variables": {},
        },
        # 只有“工位/工作站（station）”这种可以处理物料的节点，才需要配置 working area ids。
        # human_working_areas_ids 的编号来源：map_data/map_with_points_human.png（图中蓝色编号）
        # agv/gantry 的停车区编号来源：map_data/map_with_points_robot.png（图中蓝色编号）
        "working_area_ids": {
            "num00_rotaryPipeAutomaticWeldingMachine_part_01_station": {
                "human_working_areas_ids": [56],
                "robot_parking_areas_ids": [57],
                "gantry_parking_areas_ids": [56],
            },
            "num00_rotaryPipeAutomaticWeldingMachine_part_02_station": {
                "human_working_areas_ids": [60],
                "robot_parking_areas_ids": [61],
                "gantry_parking_areas_ids": [60],
            },
        },
        "material_placement_cfg": {
            "num00_rotaryPipeAutomaticWeldingMachine_part_01_station": {
                "position": torch.tensor([43.12303, 15.73721, 1.146]),
                "orientation": torch.tensor([0.0, 0.0, 0.0, 1.0]),
            },
            "num00_rotaryPipeAutomaticWeldingMachine_part_02_station": {
                "position": torch.tensor([35.14321, 15.73721, 1.146]),
                "orientation": torch.tensor([0.0, 0.0, 0.0, 1.0]),
            },
        },
        "registration_infos": {
            "num00_rotaryPipeAutomaticWeldingMachine_part_01_station": {
                #prim_paths_expr is the path in .usd file
                "prim_paths_expr": "/World/envs/env_{i}/obj/HC_factory/num00_rotaryPipeAutomaticWeldingMachine/part_01_station/track_for_mobile_base",
                "joint_positions_working": torch.tensor([0.0, 2.0]),
                "joint_positions_reset": torch.tensor([0.0, 0.0]),
                "animation_time": 20,
            },
            "num00_rotaryPipeAutomaticWeldingMachine_part_02_station": {
                "prim_paths_expr": "/World/envs/env_{i}/obj/HC_factory/num00_rotaryPipeAutomaticWeldingMachine/part_02_station/track_for_mobile_base_001",
                "joint_positions_working": torch.tensor([0.0, 0.5]),
                "joint_positions_reset": torch.tensor([0.0, 0.0]),
                "animation_time": 50,
            },
        },
    },
    "num01_weldingRobot": {
        "type_id": "01",
        "type_name": "num01_weldingRobot",
        "registration_type": "articulation",
        "corresponding_process_task": ["arc_welding_root"],
        "corresponding_logistic_task": ["logistic_for_arc_welding_root"],
        "num_workstations": 1,
        "num_registration_parts": 2,
        
        "reset_state": {
            "state": ["free"],
            "processing_time_step": [0],
            "target_joints_position": [None],
            "ongoing_task_record_index": [None],
            "key_variables": {},
        },
        "working_area_ids": {
            "num01_weldingRobot_part02_robot_arm_and_base": {
                "human_working_areas_ids": [66],
                "robot_parking_areas_ids": [65],
                "gantry_parking_areas_ids": [66],
            },
        },
        "material_placement_cfg": {
            "num01_weldingRobot_part02_robot_arm_and_base": {
                "position": torch.tensor([24.68707, 14.36929, 1.146]),
                "orientation": torch.tensor([0.0, 0.0, 0.0, 1.0]),
            },
        },
        "registration_infos": {
            "num01_weldingRobot_part02_robot_arm_and_base": {
                "prim_paths_expr": "/World/envs/env_{i}/obj/HC_factory/num01_weldingRobot/part02_robot_arm_and_base",
                "joint_positions_working": torch.tensor([3.2, -1.5, -0.3, 0.1, 0.2, 0.0]),
                "joint_positions_reset": torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                "animation_time": 20,
            },
            "num01_weldingRobot_part04_mobile_base_for_material": {
                "prim_paths_expr": "/World/envs/env_{i}/obj/HC_factory/num01_weldingRobot/part04_mobile_base_for_material",
                "joint_positions_working": torch.tensor([-2.0]),
                "joint_positions_reset": torch.tensor([0.0]),
                "animation_time": 20,
            },
        },
    },

    "num02_rollerbedCNCPipeIntersectionCuttingMachine": {
        "type_id": "02",
        "type_name": "num02_rollerbedCNCPipeIntersectionCuttingMachine",
        "registration_type": "articulation",
        "corresponding_process_task": ["pipe_cutting"],
        "corresponding_logistic_task": ["logistic_for_pipe_cutting"],
        "num_workstations": 1,
        "num_registration_parts": 2,
        
        "reset_state": {
            "state": ["free"],
            "processing_time_step": [0],
            "target_joints_position": [None],
            "ongoing_task_record_index": [None],
            "key_variables": {},
        },
        "working_area_ids": {
            "num02_rollerbedCNCPipeIntersectionCuttingMachine_part01_station": {
                "human_working_areas_ids": [90],
                "robot_parking_areas_ids": [78],
                "gantry_parking_areas_ids": [78],
            },
        },
        "material_placement_cfg": {
            "num02_rollerbedCNCPipeIntersectionCuttingMachine_part01_station": {
                "position": torch.tensor([10.06736, 16.86987, 1.146]),
                "orientation": torch.tensor([0.0, 0.0, 0.0, 1.0]),
            },
        },
        "registration_infos": {
            "num02_rollerbedCNCPipeIntersectionCuttingMachine_part01_station": {
                "prim_paths_expr": "/World/envs/env_{i}/obj/HC_factory/num02_rollerbedCNCPipeIntersectionCuttingMachine/part01_station",
                "joint_positions_working": torch.tensor([1.0]),
                "joint_positions_reset": torch.tensor([0.0]),
                "animation_time": 20,
            },
            "num02_rollerbedCNCPipeIntersectionCuttingMachine_part05_cutting_machine": {
                "prim_paths_expr": "/World/envs/env_{i}/obj/HC_factory/num02_rollerbedCNCPipeIntersectionCuttingMachine/part05_cutting_machine",
                "joint_positions_working": torch.tensor([-2.0, 0.3, 0.5]),
                "joint_positions_reset": torch.tensor([0.0, 0.0, 0.0]),
                "animation_time": 20,
            },
        },
    },

    "num03_laserCuttingMachine": {
        "type_id": "03",
        "type_name": "num03_laserCuttingMachine",
        "registration_type": "articulation",
        "corresponding_process_task": ["none"],
        "corresponding_logistic_task": ["none"],
        "num_workstations": 1,
        "num_registration_parts": 1,
        
        "reset_state": {
            "state": ["free"],
            "processing_time_step": [0],
            "target_joints_position": [None],
            "ongoing_task_record_index": [None],
            "key_variables": {},
        },
        "working_area_ids": {
            "num03_laserCuttingMachine": {
                "human_working_areas_ids": [113],
                "robot_parking_areas_ids": [111],
                "gantry_parking_areas_ids": [111],
            },
        },
        "material_placement_cfg": {
            "num03_laserCuttingMachine": {
                "position": torch.tensor([0.0, 0.0, 0.0]),
                "orientation": torch.tensor([0.0, 0.0, 0.0, 1.0]),
            },
        },
        "registration_infos": {
            "num03_laserCuttingMachine": {
                "prim_paths_expr": "/World/envs/env_{i}/obj/HC_factory/num03_laserCuttingMachine",
                "joint_positions_working": torch.tensor([-3.5]),
                "joint_positions_reset": torch.tensor([0.0]),
                "animation_time": 20,
            },
        },
    },

    "num04_groovingMachineLarge": {
        "type_id": "04",
        "type_name": "num04_groovingMachineLarge",
        "registration_type": "articulation",
        "corresponding_process_task": ["pipe_grooving"],
        "corresponding_logistic_task": ["logistic_for_pipe_grooving"],
        "num_workstations": 1,
        "num_registration_parts": 2,
        
        "reset_state": {
            "state": ["free"],
            "processing_time_step": [0],
            "target_joints_position": [None],
            "ongoing_task_record_index": [None],
            "key_variables": {},
        },
        "working_area_ids": {
            "num04_groovingMachineLarge_part01_large_fixed_base": {
                "human_working_areas_ids": [138],
                "robot_parking_areas_ids": [136],
                "gantry_parking_areas_ids": [136],
            },
        },
        "material_placement_cfg": {
            "num04_groovingMachineLarge_part01_large_fixed_base": {
                "position": torch.tensor([-7.64795, 15.33569, 1.45811]),
                "orientation": torch.tensor([0.7071, 0, 0, -0.7071]),
            },
        },
        "registration_infos": {
            "num04_groovingMachineLarge_part01_large_fixed_base": {
                "prim_paths_expr": "/World/envs/env_{i}/obj/HC_factory/num04_groovingMachineLarge/part01_large_fixed_base",
                "joint_positions_working": torch.tensor([-0.2, 0.0]),
                "joint_positions_reset": torch.tensor([0.0, 0.0]),
                "animation_time": 20,
            },
            "num04_groovingMachineLarge_part02_large_mobile_base": {
                "prim_paths_expr": "/World/envs/env_{i}/obj/HC_factory/num04_groovingMachineLarge/part02_large_mobile_base",
                "joint_positions_working": torch.tensor([-1.0]),
                "joint_positions_reset": torch.tensor([0.0]),
                "animation_time": 20,
            },
        },
    },
    
    "num05_groovingMachineSmall": {
        "type_id": "05",
        "type_name": "num05_groovingMachineSmall",
        "registration_type": "articulation",
        "corresponding_process_task": ["none"],
        "corresponding_logistic_task": ["none"],
        "num_workstations": 1,
        "num_registration_parts": 2,
        
        "reset_state": {
            "state": ["free"],
            "processing_time_step": [0],
            "ongoing_task_record_index": [None],
            "key_variables": {},
            "target_joints_position": [None],
        },
        "working_area_ids": {
            "num05_groovingMachineSmall_part01_small_fixed_base": {
                "human_working_areas_ids": [160],
                "robot_parking_areas_ids": [139],
                "gantry_parking_areas_ids": [139],
            },
        },
        "material_placement_cfg": {
            "num05_groovingMachineSmall_part01_small_fixed_base": {
                "position": torch.tensor([0.0, 0.0, 0.0]),
                "orientation": torch.tensor([0.0, 0.0, 0.0, 1.0]),
            },
        },
        "registration_infos": {
            "num05_groovingMachineSmall_part01_small_fixed_base": {
                "prim_paths_expr": "/World/envs/env_{i}/obj/HC_factory/num05_groovingMachineSmall/part01_small_fixed_base",
                "joint_positions_working": torch.tensor([-0.3, -0.5]),
                "joint_positions_reset": torch.tensor([0.0, 0.0]),
                "animation_time": 20,
            },
            "num05_groovingMachineSmall_part02_small_mobile_handle": {
                "prim_paths_expr": "/World/envs/env_{i}/obj/HC_factory/num05_groovingMachineSmall/part02_small_mobile_handle",
                "joint_positions_working": torch.tensor([-0.3, -0.5]),
                "joint_positions_reset": torch.tensor([0.0, 0.0]),
                "animation_time": 20,
            },
        },
    },

    "num06_highPressureFoamingMachine": {
        "type_id": "06",
        "type_name": "num06_highPressureFoamingMachine",
        "registration_type": "articulation",
        "corresponding_process_task": ["none"],
        "corresponding_logistic_task": ["none"],
        "num_workstations": 1,
        "num_registration_parts": 1,
        
        "reset_state": {
            "state": ["free"],
            "processing_time_step": [0],
            "ongoing_task_record_index": [None],
            "key_variables": {},
            "target_joints_position": [None],
        },
        "working_area_ids": {
            "num06_highPressureFoamingMachine": {
                "human_working_areas_ids": [130],
                "robot_parking_areas_ids": [131],
                "gantry_parking_areas_ids": [131],
            },
        },
        "material_placement_cfg": {
            "num06_highPressureFoamingMachine": {
                "position": torch.tensor([0.0, 0.0, 0.0]),
                "orientation": torch.tensor([0.0, 0.0, 0.0, 1.0]),
            },
        },
        "registration_infos": {
            "num06_highPressureFoamingMachine": {
                "prim_paths_expr": "/World/envs/env_{i}/obj/HC_factory/num06_highPressureFoamingMachine",
                "joint_positions_working": torch.tensor([-0.7]),
                "joint_positions_reset": torch.tensor([0.0]),
                "animation_time": 20,
            },
        },
    },

    "num07_gantry_group": {
        "type_id": "07",
        "type_name": "num07_gantry_group",
        "registration_type": "articulation",
        # "corresponding_process_task": [
        #     "pipe_cutting",
        #     "pipe_grooving",
        #     "batch_spot_welding",
        #     "arc_welding_root",
        #     "MIG_welding_surface",
        #     "paint_rust_proof",
        # ],
        "corresponding_logistic_task": [
            "logistic_for_pipe_cutting",
            "logistic_for_pipe_grooving",
            "logistic_for_batch_spot_welding",
            "logistic_for_arc_welding_root",
            "logistic_for_MIG_welding_surface",
            "logistic_for_paint_rust_proof",
        ],
        "num_workstations": 4,
        "num_registration_parts": 1,
        "state_gallery": {0: "free", 1: "moving_to_working", 2: "working", 3: "resetting", 4: "invalid"},
        "reset_state": {
            "state": ["free", "invalid", "invalid", "invalid"],
            "ongoing_task_record_index": [None, None, None, None],
            "key_variables": {},
            "target_area_id": None,
            "target_area_xy": None,
            "target_joints_position": None,
        },
        "working_area_ids": {
            "num07_gantry_group_station_00": {
                "human_working_areas_ids": [],
                "robot_parking_areas_ids": [],
                "gantry_parking_areas_ids": [],
            },
            "num07_gantry_group_station_01": {
                "human_working_areas_ids": [],
                "robot_parking_areas_ids": [],
                "gantry_parking_areas_ids": [],
            },
            "num07_gantry_group_station_02": {
                "human_working_areas_ids": [],
                "robot_parking_areas_ids": [],
                "gantry_parking_areas_ids": [],
            },
            "num07_gantry_group_station_03": {
                "human_working_areas_ids": [],
                "robot_parking_areas_ids": [],
                "gantry_parking_areas_ids": [],
            },
        },
        "material_placement_cfg": {},
        "registration_infos": {
            "num07_gantry_group": {
                "prim_paths_expr": "/World/envs/env_{i}/obj/HC_factory/num07_gantry_group/gantry_00",
                "joint_positions_working": torch.tensor([10.0, 10.0, 10.0, 10.0, 5.0, -5.0, 5.0, -5.0]),
                "joint_positions_reset": torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                ####the articulation pose corresponding pose, and for calculating the offset
                "xy_position_reset": torch.tensor([36.74226, 16.74226, -11.35883, -30.35883, 10.18675, 10.18675, 10.18675, 10.18675]),
                # Assume joint_position has 8 elements: [x0, x1, x2, x3, y0, y1, y2, y3] for 4 subgantrys
                "gantry_indexs": torch.tensor([0, 1, 2, 3, 0, 1, 2, 3]),
                ### to simplify the problem, we set the z of hook to be fixed, and only control the x and y movement of the gantry, so the joint position of z axis is not used for calculating the reward and is not included in the observation, but it is still needed for animation and calculating the offset between the hook and the material when gripping.
                "fixed_hook_height": 8.90808,
                "animation_time": 20,
            },
        },
    },

    "num08_workbench": {
        "type_id": "08",
        "type_name": "num08_workbench",
        "registration_type": "articulation",
        "corresponding_process_task": ["batch_spot_welding", "paint_rust_proof"],
        "corresponding_logistic_task": [
            "logistic_for_batch_spot_welding",
            "logistic_for_paint_rust_proof",
        ],
        "num_workstations": 2,
        "num_registration_parts": 1,
        
        "reset_state": {
            "state": ["free", "free"],
            "processing_time_step": [0, 0],
            "ongoing_task_record_index": [None, None],
            "key_variables": {},
        },
        "working_area_ids": {
            "num08_workbench_station_00": {
                "human_working_areas_ids": [45],
                "robot_parking_areas_ids": [40],
                "gantry_parking_areas_ids": [47],
            },
            "num08_workbench_station_01": {
                "human_working_areas_ids": [49],
                "robot_parking_areas_ids": [41],
                "gantry_parking_areas_ids": [47],
            },
        },
        "material_placement_cfg": {
            "num08_workbench_station_00": {
                "position": torch.tensor([28.44292, -2.08669, 0.53412]), 
                "orientation": torch.tensor([0.0, 0.0, 0.0, 1.0])
            },
            "num08_workbench_station_01": {
                "position": torch.tensor([25.16699, -3.05047, 0.53412]),
                "orientation": torch.tensor([0.0, 0.0, 0.0, 1.0]),
            },
        },
        # The num08_workbench is actually a manual workbench, so the joint positions are not important.
        # We simply set them to the reset positions to maintain a consistent format.
        "registration_infos": {
            "num08_workbench": {
                "prim_paths_expr": "/World/envs/env_{i}/obj/HC_factory/num08_workbench",
                "joint_positions_working": torch.tensor([0.0]),
                "joint_positions_reset": torch.tensor([0.0]),
                "animation_time": 20,
            },
        },
    },
}


