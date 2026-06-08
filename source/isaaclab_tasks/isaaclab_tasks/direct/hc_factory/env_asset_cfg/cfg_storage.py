
import math


CfgResetStateTemplate = {
    "key_variables": {},
    # state gallery: empty, partial, full
    "state": "empty",
    "num_material": 0,
    "material_type": None,
    "material_idx_list": [],
}

###################################################################
###################################################################
# 1. functions for building storage placement config
###################################################################
###################################################################
def _build_relative_placement_cfg(example_data_from_isaac: dict) -> dict:
    """Build relative pose offsets from one storage sample.

    Later, once the runtime knows the absolute pose of a specific storage base,
    the absolute pose of each material slot can be recovered by:
    1. absolute_position = storage_base_position + relative_position
    2. absolute_orientation = storage_base_orientation composed with relative_orientation
    """
    base_pose = example_data_from_isaac["base_pose"]
    base_position = base_pose["position"]
    base_orientation = base_pose["orientation"]
    pose_list = []

    for key, pose in example_data_from_isaac.items():
        if not key.startswith("storage_pose_"):
            continue
        pose_list.append(
            {
                "placement_name": key,
                "position": [
                    pose["position"][0] - base_position[0],
                    pose["position"][1] - base_position[1],
                    pose["position"][2] - base_position[2],
                ],
                "orientation": [
                    *_quat_multiply(_quat_conjugate(base_orientation), pose["orientation"])
                ],
            }
        )

    pose_list.sort(key=lambda pose: pose["placement_name"])
    return {
        "capacity": example_data_from_isaac["capacity"],
        "data_type": "relative",
        "pose_list": pose_list,
    }


def _quat_conjugate(quaternion_wxyz: list[float]) -> list[float]:
    return [
        quaternion_wxyz[0],
        -quaternion_wxyz[1],
        -quaternion_wxyz[2],
        -quaternion_wxyz[3],
    ]


def _quat_multiply(quat_a_wxyz: list[float], quat_b_wxyz: list[float]) -> list[float]:
    w1, x1, y1, z1 = quat_a_wxyz
    w2, x2, y2, z2 = quat_b_wxyz
    return [
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ]


def _quat_rotate_vector(quaternion_wxyz: list[float], vector_xyz: list[float]) -> list[float]:
    vector_quat = [0.0, vector_xyz[0], vector_xyz[1], vector_xyz[2]]
    rotated_vector_quat = _quat_multiply(
        _quat_multiply(quaternion_wxyz, vector_quat),
        _quat_conjugate(quaternion_wxyz),
    )
    return rotated_vector_quat[1:]


def _build_vertical_placement_cfg_from_parallel(example_data_from_isaac: dict) -> dict:
    """Rotate the parallel placement around storage local z-axis, but express the delta in global axes."""
    half_angle = math.pi / 4.0
    local_rotation_quat = [math.cos(half_angle), 0.0, 0.0, math.sin(half_angle)]
    base_orientation = example_data_from_isaac["base_pose"]["orientation"]
    global_rotation_quat = _quat_multiply(
        _quat_multiply(base_orientation, local_rotation_quat),
        _quat_conjugate(base_orientation),
    )
    parallel_placement_cfg_relative = _build_relative_placement_cfg(example_data_from_isaac)
    rotated_pose_list = []

    for pose in parallel_placement_cfg_relative["pose_list"]:
        rotated_pose_list.append(
            {
                "placement_name": pose["placement_name"],
                "position": [*_quat_rotate_vector(global_rotation_quat, pose["position"])],
                "orientation": [*_quat_multiply(global_rotation_quat, pose["orientation"])],
            }
        )

    return {
        "capacity": parallel_placement_cfg_relative["capacity"],
        "data_type": "relative",
        "pose_list": rotated_pose_list,
    }


def _infer_grid_count(start_value: float, step_value: float, end_value: float) -> int:
    step_length = abs(step_value - start_value)
    total_length = abs(end_value - start_value)
    if math.isclose(step_length, 0.0):
        return 1
    return int(round(total_length / step_length)) + 1


def _build_ground_storage_placement_cfg(example_poses_from_isaac: dict) -> dict:
    """Build absolute slot poses for ground storage.

    Ground storage is configured directly in world coordinates, so downstream code
    can use pose_list entries as the final absolute poses without another transform.
    """
    start_pose = example_poses_from_isaac["start_storage_pose_00"]
    one_step_x = example_poses_from_isaac["one_step_x"]
    one_step_y = example_poses_from_isaac["one_step_y"]
    end_pose = example_poses_from_isaac["end_storage_pose_00"]

    num_columns = _infer_grid_count(
        start_pose["position"][0], one_step_x["position"][0], end_pose["position"][0]
    )
    num_rows = _infer_grid_count(
        start_pose["position"][1], one_step_y["position"][1], end_pose["position"][1]
    )

    delta_x = [
        one_step_x["position"][0] - start_pose["position"][0],
        one_step_x["position"][1] - start_pose["position"][1],
        one_step_x["position"][2] - start_pose["position"][2],
    ]
    delta_y = [
        one_step_y["position"][0] - start_pose["position"][0],
        one_step_y["position"][1] - start_pose["position"][1],
        one_step_y["position"][2] - start_pose["position"][2],
    ]

    pose_list = []
    for row_idx in range(num_rows):
        for column_idx in range(num_columns):
            pose_list.append(
                {
                    "placement_name": f"storage_pose_{row_idx:02d}_{column_idx:02d}",
                    "position": [
                        start_pose["position"][0] + column_idx * delta_x[0] + row_idx * delta_y[0],
                        start_pose["position"][1] + column_idx * delta_x[1] + row_idx * delta_y[1],
                        start_pose["position"][2] + column_idx * delta_x[2] + row_idx * delta_y[2],
                    ],
                    "orientation": start_pose["orientation"].copy(),
                }
            )

    return {
        "num_columns": num_columns,
        "num_rows": num_rows,
        "capacity": num_columns * num_rows,
        "data_type": "absolute",
        "pose_list": pose_list,
    }

###################################################################
###################################################################
# 2. example data from isaac and
# these example data are used to build the placement config for the storage in the environment
###################################################################
###################################################################

BlackStorage_parallel_example_data_from_isaac = {
    "base_pose": {
        "position": [30.68705, 1.38209, 0.53671],
        # quaternion format: [w, x, y, z]
        "orientation": [1.0, 0.0, 0.0, 0.0],
    },
    "capacity": 6,
    "placement_type": "parallel, the storage is parallel to the material by default, so their local coordinate xyz is parallel",
    "storage_pose_00": {"position": [30.76917, 2.65819, 0.56230], "orientation": [1.0, 0.0, 0.0, 0.0]},
    "storage_pose_01": {"position": [30.76917, 2.22427, 0.56230], "orientation": [1.0, 0.0, 0.0, 0.0]},
    "storage_pose_02": {"position": [30.76917, 1.78928, 0.56230], "orientation": [1.0, 0.0, 0.0, 0.0]},
    "storage_pose_03": {"position": [30.76917, 2.43246, 0.93627], "orientation": [1.0, 0.0, 0.0, 0.0]},
    "storage_pose_04": {"position": [30.76917, 2.01911, 0.93681], "orientation": [1.0, 0.0, 0.0, 0.0]},
    "storage_pose_05": {"position": [30.76917, 2.21886, 1.29129], "orientation": [1.0, 0.0, 0.0, 0.0]},
}


BlackStorage_vertical_example_data_from_isaac = BlackStorage_parallel_example_data_from_isaac.copy()
BlackStorage_vertical_example_data_from_isaac["placement_type"] = (
    "vertical: The storage orientation is vertical to the material. "
    "To align them, the material should be rotated 90 degrees around the z-axis. "
    "After rotation, the storage aligns in parallel with the material, "
    "and the rotated material's pose matches the storage_poses from BlackStorage_parallel_example_data_from_isaac."
)

YellowStorage_parallel_example_data_from_isaac = {
    "base_pose": {
        "position": [-21.31862, 14.86175, 0.5251],
        # quaternion format: [w, x, y, z]
        "orientation": [1.0, 0.0, 0.0, 0.0],
    },
    "capacity": 4,
    "storage_pose_00": {"position": [-21.31862, 15.90933, 0.56230], "orientation": [1.0, 0.0, 0.0, 0.0]},
    "storage_pose_01": {"position": [-21.31862, 15.47541, 0.56230], "orientation": [1.0, 0.0, 0.0, 0.0]},
    "storage_pose_02": {"position": [-21.31862, 15.87804, 0.93627], "orientation": [1.0, 0.0, 0.0, 0.0]},
    "storage_pose_03": {"position": [-21.31862, 15.46469, 0.93681], "orientation": [1.0, 0.0, 0.0, 0.0]},
}

YellowStorage_vertical_example_data_from_isaac = YellowStorage_parallel_example_data_from_isaac.copy()
YellowStorage_vertical_example_data_from_isaac["placement_type"] = (
    "vertical: The storage orientation is vertical to the material. "
    "To align them, the material should be rotated 90 degrees around the z-axis. "
    "After rotation, the storage aligns in parallel with the material, "
    "and the rotated material's pose matches the storage_poses from YellowStorage_parallel_example_data_from_isaac."
)

BlackStorage_parallel_placement_cfg_relative = _build_relative_placement_cfg(BlackStorage_parallel_example_data_from_isaac)
BlackStorage_vertical_placement_cfg_relative = _build_vertical_placement_cfg_from_parallel(
    BlackStorage_parallel_example_data_from_isaac
)
YellowStorage_parallel_placement_cfg_relative = _build_relative_placement_cfg(YellowStorage_parallel_example_data_from_isaac)
YellowStorage_vertical_placement_cfg_relative = _build_vertical_placement_cfg_from_parallel(
    YellowStorage_parallel_example_data_from_isaac
)

GroundStorage_example_elbow_poses_from_isaac = {
    # [x, y, z] range: [33.72357, -3.88112, 0.3] to [30.67677, -0.51596, 0.3]
    # x: [33.72357, 30.67677]
    # y: [-3.88112, -0.51596]
    # z: [0.3, 0.3]
    "start_storage_pose_00": {"position": [33.72357, -3.88112, 0.3], "orientation": [1.0, 0.0, 0.0, 0.0]},
    # one step "y" from start
    "one_step_y": {"position": [33.72357, -3.32026, 0.3], "orientation": [1.0, 0.0, 0.0, 0.0]},
    # one step "x" from start
    "one_step_x": {"position": [33.11421, -3.88112, 0.3], "orientation": [1.0, 0.0, 0.0, 0.0]},
    # end pose (might not be integer multiple of one_step_x and one_step_y)
    "end_storage_pose_00": {"position": [30.67677, -0.51596, 0.3], "orientation": [1.0, 0.0, 0.0, 0.0]},
    ### the following are inferred from the start_storage_pose_00, one_step_x, one_step_y, end_storage_pose_00
    # "per_length_x": none,
    # "per_length_y": none,
    # "start_end_x_length": none,
    # "start_end_y_length": none,
    # "num_columns": none,
    # "num_rows": none,
}

GroundStorage_example_flange_poses_from_isaac = {
    # [x, y, z] range: [19.20972, -3.50812, 0.2177] to [22.5889, -1.13456, 0.2177]
    # x: [19.20972, 22.5889]
    # y: [-3.50812, -1.13456]
    # z: [0.2177, 0.2177]
    "start_storage_pose_00": {"position": [19.20972, -3.50812, 0.2177], "orientation": [1.0, 0.0, 0.0, 0.0]},
    # one step forward in the x direction from the start pose
    "one_step_x": {"position": [19.88115, -3.45602, 0.2177], "orientation": [1.0, 0.0, 0.0, 0.0]},
    # one step forward in the y direction from the start pose
    "one_step_y": {"position": [19.18498, -2.83761, 0.2177], "orientation": [1.0, 0.0, 0.0, 0.0]},
    # end pose (might not be integer multiple of one_step_x and one_step_y)
    "end_storage_pose_00": {"position": [22.5889, -1.13456,  0.2177], "orientation": [1.0, 0.0, 0.0, 0.0]},
    ### the following are inferred from the start_storage_pose_00, one_step_x, one_step_y, end_storage_pose_00
    # "per_length_x": none,
    # "per_length_y": none,
    # "start_end_x_length": none,
    # "start_end_y_length": none,
    # "num_columns": none,
    # "num_rows": none,
}


GroundStorage_example_elbow_placement_cfg_absolute = _build_ground_storage_placement_cfg(
    GroundStorage_example_elbow_poses_from_isaac
)
GroundStorage_example_flange_placement_cfg_absolute = _build_ground_storage_placement_cfg(
    GroundStorage_example_flange_poses_from_isaac
)


###################################################################
###################################################################
# 3. cfg storage for the environment
###################################################################
###################################################################
CfgStorage = {
    "BlackStorage": {
        "num_storage": 5,
        "storage_cfg_dict": {
            "BlackStorage_00": {
                "type_id": "00",
                "type_name": "Black Storage 00 nearby num08_workbench robot_parking_areas_ids left bottom corner",
                "class_name": "BlackStorage",
                "meta_registeration_info": {
                    "prim_paths_expr": "/World/envs/env_{i}/obj/HC_factory/areas_for_material/area_black_base/base_00",
                    "name": "black_storage_00",
                },
                "capacity": 6,
                "supporting_materials": {"product_00_pipe", "product_00_semi"},
                "human_working_areas_ids": [39],
                "robot_parking_areas_ids": [38],
                #same with human working areas ids
                "gantry_parking_areas_ids": [39],
                "placement_type": "parallel, the storage is parallel to the material by default, so their local coordinate xyz is parallel",
                "placement_cfg": BlackStorage_parallel_placement_cfg_relative,
            },
            
            "BlackStorage_01": {
                "type_id": "01",
                "type_name": "Black Storage 01 nearby num07_gantry_group machine right bottom corner",
                "class_name": "BlackStorage",
                "meta_registeration_info": {
                    "prim_paths_expr": "/World/envs/env_{i}/obj/HC_factory/areas_for_material/area_black_base/base_01",
                    "name": "black_storage_01",
                }
                ,
                "capacity": 6,
                "supporting_materials": {"product_00_pipe", "product_00_semi"},
                "human_working_areas_ids": [42],
                "robot_parking_areas_ids": [41],
                "gantry_parking_areas_ids": [42],
                "placement_type": "parallel, the storage is parallel to the material by default, so their local coordinate xyz is parallel",
                "placement_cfg": BlackStorage_parallel_placement_cfg_relative,
            },

            "BlackStorage_02": {
                "type_id": "02",
                "type_name": "Black Storage 02 nearby num07_gantry_group machine right side corner",
                "class_name": "BlackStorage",
                "meta_registeration_info": {
                    "prim_paths_expr": "/World/envs/env_{i}/obj/HC_factory/areas_for_material/area_black_base/base_02",
                    "name": "black_storage_02",
                }
                ,
                "capacity": 6,
                "supporting_materials": {"product_00_pipe", "product_00_semi"},
                "human_working_areas_ids": [53],
                "robot_parking_areas_ids": [54],
                "gantry_parking_areas_ids": [53],
                "placement_type": "vertical, the storage is vertical to the material, the material should be rotated 90 degrees around the z axis",
                "placement_cfg": BlackStorage_vertical_placement_cfg_relative,
            },
            
            "BlackStorage_03": {
                "type_id": "03",
                "type_name": "Black Storage 03 nearby num04_groovingMachineLarge machine left side corner",
                "class_name": "BlackStorage",
                "meta_registeration_info": {
                    "prim_paths_expr": "/World/envs/env_{i}/obj/HC_factory/areas_for_material/area_black_base/base_03",
                    "name": "black_storage_03",
                }
                ,
                "capacity": 6,
                "supporting_materials": {"product_00_pipe", "product_00_semi"},
                "human_working_areas_ids": [221],
                "robot_parking_areas_ids": [219],
                "gantry_parking_areas_ids": [221],
                "placement_type": "vertical, the storage is vertical to the material, the material should be rotated 90 degrees around the z axis",
                "placement_cfg": BlackStorage_vertical_placement_cfg_relative,
            },
            
            "BlackStorage_04": {
                "type_id": "04",
                "type_name": "Black Storage 04 nearby num04_groovingMachineLarge machine right side corner",
                "class_name": "BlackStorage",
                "meta_registeration_info": {
                    "prim_paths_expr": "/World/envs/env_{i}/obj/HC_factory/areas_for_material/area_black_base/base_04",
                    "name": "black_storage_04",
                }
                ,
                "capacity": 6,
                "supporting_materials": {"product_00_pipe", "product_00_semi"},
                "human_working_areas_ids": [140],
                "robot_parking_areas_ids": [139],
                "gantry_parking_areas_ids": [140],
                "placement_type": "vertical, the storage is vertical to the material, the material should be rotated 90 degrees around the z axis",
                "placement_cfg": BlackStorage_vertical_placement_cfg_relative,
            },
            
            "BlackStorage_05": {
                "type_id": "05",
                "type_name": "Black Storage 05 adjacent to Black Storage_04 on the right side",   
                "class_name": "BlackStorage",
                "meta_registeration_info": {
                    "prim_paths_expr": "/World/envs/env_{i}/obj/HC_factory/areas_for_material/area_black_base/base_05",
                    "name": "black_storage_05",
                }
                ,
                "capacity": 6,
                "supporting_materials": {"product_00_pipe", "product_00_semi"},
                "human_working_areas_ids": [142],
                "robot_parking_areas_ids": [141],
                "gantry_parking_areas_ids": [142],
                "placement_type": "vertical, the storage is vertical to the material, the material should be rotated 90 degrees around the z axis",
                "placement_cfg": BlackStorage_vertical_placement_cfg_relative,
            },
        },
    },


    "YellowStorage": {
        "num_storage": 11,
        "storage_cfg_dict": {
            "YellowStorage_00": {
                "type_id": "06",
                "type_name": "Yellow Storage 00, located opposite the num04_groovingMachineLarge machine",
                "class_name": "YellowStorage",
                "meta_registeration_info": {
                    "prim_paths_expr": "/World/envs/env_{i}/obj/HC_factory/areas_for_material/area_yellow_base/yellow_base_00",
                    "name": "yellow_storage_00",
                }
                ,
                "capacity": 6,
                "supporting_materials": {"product_00_pipe", "product_00_semi", "product_00_maded"},
                "human_working_areas_ids": [116],
                "robot_parking_areas_ids": [272],
                "gantry_parking_areas_ids": [116],
                "placement_type": "vertical, the storage is vertical to the material, the material should be rotated 90 degrees around the z axis",
                "placement_cfg": YellowStorage_vertical_placement_cfg_relative,
            },

            "YellowStorage_01": {
                "type_id": "07",
                "type_name": "Yellow Storage 01 adjacent to Yellow Storage_00 on the right side, located opposite the num04_groovingMachineLarge machine",
                "class_name": "YellowStorage",
                "meta_registeration_info": {
                    "prim_paths_expr": "/World/envs/env_{i}/obj/HC_factory/areas_for_material/area_yellow_base/yellow_base_01",
                    "name": "yellow_storage_01",
                }
                ,
                "capacity": 6,
                "supporting_materials": {"product_00_pipe", "product_00_semi", "product_00_maded"},
                "human_working_areas_ids": [116],
                "robot_parking_areas_ids": [272],
                "gantry_parking_areas_ids": [116],
                "placement_type": "vertical, the storage is vertical to the material, the material should be rotated 90 degrees around the z axis",
                "placement_cfg": YellowStorage_vertical_placement_cfg_relative,
            },

            "YellowStorage_02": {
                "type_id": "08",
                "type_name": "Yellow Storage 02 adjacent to Yellow Storage_01 on the right side, located opposite the num04_groovingMachineLarge machine",
                "class_name": "YellowStorage",
                "meta_registeration_info": {
                    "prim_paths_expr": "/World/envs/env_{i}/obj/HC_factory/areas_for_material/area_yellow_base/yellow_base_02",
                    "name": "yellow_storage_02",
                }
                ,
                "capacity": 6,
                "supporting_materials": {"product_00_pipe", "product_00_semi", "product_00_maded"},
                "human_working_areas_ids": [117],
                "robot_parking_areas_ids": [272],
                "gantry_parking_areas_ids": [117],
                "placement_type": "vertical, the storage is vertical to the material, the material should be rotated 90 degrees around the z axis",
                "placement_cfg": YellowStorage_vertical_placement_cfg_relative,
            },

            "YellowStorage_03": {
                "type_id": "09",
                "type_name": "Yellow Storage 03 adjacent to Yellow Storage_02 on the right side, located opposite the num04_groovingMachineLarge machine",
                "class_name": "YellowStorage",
                "meta_registeration_info": {
                    "prim_paths_expr": "/World/envs/env_{i}/obj/HC_factory/areas_for_material/area_yellow_base/yellow_base_03",
                    "name": "yellow_storage_03",
                }
                ,
                "capacity": 6,
                "supporting_materials": {"product_00_pipe", "product_00_semi", "product_00_maded"},
                "human_working_areas_ids": [118],
                "robot_parking_areas_ids": [273],
                "gantry_parking_areas_ids": [118],
                "placement_type": "vertical, the storage is vertical to the material, the material should be rotated 90 degrees around the z axis",
                "placement_cfg": YellowStorage_vertical_placement_cfg_relative,
            },

            "YellowStorage_04": {
                "type_id": 10,
                "type_name": "Yellow Storage 04 adjacent to Yellow Storage_03 on the right side, located opposite the num04_groovingMachineLarge machine",
                "class_name": "YellowStorage",
                "meta_registeration_info": {
                    "prim_paths_expr": "/World/envs/env_{i}/obj/HC_factory/areas_for_material/area_yellow_base/yellow_base_04",
                    "name": "yellow_storage_04",
                }
                ,
                "capacity": 6,
                "supporting_materials": {"product_00_pipe", "product_00_semi", "product_00_maded"},
                "human_working_areas_ids": [118],
                "robot_parking_areas_ids": [273],
                "gantry_parking_areas_ids": [118],
                "placement_type": "vertical, the storage is vertical to the material, the material should be rotated 90 degrees around the z axis",
                "placement_cfg": YellowStorage_vertical_placement_cfg_relative,
            },
            "YellowStorage_05": {
                "type_id": 11,
                "type_name": "Yellow Storage 05 adjacent to Black Storage_05 on the right side, located opposite the semiFinishedArea_02",
                "class_name": "YellowStorage",
                "meta_registeration_info": {
                    "prim_paths_expr": "/World/envs/env_{i}/obj/HC_factory/areas_for_material/area_yellow_base/yellow_base_05",
                    "name": "yellow_storage_05",
                }
                ,
                "capacity": 6,
                "supporting_materials": {"product_00_pipe", "product_00_semi", "product_00_maded"},
                "human_working_areas_ids": [144],
                "robot_parking_areas_ids": [299],
                "gantry_parking_areas_ids": [144],
                "placement_type": "parallel, the storage is parallel to the material by default, so their local coordinate xyz is parallel",
                "placement_cfg": YellowStorage_parallel_placement_cfg_relative,
            },
            "YellowStorage_06": {
                "type_id": 12,
                "type_name": "Yellow Storage 06 adjacent to Yellow Storage_05, located opposite the semiFinishedArea_02",
                "class_name": "YellowStorage",
                "meta_registeration_info": {
                    "prim_paths_expr": "/World/envs/env_{i}/obj/HC_factory/areas_for_material/area_yellow_base/yellow_base_06",
                    "name": "yellow_storage_06",
                }
                ,
                "capacity": 6,
                "supporting_materials": {"product_00_pipe", "product_00_semi", "product_00_maded"},
                "human_working_areas_ids": [144],
                "robot_parking_areas_ids": [299],
                "gantry_parking_areas_ids": [144],
                "placement_type": "parallel, the storage is parallel to the material by default, so their local coordinate xyz is parallel",
                "placement_cfg": YellowStorage_parallel_placement_cfg_relative,
            },
            "YellowStorage_07": {
                "type_id": 13,
                "type_name": "Yellow Storage 07 adjacent to Yellow Storage_06, located opposite the semiFinishedArea_02",
                "class_name": "YellowStorage",
                "meta_registeration_info": {
                    "prim_paths_expr": "/World/envs/env_{i}/obj/HC_factory/areas_for_material/area_yellow_base/yellow_base_07",
                    "name": "yellow_storage_07",
                }
                ,
                "capacity": 6,
                "supporting_materials": {"product_00_pipe", "product_00_semi", "product_00_maded"},
                "human_working_areas_ids": [144],
                "robot_parking_areas_ids": [299],
                "gantry_parking_areas_ids": [144],
                "placement_type": "parallel, the storage is parallel to the material by default, so their local coordinate xyz is parallel",
                "placement_cfg": YellowStorage_parallel_placement_cfg_relative,
            },
            "YellowStorage_08": {
                "type_id": 14,
                "type_name": "Yellow Storage 08 adjacent to Yellow Storage_07, located opposite the semiFinishedArea_02",
                "class_name": "YellowStorage",
                "meta_registeration_info": {
                    "prim_paths_expr": "/World/envs/env_{i}/obj/HC_factory/areas_for_material/area_yellow_base/yellow_base_08",
                    "name": "yellow_storage_08",
                }
                ,
                "capacity": 6,
                "supporting_materials": {"product_00_pipe", "product_00_semi", "product_00_maded"},
                "human_working_areas_ids": [148],
                "robot_parking_areas_ids": [299],
                "gantry_parking_areas_ids": [148],
                "placement_type": "parallel, the storage is parallel to the material by default, so their local coordinate xyz is parallel",
                "placement_cfg": YellowStorage_parallel_placement_cfg_relative,
            },
            "YellowStorage_09": {
                "type_id": 15,
                "type_name": "Yellow Storage 09 adjacent to Yellow Storage_08 located opposite the semiFinishedArea_02",
                "class_name": "YellowStorage",
                "meta_registeration_info": {
                    "prim_paths_expr": "/World/envs/env_{i}/obj/HC_factory/areas_for_material/area_yellow_base/yellow_base_09",
                    "name": "yellow_storage_09",
                }
                ,
                "capacity": 6,
                "supporting_materials": {"product_00_pipe", "product_00_semi", "product_00_maded"},
                "human_working_areas_ids": [148],
                "robot_parking_areas_ids": [299],
                "gantry_parking_areas_ids": [148],
                "placement_type": "parallel, the storage is parallel to the material by default, so their local coordinate xyz is parallel",
                "placement_cfg": YellowStorage_parallel_placement_cfg_relative,
            },
            "YellowStorage_10": {
                "type_id": 16,
                "type_name": "Yellow Storage 10 adjacent to Yellow Storage_09, located opposite the semiFinishedArea_02",
                "class_name": "YellowStorage",
                "meta_registeration_info": {
                    "prim_paths_expr": "/World/envs/env_{i}/obj/HC_factory/areas_for_material/area_yellow_base/yellow_base_10",
                    "name": "yellow_storage_10",
                }
                ,
                "capacity": 6,
                "supporting_materials": {"product_00_pipe", "product_00_semi", "product_00_maded"},
                "human_working_areas_ids": [148],
                "robot_parking_areas_ids": [299],
                "gantry_parking_areas_ids": [148],
                "placement_type": "parallel, the storage is parallel to the material by default, so their local coordinate xyz is parallel",
                "placement_cfg": YellowStorage_parallel_placement_cfg_relative,
            },
        },
    },

    "GroundStorage": {
        "num_storage": 2,
        "storage_cfg_dict": {
            "GroundStorage_00": {
                "type_id": 17,
                "type_name": "Ground Storage 00 adjacent to num08_workbench machine",
                "class_name": "GroundStorage",
                "meta_registeration_info": {
                    "prim_paths_expr": "",
                    "name": "ground_storage_00",
                },
                "capacity": 100,
                "supporting_materials": {"product_00_elbow"},
                "human_working_areas_ids": [45],
                "robot_parking_areas_ids": [53],
                "gantry_parking_areas_ids": [52],
                "placement_cfg": GroundStorage_example_elbow_placement_cfg_absolute,
                "placement_type": "grid: The storage consists of a grid of placements arranged within a rectangular area.",
           
            },
            
            "GroundStorage_01": {
                "type_id": 18,
                "type_name": "Ground Storage 01 adjacent to num08_workbench machine",
                "class_name": "GroundStorage",
                "meta_registeration_info": {
                    "prim_paths_expr": "",
                    "name": "ground_storage_01",
                },
                "capacity": 100,
                "supporting_materials": {"product_00_flange"},
                "human_working_areas_ids": [49],
                "robot_parking_areas_ids": [54],
                "gantry_parking_areas_ids": [51],
                "placement_cfg": GroundStorage_example_flange_placement_cfg_absolute,
                "placement_type": "grid: The storage consists of a grid of placements arranged within a rectangular area.",
            },
        },
    },
}     


