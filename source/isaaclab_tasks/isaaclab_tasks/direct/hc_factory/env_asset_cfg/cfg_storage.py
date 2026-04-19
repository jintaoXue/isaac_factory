

CfgCommonState = {
    "state_gallery": {
        0: "empty",
        1: "partial",
        2: "full",
    },
    "reset_state": {
        "state": [0],
        "current_pose": None,
        "material_type": None,
        "material_idx": [],
    },
}

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
                "supporting_materials": ["product_00_pipe", "product_00_semi", "product_00_maded"],
                "human_working_areas_ids": [39],
                "robot_parking_areas_ids": [39],
                #same with human working areas ids
                "gantry_parking_areas_ids": [39],
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
                "supporting_materials": ["product_00_pipe", "product_00_semi", "product_00_maded"],
                "human_working_areas_ids": [42],
                "robot_parking_areas_ids": [42],
                "gantry_parking_areas_ids": [42],
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
                "supporting_materials": ["product_00_pipe", "product_00_semi", "product_00_maded"],
                "human_working_areas_ids": [53],
                "robot_parking_areas_ids": [54],
                "gantry_parking_areas_ids": [53],
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
                "supporting_materials": ["product_00_pipe", "product_00_semi", "product_00_maded"],
                "human_working_areas_ids": [221],
                "robot_parking_areas_ids": [83],
                "gantry_parking_areas_ids": [221],
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
                "supporting_materials": ["product_00_pipe", "product_00_semi", "product_00_maded"],
                "human_working_areas_ids": [140],
                "robot_parking_areas_ids": [9],
                "gantry_parking_areas_ids": [140],
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
                "supporting_materials": ["product_00_pipe", "product_00_semi", "product_00_maded"],
                "human_working_areas_ids": [142],
                "robot_parking_areas_ids": [9],
                "gantry_parking_areas_ids": [142],
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
                    "prim_paths_expr": "/World/envs/env_{i}/obj/HC_factory/areas_for_material/area_black_base/yellow_base_00",
                    "name": "yellow_storage_00",
                }
                ,
                "capacity": 6,
                "supporting_materials": ["product_00_pipe", "product_00_semi", "product_00_maded"],
                "human_working_areas_ids": [116],
                "robot_parking_areas_ids": [9],
                "gantry_parking_areas_ids": [116],
            },

            "YellowStorage_01": {
                "type_id": "07",
                "type_name": "Yellow Storage 01 adjacent to Yellow Storage_00 on the right side, located opposite the num04_groovingMachineLarge machine",
                "class_name": "YellowStorage",
                "meta_registeration_info": {
                    "prim_paths_expr": "/World/envs/env_{i}/obj/HC_factory/areas_for_material/area_black_base/yellow_base_01",
                    "name": "yellow_storage_01",
                }
                ,
                "capacity": 6,
                "supporting_materials": ["product_00_pipe", "product_00_semi", "product_00_maded"],
                "human_working_areas_ids": [116],
                "robot_parking_areas_ids": [9],
                "gantry_parking_areas_ids": [116],
            },

            "YellowStorage_02": {
                "type_id": "08",
                "type_name": "Yellow Storage 02 adjacent to Yellow Storage_01 on the right side, located opposite the num04_groovingMachineLarge machine",
                "class_name": "YellowStorage",
                "meta_registeration_info": {
                    "prim_paths_expr": "/World/envs/env_{i}/obj/HC_factory/areas_for_material/area_black_base/yellow_base_02",
                    "name": "yellow_storage_02",
                }
                ,
                "capacity": 6,
                "supporting_materials": ["product_00_pipe", "product_00_semi", "product_00_maded"],
                "human_working_areas_ids": [117],
                "robot_parking_areas_ids": [9],
                "gantry_parking_areas_ids": [117],
            },

            "YellowStorage_03": {
                "type_id": "09",
                "type_name": "Yellow Storage 03 adjacent to Yellow Storage_02 on the right side, located opposite the num04_groovingMachineLarge machine",
                "class_name": "YellowStorage",
                "meta_registeration_info": {
                    "prim_paths_expr": "/World/envs/env_{i}/obj/HC_factory/areas_for_material/area_black_base/yellow_base_03",
                    "name": "yellow_storage_03",
                }
                ,
                "capacity": 6,
                "supporting_materials": ["product_00_pipe", "product_00_semi", "product_00_maded"],
                "human_working_areas_ids": [118],
                "robot_parking_areas_ids": [9],
                "gantry_parking_areas_ids": [118],
            },

            "YellowStorage_04": {
                "type_id": 10,
                "type_name": "Yellow Storage 04 adjacent to Yellow Storage_03 on the right side, located opposite the num04_groovingMachineLarge machine",
                "class_name": "YellowStorage",
                "meta_registeration_info": {
                    "prim_paths_expr": "/World/envs/env_{i}/obj/HC_factory/areas_for_material/area_black_base/yellow_base_04",
                    "name": "yellow_storage_04",
                }
                ,
                "capacity": 6,
                "supporting_materials": ["product_00_pipe", "product_00_semi", "product_00_maded"],
                "human_working_areas_ids": [118],
                "robot_parking_areas_ids": [9],
                "gantry_parking_areas_ids": [118],
            },
            "YellowStorage_05": {
                "type_id": 11,
                "type_name": "Yellow Storage 05 adjacent to Black Storage_05 on the right side, located opposite the semiFinishedArea_02",
                "class_name": "YellowStorage",
                "meta_registeration_info": {
                    "prim_paths_expr": "/World/envs/env_{i}/obj/HC_factory/areas_for_material/area_black_base/yellow_base_05",
                    "name": "yellow_storage_05",
                }
                ,
                "capacity": 6,
                "supporting_materials": ["product_00_pipe", "product_00_semi", "product_00_maded"],
                "human_working_areas_ids": [144],
                "robot_parking_areas_ids": [9],
                "gantry_parking_areas_ids": [144],
            },
            "YellowStorage_06": {
                "type_id": 12,
                "type_name": "Yellow Storage 06 adjacent to Yellow Storage_05, located opposite the semiFinishedArea_02",
                "class_name": "YellowStorage",
                "meta_registeration_info": {
                    "prim_paths_expr": "/World/envs/env_{i}/obj/HC_factory/areas_for_material/area_black_base/yellow_base_06",
                    "name": "yellow_storage_06",
                }
                ,
                "capacity": 6,
                "supporting_materials": ["product_00_pipe", "product_00_semi", "product_00_maded"],
                "human_working_areas_ids": [144],
                "robot_parking_areas_ids": [9],
                "gantry_parking_areas_ids": [144],
            },
            "YellowStorage_07": {
                "type_id": 13,
                "type_name": "Yellow Storage 07 adjacent to Yellow Storage_06, located opposite the semiFinishedArea_02",
                "class_name": "YellowStorage",
                "meta_registeration_info": {
                    "prim_paths_expr": "/World/envs/env_{i}/obj/HC_factory/areas_for_material/area_black_base/yellow_base_07",
                    "name": "yellow_storage_07",
                }
                ,
                "capacity": 6,
                "supporting_materials": ["product_00_pipe", "product_00_semi", "product_00_maded"],
                "human_working_areas_ids": [144],
                "robot_parking_areas_ids": [9],
                "gantry_parking_areas_ids": [144],
            },
            "YellowStorage_08": {
                "type_id": 14,
                "type_name": "Yellow Storage 08 adjacent to Yellow Storage_07, located opposite the semiFinishedArea_02",
                "class_name": "YellowStorage",
                "meta_registeration_info": {
                    "prim_paths_expr": "/World/envs/env_{i}/obj/HC_factory/areas_for_material/area_black_base/yellow_base_08",
                    "name": "yellow_storage_08",
                }
                ,
                "capacity": 6,
                "supporting_materials": ["product_00_pipe", "product_00_semi", "product_00_maded"],
                "human_working_areas_ids": [148],
                "robot_parking_areas_ids": [9],
                "gantry_parking_areas_ids": [148],
            },
            "YellowStorage_09": {
                "type_id": 15,
                "type_name": "Yellow Storage 09 adjacent to Yellow Storage_08 located opposite the semiFinishedArea_02",
                "class_name": "YellowStorage",
                "meta_registeration_info": {
                    "prim_paths_expr": "/World/envs/env_{i}/obj/HC_factory/areas_for_material/area_black_base/yellow_base_09",
                    "name": "yellow_storage_09",
                }
                ,
                "capacity": 6,
                "supporting_materials": ["product_00_pipe", "product_00_semi", "product_00_maded"],
                "human_working_areas_ids": [148],
                "robot_parking_areas_ids": [9],
                "gantry_parking_areas_ids": [148],
            },
            "YellowStorage_10": {
                "type_id": 16,
                "type_name": "Yellow Storage 10 adjacent to Yellow Storage_09, located opposite the semiFinishedArea_02",
                "class_name": "YellowStorage",
                "meta_registeration_info": {
                    "prim_paths_expr": "/World/envs/env_{i}/obj/HC_factory/areas_for_material/area_black_base/yellow_base_10",
                    "name": "yellow_storage_10",
                }
                ,
                "capacity": 6,
                "supporting_materials": ["product_00_pipe", "product_00_semi", "product_00_maded"],
                "human_working_areas_ids": [148],
                "robot_parking_areas_ids": [9],
                "gantry_parking_areas_ids": [148],
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
                "capacity": 20,
                "supporting_materials": ["product_00_elbow", "product_00_flange"],
                "human_working_areas_ids": [45],
                "robot_parking_areas_ids": [],
                "gantry_parking_areas_ids": [],
            },
            
            "GroundStorage_01": {
                "type_id": 18,
                "type_name": "Ground Storage 01 adjacent to num08_workbench machine",
                "class_name": "GroundStorage",
                "meta_registeration_info": {
                    "prim_paths_expr": "",
                    "name": "ground_storage_01",
                },
                "capacity": 20,
                "supporting_materials": ["product_00_elbow", "product_00_flange"],
                "human_working_areas_ids": [49],
                "robot_parking_areas_ids": [],
                "gantry_parking_areas_ids": [],
            },
        },
    },
}     

