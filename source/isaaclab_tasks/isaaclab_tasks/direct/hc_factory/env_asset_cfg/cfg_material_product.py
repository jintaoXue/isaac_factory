
CfgProductProcess= {
    "ProductWaterPipe": {
        "type_id": "00",
        "type_name": "ProductWaterPipe",
        "state_gallery": {
            "product_00_pipe": {"raw_pipe", "in_list", "conveying", "on_machine", "cutting", "cut_done", "integrated"},
            "product_00_flange": {"raw_flange", "in_list", "conveying", "integrated"},
            "product_00_elbow": {"raw_elbow", "in_list", "conveying", "integrated"},
            "product_00_semi": {"separated", "integrated"},
            "product_00_maded": {"separated", "integrated"},
        },
        "reset_state": {
            "state": [0, 0, 0, 0, 0],
            "current_pose": [None, None, None, None, None, None, None],
            "target_pose": [None, None, None, None, None, None, None],
        },
        "meta_registeration_info": {
            # The asterisk (*) in the key denotes a placeholder for the product number. For example, "product_00_maded_00", "product_00_maded_01", etc., 
            # where the first "00" represents the product ID, and the second "*" represents the product number.
            #prim_paths_expr is the path in .usd file
            "product_00_pipe": {
                "prim_paths_expr": "/World/envs/env_{i}/obj/HC_factory/product_water_pipe_group/cubes/cube_{idx}",
                "name": "product_00_pipe_{idx}",
                "requried_number": 1,
            },
            "product_00_flange": {
                "prim_paths_expr": "/World/envs/env_{i}/obj/HC_factory/product_water_pipe_group/hoops/hoop_{idx}",
                "name": "product_00_flange_{idx}",
                "requried_number": 1,
            },
            "product_00_elbow": {
                "prim_paths_expr": "/World/envs/env_{i}/obj/HC_factory/product_water_pipe_group/bending_tubes/bending_tube_{idx}",
                "name": "product_00_elbow_{idx}",
                "requried_number": 1,
            },
            "product_00_semi": {
                "prim_paths_expr": "/World/envs/env_{i}/obj/HC_factory/product_water_pipe_group/products/semi_product_{idx}",
                "name": "product_00_semi_{idx}",
                "requried_number": 1,
            },
            "product_00_maded": {
                "prim_paths_expr": "/World/envs/env_{i}/obj/HC_factory/product_water_pipe_group/products/product_{idx}",
                "name": "product_00_maded_{idx}",
                "requried_number": 1,
            },
        },
        "num_process_steps": 7,
        "process_steps": {
            "pipe_cutting": {
                "machine": "num02_rollerbedCNCPipeIntersectionCuttingMachine",
                "required_materials": {"pipe": 1},
                "process_time": 100,
                "gaussian_random_time": 10,
                "material_state_sequence": ["on_machine", "cutting", "cut_done"],
            },
            "pipe_grooving": {
                "machine": "num04_groovingMachineLarge",
                "required_materials": {"pipe": 1},
                "process_time": 100,
                "gaussian_random_time": 10,
                "material_state_sequence": ["on_machine", "grooving", "grooved"],
            },
            "batch_spot_welding": {
                "machine": "num08_workbench",
                "required_materials": {"pipe": 1, "flange": 1, "elbow": 1},
                "process_time": 100,
                "gaussian_random_time": 10,
                "material_state_sequence": ["on_workbench", "spot_welding_flange", "spot_welding_elbow", "spot_welded"],
            },
            # This process step refers to "Argon arc welding root" (氩弧焊底焊 in Chinese)
            "arc_welding_root": {
                "machine": "num01_weldingRobot",
                "required_materials": {"product_water_pipe": 1},
                "process_time": 100,
                "gaussian_random_time": 10,
                "material_state_sequence": ["on_machine", "welding_root", "welded_root"],
            },
            # This process step refers to "MIG welding surface" (MIG焊面焊 in Chinese), MIG full name is Metal Inert Gas welding
            "MIG_welding_surface": {
                "machine": "num00_rotaryPipeAutomaticWeldingMachine",
                "required_materials": {"product_water_pipe": 1},
                "process_time": 100,
                "gaussian_random_time": 10,
                "material_state_sequence": ["on_machine", "welding_surface", "welded_surface"],
            },
            "paint_rust_proof": {
                "machine": "num08_workbench",
                "required_materials": {"product_water_pipe": 1},
                "process_time": 100,
                "gaussian_random_time": 10,
                "material_state_sequence": ["on_machine", "painting_rust_proof", "painted_rust_proof"],
            },
            # 运存放区
            "product_to_storage": {
                "machine": "num07_gantry_group",
                "alternative_robot": "AGV",
                "required_materials": {"product_water_pipe": 1},
                "process_time": 100,
                "gaussian_random_time": 10,
                "material_state_sequence": ["on_machine", "conveying", "in_storage"],
            },
        }
    }
}

CfgRegistrationInfos = {
    
    "ProductWaterPipe": 5, #idx: 00-04
    
}