cfg_products_process= {
    "product_water_pipe": {
        "product_id": 00,
        "related_materials": {
            "product_00_pipe": 1,
            "product_00_flange": 1,
            "product_00_elbow": 1,
            #semi是指半成品
            "product_00_semi": 1,
            #maded product是指成品
            "product_00_maded": 1,
        },
        #prim_paths_expr is the path in .usd file
        "meta_registeration_info": {
            # The asterisk (*) in the key denotes a placeholder for the product number. For example, "product_00_maded_00", "product_00_maded_01", etc., 
            # where the first "00" represents the product ID, and the second "*" represents the product number.
            "product_00_maded_*": {
                "prim_paths_expr": "/World/envs/.*/obj/HC_factory/product_water_pipe_group/products/product_*",
                "name": "product_00_maded_*",
            },
            "product_00_semi_*": {
                "prim_paths_expr": "/World/envs/.*/obj/HC_factory/product_water_pipe_group/products/semi_product_*",
                "name": "product_00_semi_*",
            },
            "product_00_pipe_*": {
                "prim_paths_expr": "/World/envs/.*/obj/HC_factory/product_water_pipe_group/cubes/cube_*",
                "name": "product_00_pipe_*",
            },
            "product_00_flange_*": {
                "prim_paths_expr": "/World/envs/.*/obj/HC_factory/product_water_pipe_group/hoops/hoop_*",
                "name": "product_00_flange_*",
            },
            "product_00_elbow_*": {
                "prim_paths_expr": "/World/envs/.*/obj/HC_factory/product_water_pipe_group/bending_tubes/bending_tube_*",
                "name": "product_00_elbow_*",
            },
        },
        "material_states": {
            "pipe": {"raw_pipe", "in_list", "conveying", "on_machine", "cutting", "cut_done", "integrated"},
            "flange": {"raw_flange", "in_list", "conveying", "integrated"},
            "elbow": {"raw_elbow", "in_list", "conveying", "integrated"},
            "product_water_pipe": {"separated", "integrated"},
        },
        "num_process_steps": 7,
        "process_steps": {
            "pipe_cutting": {
                "machine": "num03_rollerbedCNCPipeIntersectionCuttingMachine",
                "required_materials": {"pipe": 1},
                "process_time": 100,
                "gaussian_random_time": 10,
                "material_state_sequence": ["on_machine", "cutting", "cut_done"],
            },
            "pipe_grooving": {
                "machine": "num05_groovingMachineLarge",
                "required_materials": {"pipe": 1},
                "process_time": 100,
                "gaussian_random_time": 10,
                "material_state_sequence": ["on_machine", "grooving", "grooved"],
            },
            "batch_spot_welding": {
                "machine": "num09_workbench",
                "required_materials": {"pipe": 1, "flange": 1, "elbow": 1},
                "process_time": 100,
                "gaussian_random_time": 10,
                "material_state_sequence": ["on_workbench", "spot_welding_flange", "spot_welding_elbow", "spot_welded"],
            },
            # This process step refers to "Argon arc welding root" (氩弧焊底焊 in Chinese)
            "arc_welding_root": {
                "machine": "num02_weldingRobot",
                "required_materials": {"product_water_pipe": 1},
                "process_time": 100,
                "gaussian_random_time": 10,
                "material_state_sequence": ["on_machine", "welding_root", "welded_root"],
            },
            # This process step refers to "MIG welding surface" (MIG焊面焊 in Chinese), MIG full name is Metal Inert Gas welding
            "MIG_welding_surface": {
                "machine": "num01_rotaryPipeAutomaticWeldingMachine",
                "required_materials": {"product_water_pipe": 1},
                "process_time": 100,
                "gaussian_random_time": 10,
                "material_state_sequence": ["on_machine", "welding_surface", "welded_surface"],
            },
            "paint_rust_proof": {
                "machine": "num09_workbench",
                "required_materials": {"product_water_pipe": 1},
                "process_time": 100,
                "gaussian_random_time": 10,
                "material_state_sequence": ["on_machine", "painting_rust_proof", "painted_rust_proof"],
            },
            # 运存放区
            "product_to_storage": {
                "machine": "num08_gantry_group",
                "alternative_robot": "AGV",
                "required_materials": {"product_water_pipe": 1},
                "process_time": 100,
                "gaussian_random_time": 10,
                "material_state_sequence": ["on_machine", "conveying", "in_storage"],
            },
        }
    }
}

cfg_material_registration_infos = {
    "registeration_type": "rigid_prim",
    "registeration_infos": {
        "product_water_pipe": 4, #0-3
    }
}