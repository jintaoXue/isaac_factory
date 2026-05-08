CfgProductProcessGallery = {
    "ProductWaterPipe": {
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


CfgProcessTaskGallery = {
    "ProductWaterPipe": {
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