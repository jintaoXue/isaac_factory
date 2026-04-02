product_cfg = {
    "product_water_pipe": {
        "related_materials": {
            "pipe": 1,
            "flange": 1,
            "elbow": 1,
            "product_water_pipe": 1,
        },
        "material_states": {
            "pipe": {"raw_pipe", "in_list", "conveying", "on_machine", "cutting", "cut_done", "integrated"},
            "flange": {"raw_flange", "in_list", "conveying", "integrated"},
            "elbow": {"raw_elbow", "in_list", "conveying", "integrated"},
            "product_water_pipe": {"separated", "integrated"},
        },
        "process": {
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
