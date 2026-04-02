product_cfg = {
    "water_pipe": {
        "required_materials": {
            "pipe": 1,
            "flange": 1,
            "elbow": 1,
        },
        "material_states": {
            "pipe": {"raw_pipe", "in_list", "conveying", "on_machine","cutting", "cut_done"},
            "flange": {"raw_flange", "in_list", "conveying",},
            "elbow": {"raw_elbow", "in_list", "conveying",},
        },
        "process": {
            "pipe_cutting": {
                "machine": "num03_rollerbedCNCPipeIntersectionCuttingMachine",
                "required_material": "pipe",
                "process_time": 100,
                "gaussian_random_time": 10,
                "material_state_sequence": ["on_machine", "cutting", "cut_done"],
            },
        }
    }
}
