

CfgRegistrationInfos = {
    
    "ProductWaterPipe": 5, #idx: 00-04
    
}

CfgProductOrder = {
    # The production order is a map of product type to requested quantity.
    "ProductWaterPipe": 5, # idx: 00-04
}

for product_type, quantity in CfgProductOrder.items():
    if product_type not in CfgRegistrationInfos:
        raise ValueError(f"Unknown product type in CfgProductOrder: {product_type}")
    if quantity > CfgRegistrationInfos[product_type]:
        raise ValueError(
            f"Requested quantity for {product_type} ({quantity}) exceeds registered max "
            f"({CfgRegistrationInfos[product_type]})"
        )

CfgProductProcess = {
    "ProductWaterPipe": {
        "type_id": "00",
        "type_name": "ProductWaterPipe",
        "state_gallery": {
            "product_00_pipe": {"raw_pipe", "cutting", "cut_done", "integrated", "disappeared"},
            "product_00_flange": {"raw_flange", "integrated", "disappeared"},
            "product_00_elbow": {"raw_elbow", "integrated", "disappeared"},
            "product_00_semi": {"unappeared", "appeared"},
            "product_00_maded": {"unappeared", "appeared"},
            "logistic_state": {"unappeared", "in_storage", "conveying", "on_machine", "on_workbench", "on_storage"},
        },
        "reset_state_template": {
            "key_variables": {},
            "finished_task": "none",
            "storage_name": "none",
            "ongoing_task_record_index": None,
            "substate": {
                # task_step tracks the current production stage, indexed by CfgProcessTaskGalleryInAll (see cfg_process_task_gallery.py)
                "product_00_pipe": {
                    "state": "raw_pipe",
                    "logistic_state": "in_storage",
                },
                "product_00_flange": {
                    "state": "raw_flange",
                    "logistic_state": "in_storage",
                },
                "product_00_elbow": {
                    "state": "raw_elbow",
                    "logistic_state": "in_storage",
                },
                "product_00_semi": {
                    "state": "unappeared",
                    "logistic_state": "unappeared",
                },
                "product_00_maded": {
                    "state": "unappeared",
                    "logistic_state": "unappeared",
                },
            },
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
    }
}
