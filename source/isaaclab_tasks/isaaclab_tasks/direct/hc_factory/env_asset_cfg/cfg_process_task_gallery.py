from .cfg_process_subtask_gallery import CfgSubtaskGallery


TaskRecordTemplate = {

    "task_done": False,
    "task": None,
    "task_index": None,
    #processing, logistic
    "task_type": None,
    "is_final_task": None,

    "product": None,  
    "product_index": None,
    "new_product_selected": False,
    "submaterials": None,
    "logistic_submaterial": None,
    "processing_submaterials": None,
    "processed_material": None,
    
    "human": None,
    "human_index": None,
    "robot": None,
    "robot_index": None,

    "target_machine": None,
    "chosen_machine_workstation": None,
    "chosen_workstation_index": None,
    "logistic_machine": None,
    "chosen_gantry_index": None,

    "subtasks_dict": None,
}


CfgProductProcessGallery = {
    #Currently we only have one product type "ProductWaterPipe", but we can easily add more product types in the future
    "ProductWaterPipe": {
        "num_process_steps": 6,
        "process_steps": {
            "pipe_cutting": {
                "machine": "num02_rollerbedCNCPipeIntersectionCuttingMachine",
                "required_materials": {"pipe": 1},
                "process_time": 100,
                "gaussian_random_time": 10,
            },
            "pipe_grooving": {
                "machine": "num04_groovingMachineLarge",
                "required_materials": {"pipe": 1},
                "process_time": 100,
                "gaussian_random_time": 10,
            },
            "batch_spot_welding": {
                "machine": "num08_workbench",
                "required_materials": {"pipe": 1, "flange": 1, "elbow": 1},
                "process_time": 100,
                "gaussian_random_time": 10,
            },
            # This process step refers to "Argon arc welding root" (氩弧焊底焊 in Chinese)
            "arc_welding_root": {
                "machine": "num01_weldingRobot",
                "required_materials": {"product_water_pipe": 1},
                "process_time": 100,
                "gaussian_random_time": 10,
            },
            # This process step refers to "MIG welding surface" (MIG焊面焊 in Chinese), MIG full name is Metal Inert Gas welding
            "MIG_welding_surface": {
                "machine": "num00_rotaryPipeAutomaticWeldingMachine",
                "required_materials": {"product_water_pipe": 1},
                "process_time": 100,
                "gaussian_random_time": 10,
            },
            "paint_rust_proof": {
                "machine": "num08_workbench",
                "required_materials": {"product_water_pipe": 1},
                "process_time": 100,
                "gaussian_random_time": 10,
            },
        }
    }
}


# This contains the task gallery for all product types; each product type has its own process gallery.
# All product process tasks share a common encoded index space defined by CfgProcessTaskGallery.
CfgProcessTaskGalleryInAll = {
    "none": 0,
    "logistic_for_pipe_cutting": 1,
    "pipe_cutting": 2,
    "logistic_for_pipe_grooving": 3,
    "pipe_grooving": 4,
    "logistic_for_batch_spot_welding": 5,
    "batch_spot_welding": 6,
    "logistic_for_arc_welding_root": 7,
    "arc_welding_root": 8,
    "logistic_for_MIG_welding_surface": 9,
    "MIG_welding_surface": 10,
    "logistic_for_paint_rust_proof": 11,
    "paint_rust_proof": 12,
}



CfgProcessTaskGalleryDetailedClassified = {
    "ProductWaterPipe": {
        "none": {
            "task_type": "none",
            "target_machine": None,
            "logistic_machine": None,
            "is_final_task": False,
            "logistic_submaterial": None,
            "processing_submaterials": None,
            "processed_material": None,
            "subtasks_dict": CfgSubtaskGallery["ProductWaterPipe"]["none"],
        },
        "logistic_for_pipe_cutting": {
            "task_type": "logistic",
            "target_machine": "num02_rollerbedCNCPipeIntersectionCuttingMachine",
            "logistic_machine": "num07_gantry_group",
            "is_final_task": False,
            "logistic_submaterial": "product_00_pipe",
            "processing_submaterials": None,
            "processed_material": None,
            "subtasks_dict": CfgSubtaskGallery["ProductWaterPipe"]["logistic_for_pipe_cutting"],
        },
        "pipe_cutting": {
            "task_type": "processing",
            "target_machine": "num02_rollerbedCNCPipeIntersectionCuttingMachine",
            "logistic_machine": None,
            "is_final_task": False,
            "logistic_submaterial": None,
            "processing_submaterials": ["product_00_pipe"],
            "processed_material": "product_00_pipe",
            "subtasks_dict": CfgSubtaskGallery["ProductWaterPipe"]["pipe_cutting"],
        },
        "logistic_for_pipe_grooving": {
            "task_type": "logistic",
            "target_machine": "num04_groovingMachineLarge",
            "logistic_machine": "num07_gantry_group",
            "is_final_task": False,
            "logistic_submaterial": "product_00_pipe",
            "processing_submaterials": None,
            "processed_material": None,
            "subtasks_dict": CfgSubtaskGallery["ProductWaterPipe"]["logistic_for_pipe_grooving"],
        },
        "pipe_grooving": {
            "task_type": "processing",
            "target_machine": "num04_groovingMachineLarge",
            "logistic_machine": None,
            "is_final_task": False,
            "logistic_submaterial": None,
            "processing_submaterials": ["product_00_pipe"],
            "processed_material": "product_00_pipe",
            "subtasks_dict": CfgSubtaskGallery["ProductWaterPipe"]["pipe_grooving"],
        },
        "logistic_for_batch_spot_welding": {
            "task_type": "logistic",
            "target_machine": "num08_workbench",
            "logistic_machine": "num07_gantry_group",
            "is_final_task": False,
            "logistic_submaterial": "product_00_pipe",
            "processing_submaterials": None,
            "processed_material": None,
            "subtasks_dict": CfgSubtaskGallery["ProductWaterPipe"]["logistic_for_batch_spot_welding"],
        },
        "batch_spot_welding": {
            "task_type": "processing",
            "target_machine": "num08_workbench",
            "logistic_machine": None,
            "is_final_task": False,
            "logistic_submaterial": None,
            "processing_submaterials": ["product_00_pipe", "product_00_flange", "product_00_elbow"],
            "processed_material": "product_00_semi",
            "subtasks_dict": CfgSubtaskGallery["ProductWaterPipe"]["batch_spot_welding"],
        },
        "logistic_for_arc_welding_root": {
            "task_type": "logistic",
            "target_machine": "num01_weldingRobot",
            "logistic_machine": "num07_gantry_group",
            "is_final_task": False,
            "logistic_submaterial": "product_00_semi",
            "processing_submaterials": None,
            "processed_material": None,
            "subtasks_dict": CfgSubtaskGallery["ProductWaterPipe"]["logistic_for_arc_welding_root"],
        },
        "arc_welding_root": {
            "task_type": "processing",
            "target_machine": "num01_weldingRobot",
            "logistic_machine": None,
            "is_final_task": False,
            "logistic_submaterial": None,
            "processing_submaterials": ["product_00_semi"],
            "processed_material": "product_00_semi",
            "subtasks_dict": CfgSubtaskGallery["ProductWaterPipe"]["arc_welding_root"],
        },
        "logistic_for_MIG_welding_surface": {
            "task_type": "logistic",
            "target_machine": "num00_rotaryPipeAutomaticWeldingMachine",
            "logistic_machine": "num07_gantry_group",
            "is_final_task": False,
            "logistic_submaterial": "product_00_semi",
            "processing_submaterials": None,
            "processed_material": None,
            "subtasks_dict": CfgSubtaskGallery["ProductWaterPipe"]["logistic_for_MIG_welding_surface"],
        },
        "MIG_welding_surface": {
            "task_type": "processing",
            "target_machine": "num00_rotaryPipeAutomaticWeldingMachine",
            "logistic_machine": None,
            "is_final_task": False,
            "logistic_submaterial": None,
            "processing_submaterials": ["product_00_semi"],
            "processed_material": "product_00_semi",
            "subtasks_dict": CfgSubtaskGallery["ProductWaterPipe"]["MIG_welding_surface"],
        },
        "logistic_for_paint_rust_proof": {
            "task_type": "logistic",
            "target_machine": "num08_workbench",
            "logistic_machine": "num07_gantry_group",
            "is_final_task": False,
            "logistic_submaterial": "product_00_semi",
            "processing_submaterials": None,
            "processed_material": None,
            "subtasks_dict": CfgSubtaskGallery["ProductWaterPipe"]["logistic_for_paint_rust_proof"],
        },
        "paint_rust_proof": {
            "task_type": "processing",
            "target_machine": "num08_workbench",
            "logistic_machine": None,
            "is_final_task": True,
            "logistic_submaterial": None,
            "processing_submaterials": ["product_00_semi"],
            "processed_material": "product_00_maded",
            "subtasks_dict": CfgSubtaskGallery["ProductWaterPipe"]["paint_rust_proof"],
        },
    },
}



