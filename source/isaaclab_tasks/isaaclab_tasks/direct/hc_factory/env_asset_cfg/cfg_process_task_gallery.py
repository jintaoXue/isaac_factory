from isaaclab_tasks.direct.hc_factory.src import human


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
            # # 运存放区
            # "product_to_storage": {
            #     "machine": "num07_gantry_group",
            #     "alternative_robot": "AGV",
            #     "required_materials": {"product_water_pipe": 1},
            #     "process_time": 100,
            #     "gaussian_random_time": 10,
            #     "material_state_sequence": ["on_machine", "conveying", "on_storage"],
            # },
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
    # "product_to_storage": 13,
}

CfgProcessTaskGalleryClassified = {
    "ProductWaterPipe": {
        #the dict value is the index in the common encoded index space defined by CfgProcessTaskGalleryInAll
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
        # "product_to_storage": 13,
    },
    #Will add more product types in the future, and they will share the same common encoded index space defined by CfgProcessTaskGalleryInAll
}


CfgSubtaskGallery = {
    "task_type_logistic":{
        "have_AGV":{
            # human: 0, gantry: 1, robot: 2
            "ongoing": ["go_to_material", "go_to_material", "go_to_material", "on_start_area"],
            "ongoing_index": 0,
            "material_start_area" : None,
            "material_goal_area" : None,
            "num_subtasks": 9,
            "finished": [False,False,False],
            "subtasks": [
                # human: 0, gantry: 1, robot: 2
                ["go_to_material", "go_to_material", "go_to_material", "on_start_area"],
                ["material_on_gantry", "wait", "wait", "on_start_area"],
                ["control_gantry", "carry_to_robot", "wait", "on_gantry"],
                ["material_on_robot", "wait", "wait", "on_gantry"],
                ["go_to_goal_area", "move_to_goal_area", "carry_to_goal_area", "on_robot"],
                ["material_on_gantry", "wait", "wait", "on_robot"],
                ["control_gantry", "move_to_goal_area", "wait", "on_gantry"],
                ["material_on_goal_area", "wait", "done", "on_gantry"],
                ["done", "done", "done", "on_goal_area"],
                # ["go_to_storage", "go_to_storage", "go_to_storage"],
                # ["material_on_gantry", "wait", "wait"],
                # ["control_gantry", "wait", "carry_to_robot"],
                # ["material_on_robot", "wait", "done"],
                # ["go_to_target_machine", "carry_to_target_area", "done"],
                # ["material_on_target_area", "done", "done"],
                # ["done", "done", "done"],
            ],
        },
        "only_have_gantry":{
            # human: 0, gantry: 1
            "ongoing": ["go_to_material", "go_to_material", "on_start_area"],
            "ongoing_index": 0,
            "material_start_area" : None,
            "material_end_area" : None,
            "num_subtasks": 5,
            "finished": [False,False],
            "subtasks": [
                #human: 0, gantry: 1, material: -1
                ["go_to_material", "go_to_material", "on_start_area"],
                ["material_on_gantry", "wait", "on_start_area"],
                ["control_gantry_while_going_to_goal_area", "carry_to_goal_area", "on_gantry"],
                ["material_on_goal_area", "wait", "on_gantry"],
                ["done", "done", "on_goal_area"],
            ],
        },
    },
    "task_type_processing":{
            # human: 0, machine: 1
            "ongoing": ["go_to_target_machine", "none", "wait", "on_machine"],
            "ongoing_index": 0,
            "material_start_area" : None,
            "material_end_area" : None,
            "index_to_decide_end_area": 5,
            "num_subtasks": 8,
            "finished": [False, None, True],
            "subtasks": [
                #human: 0, gantry: 1, machine: 2, material: -1
                ["go_to_target_machine", "none", "wait", "on_machine"],
                ["control_machine", "none", "process", "on_machine"],
                ["wait", "finding_free_gantry", "wait", "on_machine"],
                ["control_gantry", "move_to_target_machine", "wait", "on_machine"],
                ["material_on_gantry", "wait", "wait", "on_machine"],
                ["control_gantry", "move_to_end_area", "done", "on_gantry"],
                ["material_on_end_area", "wait", "done", "on_gantry"],
                ["done", "done", "done", "on_end_area"],
            ],
    },
}

CfgSubtaskPredefinedTimeGallery = {
   "material_on_gantry": 10,
   "control_gantry": 10,
   "material_on_robot": 10,
   "material_on_target_area": 10,
   "control_gantry_while_going_to_target_machine": 10,
   "control_machine": 10,
}


CfgProcessTaskToTargetMapping = {
    "none": {"task_type": "none", "target_machine": None, "logistic_machine": None, "is_final_task": False},
    "logistic_for_pipe_cutting": {"task_type": "logistic", "target_machine": "num02_rollerbedCNCPipeIntersectionCuttingMachine", "logistic_machine": "num07_gantry_group", "is_final_task": False},
    "pipe_cutting": {"task_type": "processing", "target_machine": "num02_rollerbedCNCPipeIntersectionCuttingMachine", "logistic_machine": None, "is_final_task": False},
    "logistic_for_pipe_grooving": {"task_type": "logistic", "target_machine": "num04_groovingMachineLarge", "logistic_machine": "num07_gantry_group", "is_final_task": False},
    "pipe_grooving": {"task_type": "processing", "target_machine": "num04_groovingMachineLarge", "logistic_machine": None, "is_final_task": False},
    "logistic_for_batch_spot_welding": {"task_type": "logistic", "target_machine": "num08_workbench", "logistic_machine": "num07_gantry_group", "is_final_task": False},
    "batch_spot_welding": {"task_type": "processing", "target_machine": "num08_workbench", "logistic_machine": None, "is_final_task": False},
    "logistic_for_arc_welding_root": {"task_type": "logistic", "target_machine": "num01_weldingRobot", "logistic_machine": "num07_gantry_group", "is_final_task": False},
    "arc_welding_root": {"task_type": "processing", "target_machine": "num01_weldingRobot", "logistic_machine": None, "is_final_task": False},
    "logistic_for_MIG_welding_surface": {"task_type": "logistic", "target_machine": "num00_rotaryPipeAutomaticWeldingMachine", "logistic_machine": "num07_gantry_group", "is_final_task": False},
    "MIG_welding_surface": {"task_type": "processing", "target_machine": "num00_rotaryPipeAutomaticWeldingMachine", "logistic_machine": None, "is_final_task": False},
    "logistic_for_paint_rust_proof": {"task_type": "logistic", "target_machine": "num08_workbench", "logistic_machine": "num07_gantry_group", "is_final_task": False},
    "paint_rust_proof": {"task_type": "processing", "target_machine": "num08_workbench", "logistic_machine": None, "is_final_task": True},
    # "product_to_storage": {"task_type": "logistic", "target_machine": "num08_workbench", "logistic_machine": "num07_gantry_group"},
}


TaskRecordTemplate = {

    "task_done": False,
    "task": None,
    "task_index": None,
    #processing, logistic
    "task_type": None,

    "product": None,  
    "product_index": None,
    "new_product_selected": False,
    "storage_name": None,

    "human": None,
    "human_index": None,
    "robot": None,
    "robot_index": None,

    "target_machine": None,
    "target_machine_workstation": None,
    "chosen_free_workstation_index": None,
    "logistic_machine": None,

    "subtasks_dict": None,
}


# CfgTask2SubtaskGallery = {
#     "none": [],
#     "logistic_for_pipe_cutting": ["human go to storage to get raw pipe and gantry move it to storage", 
#                                   "human use rope to lock pipe on gantry, gantry move pipe to num02_rollerbedCNCPipeIntersectionCuttingMachine"
#                                   "or put it on AGV, AGV move it to num02_rollerbedCNCPipeIntersectionCuttingMachine "
#                                   "and wait for gantry and human to release the lock", "done, material on machine and human and gantry are free"],
#     "pipe_cutting": ["human operate num02_rollerbedCNCPipeIntersectionCuttingMachine to cut the pipe", "done, material on machine and human are free"],
#     "logistic_for_pipe_grooving": ["human go to storage to get raw pipe and gantry move it to storage", 
#                                    "human use rope to lock pipe on gantry, gantry move pipe to num03_rollerbedCNCPipeGroovingMachine"
#                                    "or put it on AGV, AGV move it to num03_rollerbedCNCPipeGroovingMachine "
#                                    "and wait for gantry and human to release the lock", "done, material on machine and human and gantry are free"],
#     "pipe_grooving": ["human operate num03_rollerbedCNCPipeGroovingMachine to groove the pipe", "done, material on machine and human are free"],
#     "logistic_for_batch_spot_welding": ["human go to storage to get raw materials and gantry move them to storage", 
#                                         "human use rope to lock materials on gantry, gantry move materials to num04_batchSpotWeldingMachine"
#                                         "or put them on AGV, AGV move them to num04_batchSpotWeldingMachine "
#                                         "and wait for gantry and human to release the lock", "done, material on machine and human and gantry are free"],
#     "batch_spot_welding": ["human operate num04_batchSpotWeldingMachine to weld the materials", "done, material on machine and human are free"],
#     "logistic_for_arc_welding_root": ["human go to storage to get raw materials and gantry move them to storage", 
#                                       "human use rope to lock materials on gantry, gantry move materials to num05_arcWeldingRootMachine"
#                                       "or put them on AGV, AGV move them to num05_arcWeldingRootMachine "
#                                       "and wait for gantry and human to release the lock", "done, material on machine and human and gantry are free"],
#     "arc_welding_root": ["human operate num05_arcWeldingRootMachine to weld the materials", "done, material on machine and human are free"],
#     "logistic_for_MIG_welding_surface": ["human go to storage to get raw materials and gantry move them to storage", 
#                                          "human use rope to lock materials on gantry, gantry move materials to num06_MIGWeldingSurfaceMachine"
#                                          "or put them on AGV, AGV move them to num06_MIGWeldingSurfaceMachine "
#                                          "and wait for gantry and human to release the lock", "done, material on machine and human and gantry are free"],
#     "MIG_welding_surface": ["human operate num06_MIGWeldingSurfaceMachine to weld the materials", "done, material on machine and human are free"],
#     "logistic_for_paint_rust_proof": ["human go to storage to get raw materials and gantry move them to storage", 
#                                       "human use rope to lock materials on gantry, gantry move materials to num08_workbench"
#                                       "or put them on AGV, AGV move them to num08_workbench "
#                                       "and wait for gantry and human to release the lock", "done, material on machine and human and gantry are free"],
#     "paint_rust_proof": ["human operate num08_workbench to paint the materials", "done, material on machine and human are free"],
#     # "product_to_storage": ["human go to storage to get finished products and gantry move them to storage", 
#     #                        "human use rope to lock products on gantry, gantry move products to num07_gantry_group"
#     #                        "or put them on AGV, AGV move them to num07_gantry_group "
#     #                        "and wait for gantry and human to release the lock", "done, material on machine and human and gantry are free"],
# }