import copy
from .cfg_machine import CfgMachine

CfgSubtaskPredefinedTimeGallery = {
   "material_on_gantry": 10,
   "control_gantry": 10,
   "material_on_robot": 10,
   "material_on_target_area": 10,
   "control_gantry_while_going_to_target_machine": 10,
   "control_machine": 10,
}

CfgSubtaskGallery = {
    "ProductWaterPipe": {
        # task id 1
        "logistic_for_pipe_cutting":{
            "have_AGV":{
                # human: 0, gantry: 1, machine: 2, robot: 3, 
                "ongoing": ["go_to_material", "go_to_material", "wait", "go_to_material"],
                "ongoing_index": 0,
                "required_logistic_material": "product_00_pipe",
                # material_start_area need to be set in task_progress_manager.py
                "material_start_area" : None,
                "material_goal_area" : "num02_rollerbedCNCPipeIntersectionCuttingMachine",
                "goal_area_workstation_key" : None,
                "start_area_ids": None,
                "goal_area_ids": CfgMachine["num02_rollerbedCNCPipeIntersectionCuttingMachine"]["working_area_ids"],
                "num_subtasks": 9,
                "finished": [False, False, False, False],
                "subtasks": [
                    # human: 0, gantry: 1, machine: 2, robot: 3
                    ["go_to_material", "go_to_material", "wait", "go_to_material"],
                    ["material_on_gantry", "wait", "wait", "wait"],
                    ["control_gantry", "carry_to_robot", "wait", "wait"],
                    ["material_on_robot", "wait", "wait", "wait"],
                    ["go_to_goal_area", "move_to_goal_area", "wait", "carry_to_goal_area"],
                    ["material_on_gantry", "wait", "wait", "wait"],
                    ["control_gantry", "move_to_goal_area", "wait", "wait"],
                    ["material_on_goal_area", "wait", "wait", "done"],
                    ["done", "done", "done", "done"],
                ],
                "material_states_in_subtasks": {
                    "product_00_pipe": ["on_start_area", "on_start_area", "on_gantry", "on_gantry", "on_robot", "on_robot", "on_gantry", "on_gantry", "on_goal_area"],
                    "product_00_flange": ["on_start_area"]*9,
                    "product_00_elbow": ["on_start_area"]*9,
                    "product_00_semi": ["disappear"]*9,
                    "product_00_maded": ["disappear"]*9,
                }    
            },
            "only_have_gantry":{
                # human: 0, gantry: 1, machine: 2
                "ongoing": ["go_to_material", "go_to_material", "wait"],
                "ongoing_index": 0,
                "required_logistic_material": "product_00_pipe",
                "material_start_area" : None,
                "material_goal_area" : "num02_rollerbedCNCPipeIntersectionCuttingMachine",
                "goal_area_workstation_key" : None,
                "start_area_ids": None,
                "goal_area_ids": CfgMachine["num02_rollerbedCNCPipeIntersectionCuttingMachine"]["working_area_ids"],
                "num_subtasks": 5,
                "finished": [False, False, False],
                "subtasks": [
                    #human: 0, gantry: 1, machine: 2
                    ["go_to_material", "go_to_material", "wait"],
                    ["material_on_gantry", "wait", "wait"],
                    ["go_to_goal_area", "carry_to_goal_area", "wait"],
                    ["material_on_goal_area", "wait", "wait"],
                    ["done", "done", "done"],
                ],
                "material_states_in_subtasks": {
                    "product_00_pipe": ["on_start_area", "on_start_area", "on_gantry", "on_gantry", "on_goal_area"],
                    "product_00_flange": ["on_start_area"]*5,
                    "product_00_elbow": ["on_start_area"]*5,
                    "product_00_semi": ["disappear"]*5,
                    "product_00_maded": ["disappear"]*5,
                }    
            },
        },
        #task id 2
        "pipe_cutting":{
                # human: 0, gantry: 1, machine: 2
                "ongoing": ["go_to_target_machine", "none", "wait"],
                "ongoing_index": 0,
                "required_processing_material": "product_00_pipe",
                "processed_material": "product_00_pipe",
                "material_start_area" : "num02_rollerbedCNCPipeIntersectionCuttingMachine",
                # material_goal_area need to be set in task_progress_manager.py after processing and needed to be carryed to a storage or machine
                "material_goal_area" : None,
                ### if goal_area is a machine, then goal_area_workstation_key is the workstation key of the machine
                "goal_area_workstation_key" : None,
                "start_area_ids": CfgMachine["num02_rollerbedCNCPipeIntersectionCuttingMachine"]["working_area_ids"],
                "goal_area_ids": None,
                "index_to_decide_goal_area": 5,
                "num_subtasks": 8,
                "finished": [False, None, True],
                "subtasks": [
                    #human: 0, gantry: 1, machine: 2,
                    ["go_to_target_machine", "none", "wait"],
                    ["control_machine", "none", "process"],
                    ["wait", "finding_free_gantry", "wait"],
                    ["control_gantry", "move_to_target_machine", "wait"],
                    ["material_on_gantry", "wait", "wait"],
                    ["control_gantry", "carry_to_goal_area", "done"],
                    ["material_on_goal_area", "wait", "done"],
                    ["done", "done", "done"],
                ],
                "material_states_in_subtasks": {
                    "product_00_pipe": ["on_machine", "on_machine", "on_machine", "on_machine", "on_machine", "on_gantry", "on_gantry", "on_goal_area"],
                    "product_00_flange": ["on_start_area"]*8,
                    "product_00_elbow": ["on_start_area"]*8,
                    "product_00_semi": ["disappear"]*8,
                    "product_00_maded": ["disappear"]*8,
                }  
        },
    }
}

# task id 3
# logistic_for_pipe_grooving
CfgSubtaskGallery["ProductWaterPipe"]["logistic_for_pipe_grooving"] = copy.deepcopy(CfgSubtaskGallery["ProductWaterPipe"]["logistic_for_pipe_cutting"])
CfgSubtaskGallery["ProductWaterPipe"]["logistic_for_pipe_grooving"]["have_AGV"]["material_goal_area"] = "num04_groovingMachineLarge"
CfgSubtaskGallery["ProductWaterPipe"]["logistic_for_pipe_grooving"]["only_have_gantry"]["material_goal_area"] = "num04_groovingMachineLarge"
CfgSubtaskGallery["ProductWaterPipe"]["logistic_for_pipe_grooving"]["have_AGV"]["goal_area_ids"] = CfgMachine["num04_groovingMachineLarge"]["working_area_ids"]
CfgSubtaskGallery["ProductWaterPipe"]["logistic_for_pipe_grooving"]["only_have_gantry"]["goal_area_ids"] = CfgMachine["num04_groovingMachineLarge"]["working_area_ids"]
# task id 4
# pipe_grooving
CfgSubtaskGallery["ProductWaterPipe"]["pipe_grooving"] = copy.deepcopy(CfgSubtaskGallery["ProductWaterPipe"]["pipe_cutting"])
CfgSubtaskGallery["ProductWaterPipe"]["pipe_grooving"]["material_start_area"] = "num04_groovingMachineLarge"
CfgSubtaskGallery["ProductWaterPipe"]["pipe_grooving"]["start_area_ids"] = CfgMachine["num04_groovingMachineLarge"]["working_area_ids"]
# task id 5
# logistic_for_batch_spot_welding
CfgSubtaskGallery["ProductWaterPipe"]["logistic_for_batch_spot_welding"] = copy.deepcopy(
    CfgSubtaskGallery["ProductWaterPipe"]["logistic_for_pipe_cutting"]
)
CfgSubtaskGallery["ProductWaterPipe"]["logistic_for_batch_spot_welding"]["have_AGV"]["material_goal_area"] = "num08_workbench"
CfgSubtaskGallery["ProductWaterPipe"]["logistic_for_batch_spot_welding"]["only_have_gantry"]["material_goal_area"] = "num08_workbench"
CfgSubtaskGallery["ProductWaterPipe"]["logistic_for_batch_spot_welding"]["have_AGV"]["goal_area_ids"] = CfgMachine["num08_workbench"]["working_area_ids"]
CfgSubtaskGallery["ProductWaterPipe"]["logistic_for_batch_spot_welding"]["only_have_gantry"]["goal_area_ids"] = CfgMachine["num08_workbench"]["working_area_ids"]
# task id 6
# batch_spot_welding
CfgSubtaskGallery["ProductWaterPipe"]["batch_spot_welding"] = copy.deepcopy(CfgSubtaskGallery["ProductWaterPipe"]["pipe_cutting"])
CfgSubtaskGallery["ProductWaterPipe"]["batch_spot_welding"]["material_start_area"] = "num08_workbench"
CfgSubtaskGallery["ProductWaterPipe"]["batch_spot_welding"]["required_processing_material"] = ["product_00_pipe", "product_00_flange", "product_00_elbow"]
CfgSubtaskGallery["ProductWaterPipe"]["batch_spot_welding"]["processed_material"] = "product_00_semi"
CfgSubtaskGallery["ProductWaterPipe"]["batch_spot_welding"]["material_states_in_subtasks"] = {
        "product_00_pipe": ["on_machine"]*2 + ["disappear"]*6,
        "product_00_flange": ["on_start_area"]*2 + ["disappear"]*6,
        "product_00_elbow": ["on_start_area"]*2 + ["disappear"]*6,
        "product_00_semi": ["disappear", "disappear", "on_machine", "on_machine", "on_machine", "on_gantry", "on_gantry", "on_goal_area"],
        "product_00_maded": ["disappear"]*8,
    }
CfgSubtaskGallery["ProductWaterPipe"]["batch_spot_welding"]["start_area_ids"] = CfgMachine["num08_workbench"]["working_area_ids"]
# task id 7
# logistic_for_arc_welding_root
CfgSubtaskGallery["ProductWaterPipe"]["logistic_for_arc_welding_root"] = copy.deepcopy(
    CfgSubtaskGallery["ProductWaterPipe"]["logistic_for_pipe_cutting"]
)
for _mode in ("have_AGV", "only_have_gantry"):
    _logistic = CfgSubtaskGallery["ProductWaterPipe"]["logistic_for_arc_welding_root"][_mode]
    _logistic["material_goal_area"] = "num01_weldingRobot"
    _logistic["required_logistic_material"] = "product_00_semi"
    if _mode == "have_AGV":
        _logistic["material_states_in_subtasks"] = {
            "product_00_pipe": ["disappear"]*9,
            "product_00_flange": ["disappear"]*9,
            "product_00_elbow": ["disappear"]*9,
            "product_00_semi": ["on_start_area", "on_start_area", "on_gantry", "on_gantry", "on_robot", "on_robot", "on_gantry", "on_gantry", "on_goal_area"],
            "product_00_maded": ["disappear"]*9,
        }
    elif _mode == "only_have_gantry":
        _logistic["material_states_in_subtasks"] = {
            "product_00_pipe": ["disappear"]*5,
            "product_00_flange": ["disappear"]*5,
            "product_00_elbow": ["disappear"]*5,
            "product_00_semi": ["on_start_area", "on_start_area", "on_gantry", "on_gantry", "on_goal_area"],
            "product_00_maded": ["disappear"]*5,
        }
    _logistic["goal_area_ids"] = CfgMachine["num01_weldingRobot"]["working_area_ids"]
# task id 8
# arc_welding_root
CfgSubtaskGallery["ProductWaterPipe"]["arc_welding_root"] = copy.deepcopy(CfgSubtaskGallery["ProductWaterPipe"]["batch_spot_welding"])
CfgSubtaskGallery["ProductWaterPipe"]["arc_welding_root"]["material_start_area"] = "num01_weldingRobot"
CfgSubtaskGallery["ProductWaterPipe"]["arc_welding_root"]["required_processing_material"] = "product_00_semi"
CfgSubtaskGallery["ProductWaterPipe"]["arc_welding_root"]["material_states_in_subtasks"] = {
        "product_00_pipe": ["disappear"]*8,
        "product_00_flange": ["disappear"]*8,
        "product_00_elbow": ["disappear"]*8,
        "product_00_semi": ["on_machine", "on_machine", "on_machine", "on_machine", "on_machine", "on_gantry", "on_gantry", "on_goal_area"],
        "product_00_maded": ["disappear"]*8,
    }
CfgSubtaskGallery["ProductWaterPipe"]["arc_welding_root"]["start_area_ids"] = CfgMachine["num01_weldingRobot"]["working_area_ids"]

# task id 9
# logistic_for_MIG_welding_surface
CfgSubtaskGallery["ProductWaterPipe"]["logistic_for_MIG_welding_surface"] = copy.deepcopy(
    CfgSubtaskGallery["ProductWaterPipe"]["logistic_for_arc_welding_root"]
)
for _mode in ("have_AGV", "only_have_gantry"):
    _logistic = CfgSubtaskGallery["ProductWaterPipe"]["logistic_for_MIG_welding_surface"][_mode]
    _logistic["material_goal_area"] = "num00_rotaryPipeAutomaticWeldingMachine"
    _logistic["goal_area_ids"] = CfgMachine["num00_rotaryPipeAutomaticWeldingMachine"]["working_area_ids"]
# task id 10
# MIG_welding_surface
CfgSubtaskGallery["ProductWaterPipe"]["MIG_welding_surface"] = copy.deepcopy(CfgSubtaskGallery["ProductWaterPipe"]["arc_welding_root"])
CfgSubtaskGallery["ProductWaterPipe"]["MIG_welding_surface"]["material_start_area"] = "num00_rotaryPipeAutomaticWeldingMachine"
CfgSubtaskGallery["ProductWaterPipe"]["MIG_welding_surface"]["start_area_ids"] = CfgMachine["num00_rotaryPipeAutomaticWeldingMachine"]["working_area_ids"]

# task id 11
# logistic_for_paint_rust_proof
CfgSubtaskGallery["ProductWaterPipe"]["logistic_for_paint_rust_proof"] = copy.deepcopy(
    CfgSubtaskGallery["ProductWaterPipe"]["logistic_for_arc_welding_root"]
)

for _mode in ("have_AGV", "only_have_gantry"):
    _logistic = CfgSubtaskGallery["ProductWaterPipe"]["logistic_for_paint_rust_proof"][_mode]
    _logistic["material_goal_area"] = "num08_workbench"
    _logistic["goal_area_ids"] = CfgMachine["num08_workbench"]["working_area_ids"]
# task id 12
# paint_rust_proof
CfgSubtaskGallery["ProductWaterPipe"]["paint_rust_proof"] = copy.deepcopy(CfgSubtaskGallery["ProductWaterPipe"]["arc_welding_root"])
CfgSubtaskGallery["ProductWaterPipe"]["paint_rust_proof"]["material_start_area"] = "num08_workbench"
CfgSubtaskGallery["ProductWaterPipe"]["paint_rust_proof"]["processed_material"] = "product_00_maded"
CfgSubtaskGallery["ProductWaterPipe"]["paint_rust_proof"]["material_states_in_subtasks"] = {
        "product_00_pipe": ["disappear"]*8,
        "product_00_flange": ["disappear"]*8,
        "product_00_elbow": ["disappear"]*8,
        "product_00_semi": ["on_machine"]*2 + ["disappear"]*6,
        "product_00_maded": ["disappear", "disappear", "on_machine", "on_machine", "on_machine", "on_gantry", "on_gantry", "on_goal_area"],
    }
CfgSubtaskGallery["ProductWaterPipe"]["paint_rust_proof"]["start_area_ids"] = CfgMachine["num08_workbench"]["working_area_ids"]