import copy
from .cfg_machine import CfgMachine

CfgSubtaskPredefinedTimeGallery = {
   "go_to_material": None,
   "material_on_gantry": 25,
   "control_gantry": 25,
   "material_on_robot": 25,
   "go_to_goal_area": None,
   "material_on_goal_area": 25,
   "go_to_processing_machine": None,
   "control_machine": 100,
   "wait": None,
   "done": None,
}

# Gaussian noise (steps) sampled once when entering a time-counting subtask.
SubtaskTimeNoiseStdSteps = 2.0

CfgSubtaskGallery = {
    "ProductWaterPipe": {
        # task id 1 — haul cube_raw from storage to cutting machine
        # Logistic: start/goal known at dispatch → pick have_AGV / only_have_gantry up front.
        "logistic_for_pipe_cutting":{
            "have_AGV":{
                # Cross-zone only: start crane loads AGV → AGV carries → goal crane unloads.
                # human: 0, gantry: 1, machine: 2, robot: 3
                "ongoing": ["go_to_material", "go_to_material", "wait", "go_to_material"],
                "ongoing_index": 0,
                "required_logistic_material": "product_00_pipe_raw",
                # material_start_area need to be set in task_progress_manager.py
                "material_start_area" : None,
                "material_goal_area" : "num02_rollerbedCNCPipeIntersectionCuttingMachine",
                "goal_area_workstation_key" : None,
                "start_area_ids": None,
                "goal_area_ids": CfgMachine["num02_rollerbedCNCPipeIntersectionCuttingMachine"]["working_area_ids"],
                "num_subtasks": 11,
                "finished": [False, False, False, False],
                "subtasks": [
                    # 0–3: start-zone gantry loads onto AGV
                    ["go_to_material", "go_to_material", "wait", "go_to_material"],
                    ["material_on_gantry", "wait", "wait", "wait"],
                    ["control_gantry", "carry_to_robot", "wait", "wait"],
                    ["material_on_robot", "wait", "wait", "wait"],
                    # 4: release start gantry (done); AGV alone to goal
                    ["go_to_goal_area", "done", "wait", "carry_to_goal_area"],
                    # 5–8: goal-zone gantry (another crane) picks from AGV and places
                    ["wait", "finding_free_gantry", "wait", "wait"],
                    # Robot "done" at row 6 frees the AGV; later robot cols must be
                    # "none"/"done" (auto-finished on advance). A post-release "wait"
                    # deadlocks because no agent is bound to mark finished[3].
                    ["control_gantry", "go_to_goal_robot", "wait", "done"],
                    ["material_on_gantry", "wait", "wait", "done"],
                    ["control_gantry", "carry_to_goal_area", "wait", "none"],
                    ["material_on_goal_area", "wait", "wait", "none"],
                    ["done", "done", "done", "done"],
                ],
                "material_states_in_subtasks": {
                    "product_00_pipe_raw": (
                        ["on_start_area", "on_start_area", "on_gantry", "on_gantry"]
                        + ["on_robot", "on_robot", "on_robot"]
                        + ["on_gantry", "on_gantry", "on_gantry", "on_goal_area"]
                    ),
                    "product_00_pipe": ["disappear"] * 11,
                    "product_00_flange": ["on_start_area"] * 11,
                    "product_00_elbow": ["on_start_area"] * 11,
                    "product_00_semi": ["disappear"] * 11,
                    "product_00_maded": ["disappear"] * 11,
                }
            },
            "only_have_gantry":{
                # human: 0, gantry: 1, machine: 2
                "ongoing": ["go_to_material", "go_to_material", "wait"],
                "ongoing_index": 0,
                "required_logistic_material": "product_00_pipe_raw",
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
                    "product_00_pipe_raw": ["on_start_area", "on_start_area", "on_gantry", "on_gantry", "on_goal_area"],
                    "product_00_pipe": ["disappear"]*5,
                    "product_00_flange": ["on_start_area"]*5,
                    "product_00_elbow": ["on_start_area"]*5,
                    "product_00_semi": ["disappear"]*5,
                    "product_00_maded": ["disappear"]*5,
                }
            },
        },
        # task id 2 — processing: goal (machine/storage) is known only AFTER process.
        # Runtime loads process_prefix only; after goal is set, TaskManager appends
        # outbound_same_zone or outbound_cross_zone (see task_progress_manager).
        "pipe_cutting":{
            # human: 0, gantry: 1, machine: 2 — no robot until outbound_cross_zone is attached
            "process_prefix":{
                "ongoing": ["go_to_processing_machine", "none", "wait"],
                "ongoing_index": 0,
                "required_processing_material": "product_00_pipe_raw",
                "processed_material": "product_00_pipe",
                "material_start_area" : "num02_rollerbedCNCPipeIntersectionCuttingMachine",
                "material_goal_area" : None,
                "goal_area_workstation_key" : None,
                "start_area_ids": CfgMachine["num02_rollerbedCNCPipeIntersectionCuttingMachine"]["working_area_ids"],
                "goal_area_ids": None,
                # Decide goal once material is on the machine-zone gantry (last prefix step).
                "index_to_decide_goal_area": 4,
                "outbound_attached": False,
                "num_subtasks": 5,
                "finished": [False, True, True],
                "subtasks": [
                    ["go_to_processing_machine", "none", "wait"],
                    ["control_machine", "none", "process"],
                    ["wait", "finding_free_gantry", "wait"],
                    ["control_gantry", "go_to_processing_machine", "wait"],
                    ["material_on_gantry", "wait", "wait"],
                ],
                "material_states_in_subtasks": {
                    "product_00_pipe_raw": ["on_machine"] * 2 + ["disappear"] * 3,
                    "product_00_pipe": ["disappear"] * 2 + ["on_machine"] * 3,
                    "product_00_flange": ["on_start_area"] * 5,
                    "product_00_elbow": ["on_start_area"] * 5,
                    "product_00_semi": ["disappear"] * 5,
                    "product_00_maded": ["disappear"] * 5,
                },
            },
            # Appended when zone(machine) == zone(goal). Reuses the prefix gantry.
            "outbound_same_zone":{
                "num_subtasks": 3,
                "finished": [False, False, True],
                "subtasks": [
                    ["go_to_goal_area", "carry_to_goal_area", "done"],
                    ["material_on_goal_area", "wait", "done"],
                    ["done", "done", "done"],
                ],
                "material_states_in_subtasks": {
                    "product_00_pipe_raw": ["disappear"] * 3,
                    "product_00_pipe": ["on_gantry", "on_gantry", "on_goal_area"],
                    "product_00_flange": ["on_start_area"] * 3,
                    "product_00_elbow": ["on_start_area"] * 3,
                    "product_00_semi": ["disappear"] * 3,
                    "product_00_maded": ["disappear"] * 3,
                },
            },
            # Appended when zones differ: start gantry loads AGV, releases (done),
            # AGV carries alone, then finding_free_gantry for goal-zone crane.
            "outbound_cross_zone":{
                "num_subtasks": 9,
                "finished": [False, False, True, False],
                "subtasks": [
                    # human, gantry, machine, robot
                    ["control_gantry", "carry_to_robot", "done", "wait"],
                    ["material_on_robot", "wait", "done", "wait"],
                    # release start gantry (done → free); AGV carries; gantry col idle until re-acquire
                    ["go_to_goal_area", "done", "done", "carry_to_goal_area"],
                    ["wait", "finding_free_gantry", "done", "wait"],
                    # goal-zone crane moves to goal AGV parking (not start machine)
                    ["control_gantry", "go_to_goal_robot", "done", "done"],
                    ["material_on_gantry", "wait", "done", "done"],
                    ["control_gantry", "carry_to_goal_area", "done", "done"],
                    ["material_on_goal_area", "wait", "done", "done"],
                    ["done", "done", "done", "done"],
                ],
                "material_states_in_subtasks": {
                    "product_00_pipe_raw": ["disappear"] * 9,
                    "product_00_pipe": (
                        ["on_gantry", "on_gantry", "on_robot", "on_robot"]
                        + ["on_robot", "on_gantry", "on_gantry", "on_gantry", "on_goal_area"]
                    ),
                    "product_00_flange": ["on_start_area"] * 9,
                    "product_00_elbow": ["on_start_area"] * 9,
                    "product_00_semi": ["disappear"] * 9,
                    "product_00_maded": ["disappear"] * 9,
                },
            },
        },
    }
}

# task id 3 — logistic_for_pipe_grooving
CfgSubtaskGallery["ProductWaterPipe"]["logistic_for_pipe_grooving"] = copy.deepcopy(
    CfgSubtaskGallery["ProductWaterPipe"]["logistic_for_pipe_cutting"]
)
for _mode in ("have_AGV", "only_have_gantry"):
    _logistic = CfgSubtaskGallery["ProductWaterPipe"]["logistic_for_pipe_grooving"][_mode]
    _logistic["material_goal_area"] = "num04_groovingMachineLarge"
    _logistic["goal_area_ids"] = CfgMachine["num04_groovingMachineLarge"]["working_area_ids"]
    _logistic["required_logistic_material"] = "product_00_pipe"
    n = _logistic["num_subtasks"]
    pipe_states = _logistic["material_states_in_subtasks"]["product_00_pipe_raw"]
    _logistic["material_states_in_subtasks"] = {
        "product_00_pipe_raw": ["disappear"] * n,
        "product_00_pipe": pipe_states,
        "product_00_flange": ["on_start_area"] * n,
        "product_00_elbow": ["on_start_area"] * n,
        "product_00_semi": ["disappear"] * n,
        "product_00_maded": ["disappear"] * n,
    }

# task id 4 — pipe_grooving (processing: prefix + outbound tails)
CfgSubtaskGallery["ProductWaterPipe"]["pipe_grooving"] = copy.deepcopy(
    CfgSubtaskGallery["ProductWaterPipe"]["pipe_cutting"]
)
_proc = CfgSubtaskGallery["ProductWaterPipe"]["pipe_grooving"]["process_prefix"]
_proc["material_start_area"] = "num04_groovingMachineLarge"
_proc["start_area_ids"] = CfgMachine["num04_groovingMachineLarge"]["working_area_ids"]
_proc["required_processing_material"] = "product_00_pipe"
_proc["processed_material"] = "product_00_pipe"
n_pre = _proc["num_subtasks"]
_proc["material_states_in_subtasks"] = {
    "product_00_pipe_raw": ["disappear"] * n_pre,
    "product_00_pipe": ["on_machine"] * n_pre,
    "product_00_flange": ["on_start_area"] * n_pre,
    "product_00_elbow": ["on_start_area"] * n_pre,
    "product_00_semi": ["disappear"] * n_pre,
    "product_00_maded": ["disappear"] * n_pre,
}
# outbound tails: cut-pipe stays as the moving material (reuse pipe_cutting outbound shapes)
for _tail_key in ("outbound_same_zone", "outbound_cross_zone"):
    _tail = CfgSubtaskGallery["ProductWaterPipe"]["pipe_grooving"][_tail_key]
    n = _tail["num_subtasks"]
    pipe_moving = _tail["material_states_in_subtasks"]["product_00_pipe"]
    _tail["material_states_in_subtasks"] = {
        "product_00_pipe_raw": ["disappear"] * n,
        "product_00_pipe": pipe_moving,
        "product_00_flange": ["on_start_area"] * n,
        "product_00_elbow": ["on_start_area"] * n,
        "product_00_semi": ["disappear"] * n,
        "product_00_maded": ["disappear"] * n,
    }

# task id 5 — logistic_for_batch_spot_welding
CfgSubtaskGallery["ProductWaterPipe"]["logistic_for_batch_spot_welding"] = copy.deepcopy(
    CfgSubtaskGallery["ProductWaterPipe"]["logistic_for_pipe_cutting"]
)
for _mode in ("have_AGV", "only_have_gantry"):
    _logistic = CfgSubtaskGallery["ProductWaterPipe"]["logistic_for_batch_spot_welding"][_mode]
    _logistic["material_goal_area"] = "num08_workbench"
    _logistic["goal_area_ids"] = CfgMachine["num08_workbench"]["working_area_ids"]
    _logistic["required_logistic_material"] = "product_00_pipe"
    n = _logistic["num_subtasks"]
    pipe_states = _logistic["material_states_in_subtasks"]["product_00_pipe_raw"]
    _logistic["material_states_in_subtasks"] = {
        "product_00_pipe_raw": ["disappear"] * n,
        "product_00_pipe": pipe_states,
        "product_00_flange": ["on_start_area"] * n,
        "product_00_elbow": ["on_start_area"] * n,
        "product_00_semi": ["disappear"] * n,
        "product_00_maded": ["disappear"] * n,
    }

# task id 6 — batch_spot_welding
CfgSubtaskGallery["ProductWaterPipe"]["batch_spot_welding"] = copy.deepcopy(
    CfgSubtaskGallery["ProductWaterPipe"]["pipe_cutting"]
)
_proc = CfgSubtaskGallery["ProductWaterPipe"]["batch_spot_welding"]["process_prefix"]
_proc["material_start_area"] = "num08_workbench"
_proc["required_processing_material"] = ["product_00_pipe", "product_00_flange", "product_00_elbow"]
_proc["processed_material"] = "product_00_semi"
_proc["start_area_ids"] = CfgMachine["num08_workbench"]["working_area_ids"]
n_pre = _proc["num_subtasks"]
_proc["material_states_in_subtasks"] = {
    "product_00_pipe_raw": ["disappear"] * n_pre,
    "product_00_pipe": ["on_machine"] * 2 + ["disappear"] * (n_pre - 2),
    "product_00_flange": ["on_start_area"] * 2 + ["disappear"] * (n_pre - 2),
    "product_00_elbow": ["on_start_area"] * 2 + ["disappear"] * (n_pre - 2),
    "product_00_semi": ["disappear"] * 2 + ["on_machine"] * (n_pre - 2),
    "product_00_maded": ["disappear"] * n_pre,
}
for _tail_key in ("outbound_same_zone", "outbound_cross_zone"):
    _tail = CfgSubtaskGallery["ProductWaterPipe"]["batch_spot_welding"][_tail_key]
    n = _tail["num_subtasks"]
    moving = _tail["material_states_in_subtasks"]["product_00_pipe"]
    _tail["material_states_in_subtasks"] = {
        "product_00_pipe_raw": ["disappear"] * n,
        "product_00_pipe": ["disappear"] * n,
        "product_00_flange": ["disappear"] * n,
        "product_00_elbow": ["disappear"] * n,
        "product_00_semi": moving,
        "product_00_maded": ["disappear"] * n,
    }

# task id 7 — logistic_for_arc_welding_root
CfgSubtaskGallery["ProductWaterPipe"]["logistic_for_arc_welding_root"] = copy.deepcopy(
    CfgSubtaskGallery["ProductWaterPipe"]["logistic_for_pipe_cutting"]
)
for _mode in ("have_AGV", "only_have_gantry"):
    _logistic = CfgSubtaskGallery["ProductWaterPipe"]["logistic_for_arc_welding_root"][_mode]
    _logistic["material_goal_area"] = "num01_weldingRobot"
    _logistic["required_logistic_material"] = "product_00_semi"
    n = _logistic["num_subtasks"]
    if _mode == "have_AGV":
        _logistic["material_states_in_subtasks"] = {
            "product_00_pipe_raw": ["disappear"] * n,
            "product_00_pipe": ["disappear"] * n,
            "product_00_flange": ["disappear"] * n,
            "product_00_elbow": ["disappear"] * n,
            "product_00_semi": (
                ["on_start_area", "on_start_area", "on_gantry", "on_gantry"]
                + ["on_robot", "on_robot", "on_robot"]
                + ["on_gantry", "on_gantry", "on_gantry", "on_goal_area"]
            ),
            "product_00_maded": ["disappear"] * n,
        }
    else:
        _logistic["material_states_in_subtasks"] = {
            "product_00_pipe_raw": ["disappear"] * n,
            "product_00_pipe": ["disappear"] * n,
            "product_00_flange": ["disappear"] * n,
            "product_00_elbow": ["disappear"] * n,
            "product_00_semi": ["on_start_area", "on_start_area", "on_gantry", "on_gantry", "on_goal_area"],
            "product_00_maded": ["disappear"] * n,
        }
    _logistic["goal_area_ids"] = CfgMachine["num01_weldingRobot"]["working_area_ids"]

# task id 8 — arc_welding_root
CfgSubtaskGallery["ProductWaterPipe"]["arc_welding_root"] = copy.deepcopy(
    CfgSubtaskGallery["ProductWaterPipe"]["batch_spot_welding"]
)
_proc = CfgSubtaskGallery["ProductWaterPipe"]["arc_welding_root"]["process_prefix"]
_proc["material_start_area"] = "num01_weldingRobot"
_proc["required_processing_material"] = "product_00_semi"
_proc["start_area_ids"] = CfgMachine["num01_weldingRobot"]["working_area_ids"]
n_pre = _proc["num_subtasks"]
_proc["material_states_in_subtasks"] = {
    "product_00_pipe_raw": ["disappear"] * n_pre,
    "product_00_pipe": ["disappear"] * n_pre,
    "product_00_flange": ["disappear"] * n_pre,
    "product_00_elbow": ["disappear"] * n_pre,
    "product_00_semi": ["on_machine"] * n_pre,
    "product_00_maded": ["disappear"] * n_pre,
}
for _tail_key in ("outbound_same_zone", "outbound_cross_zone"):
    _tail = CfgSubtaskGallery["ProductWaterPipe"]["arc_welding_root"][_tail_key]
    n = _tail["num_subtasks"]
    moving = _tail["material_states_in_subtasks"]["product_00_semi"]
    _tail["material_states_in_subtasks"] = {
        "product_00_pipe_raw": ["disappear"] * n,
        "product_00_pipe": ["disappear"] * n,
        "product_00_flange": ["disappear"] * n,
        "product_00_elbow": ["disappear"] * n,
        "product_00_semi": moving,
        "product_00_maded": ["disappear"] * n,
    }

# task id 9 — logistic_for_MIG_welding_surface
CfgSubtaskGallery["ProductWaterPipe"]["logistic_for_MIG_welding_surface"] = copy.deepcopy(
    CfgSubtaskGallery["ProductWaterPipe"]["logistic_for_arc_welding_root"]
)
for _mode in ("have_AGV", "only_have_gantry"):
    _logistic = CfgSubtaskGallery["ProductWaterPipe"]["logistic_for_MIG_welding_surface"][_mode]
    _logistic["material_goal_area"] = "num00_rotaryPipeAutomaticWeldingMachine"
    _logistic["goal_area_ids"] = CfgMachine["num00_rotaryPipeAutomaticWeldingMachine"]["working_area_ids"]

# task id 10 — MIG_welding_surface
CfgSubtaskGallery["ProductWaterPipe"]["MIG_welding_surface"] = copy.deepcopy(
    CfgSubtaskGallery["ProductWaterPipe"]["arc_welding_root"]
)
_proc = CfgSubtaskGallery["ProductWaterPipe"]["MIG_welding_surface"]["process_prefix"]
_proc["material_start_area"] = "num00_rotaryPipeAutomaticWeldingMachine"
_proc["start_area_ids"] = CfgMachine["num00_rotaryPipeAutomaticWeldingMachine"]["working_area_ids"]

# task id 11 — logistic_for_paint_rust_proof
CfgSubtaskGallery["ProductWaterPipe"]["logistic_for_paint_rust_proof"] = copy.deepcopy(
    CfgSubtaskGallery["ProductWaterPipe"]["logistic_for_arc_welding_root"]
)
for _mode in ("have_AGV", "only_have_gantry"):
    _logistic = CfgSubtaskGallery["ProductWaterPipe"]["logistic_for_paint_rust_proof"][_mode]
    _logistic["material_goal_area"] = "num08_workbench"
    _logistic["goal_area_ids"] = CfgMachine["num08_workbench"]["working_area_ids"]

# task id 12 — paint_rust_proof
CfgSubtaskGallery["ProductWaterPipe"]["paint_rust_proof"] = copy.deepcopy(
    CfgSubtaskGallery["ProductWaterPipe"]["arc_welding_root"]
)
_proc = CfgSubtaskGallery["ProductWaterPipe"]["paint_rust_proof"]["process_prefix"]
_proc["material_start_area"] = "num08_workbench"
_proc["processed_material"] = "product_00_maded"
_proc["start_area_ids"] = CfgMachine["num08_workbench"]["working_area_ids"]
n_pre = _proc["num_subtasks"]
_proc["material_states_in_subtasks"] = {
    "product_00_pipe_raw": ["disappear"] * n_pre,
    "product_00_pipe": ["disappear"] * n_pre,
    "product_00_flange": ["disappear"] * n_pre,
    "product_00_elbow": ["disappear"] * n_pre,
    "product_00_semi": ["on_machine"] * 2 + ["disappear"] * (n_pre - 2),
    "product_00_maded": ["disappear"] * 2 + ["on_machine"] * (n_pre - 2),
}
for _tail_key in ("outbound_same_zone", "outbound_cross_zone"):
    _tail = CfgSubtaskGallery["ProductWaterPipe"]["paint_rust_proof"][_tail_key]
    n = _tail["num_subtasks"]
    moving = _tail["material_states_in_subtasks"]["product_00_semi"]
    _tail["material_states_in_subtasks"] = {
        "product_00_pipe_raw": ["disappear"] * n,
        "product_00_pipe": ["disappear"] * n,
        "product_00_flange": ["disappear"] * n,
        "product_00_elbow": ["disappear"] * n,
        "product_00_semi": ["disappear"] * n,
        "product_00_maded": moving,
    }
