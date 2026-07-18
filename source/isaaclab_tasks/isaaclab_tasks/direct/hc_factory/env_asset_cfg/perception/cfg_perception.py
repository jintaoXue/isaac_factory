"""Perception data collection & training configuration."""

from pathlib import Path

from ..cfg_process_task_gallery import CfgProcessTaskGalleryInAll

# agent column index in subtasks row: human=0, gantry=1, machine=2, robot=3
AGENT_COL_HUMAN = 0
AGENT_COL_GANTRY = 1
AGENT_COL_MACHINE = 2
AGENT_COL_ROBOT = 3

_DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parent.parent.parent / "output" / "perception_dataset"
)

# ---------------------------------------------------------------------------
# Perception 标签集合（直接枚举，与 cfg_process_*_gallery 一致）
# 目标：推断每个 human 当前的 task + subtask
# ---------------------------------------------------------------------------

# human 无任务时 human_labels.state = "free"，task/subtask_name 为 null

# --- human 列 subtask（全集）---
HumanSubtaskLabels = [
    "go_to_material",
    "material_on_gantry",
    "control_gantry",
    "material_on_robot",
    "go_to_goal_area",
    "material_on_goal_area",
    "go_to_processing_machine",
    "control_machine",
    "wait",
    "done",
]

# 训练分类 vocab（排除终端 done）
HumanSubtaskVocab = [
    "go_to_material",
    "material_on_gantry",
    "control_gantry",
    "material_on_robot",
    "go_to_goal_area",
    "material_on_goal_area",
    "go_to_processing_machine",
    "control_machine",
    "wait",
]

# --- 工艺 task（与 CfgProcessTaskGalleryInAll 同步）---
TaskLabelToIndex: dict[str, int] = dict(CfgProcessTaskGalleryInAll)
TaskLabels: list[str] = sorted(TaskLabelToIndex, key=TaskLabelToIndex.get)
# 训练用：排除 none，free human 的 GT task 为 null
TaskVocab: list[str] = [t for t in TaskLabels if t != "none"]

# --- 伙伴列 subtask（供 agent_signals 参考，非 human 预测目标）---
GantrySubtaskLabels = [
    "go_to_material",
    "carry_to_robot",
    "carry_to_goal_area",
    "move_to_goal_area",
    "go_to_processing_machine",
    "finding_free_gantry",
    "wait",
    "done",
    "none",
]

MachineSubtaskLabels = [
    "process",
    "wait",
    "done",
    "none",
]

RobotSubtaskLabels = [
    "go_to_material",
    "carry_to_goal_area",
    "wait",
    "done",
]

# ---------------------------------------------------------------------------
# 跨智能体 subtask 耦合规则（来自 cfg_process_subtask_gallery.py）
# ---------------------------------------------------------------------------
CoupledDoneRules: list[tuple[str, int, str]] = [
    ("control_gantry", AGENT_COL_GANTRY, "carry_to_robot"),
    ("control_gantry", AGENT_COL_GANTRY, "carry_to_goal_area"),
    ("control_gantry", AGENT_COL_GANTRY, "go_to_processing_machine"),
    ("control_machine", AGENT_COL_MACHINE, "process"),
]

IndependentSubtaskRows: list[tuple[str, int, str]] = [
    ("go_to_material", AGENT_COL_GANTRY, "go_to_material"),
    ("go_to_material", AGENT_COL_ROBOT, "go_to_material"),
    ("go_to_goal_area", AGENT_COL_GANTRY, "move_to_goal_area"),
    ("go_to_goal_area", AGENT_COL_GANTRY, "carry_to_goal_area"),
    ("go_to_goal_area", AGENT_COL_ROBOT, "carry_to_goal_area"),
    ("wait", AGENT_COL_GANTRY, "finding_free_gantry"),
]




ImageSampleTemplate = {
    "highrise_for_env": {
        "camera_num00_highrise_for_env": {
            "image": "path/to/image.jpg",
        },
        "camera_num01_highrise_for_env": {
            "image": "path/to/image.jpg",
        },
    },
    "storage_area": {
        "camera_num00_storage_area": {
            "image": "path/to/image.jpg",
        },
    },
    "num00_rotaryPipeAutomaticWeldingMachine": {
        "camera_num00_rotaryPipeAutomaticWeldingMachine_part_01_station": {
            "image": "path/to/image.jpg",
        },
        "camera_num00_rotaryPipeAutomaticWeldingMachine_part_02_station": {
            "image": "path/to/image.jpg",
        },
    },
    "num01_weldingRobot": {
        "camera_num01_weldingRobot_part02_robot_arm_and_base": {
            "image": "path/to/image.jpg",
        },
    },
    "num02_rollerbedCNCPipeIntersectionCuttingMachine": {
        "camera_num02_rollerbedCNCPipeIntersectionCuttingMachine_part01_station": {
            "image": "path/to/image.jpg",
        },
    },
    "num04_groovingMachineLarge": {
        "camera_num04_groovingMachineLarge_part01_large_fixed_base": {
            "image": "path/to/image.jpg",
        },
    },
    "num08_workbench": {
        "camera_num08_workbench": {
            "image": "path/to/image.jpg",
        },
    },
}




# will storage in meta.jsonl file (one line = one env step)
PerceptionSampleTemplate = {
    "episode_id": 0,
    "step_id": 0,
    "time_step": 0,
    "env_id": 0,
    "human_id_recognition": {
        "input": {
            "task": "logistic_for_pipe_cutting",
            "images": ImageSampleTemplate,
        },
        "output_label": {
            "highrise_for_env": {
                "camera_num00_highrise_for_env": {"human_ids": []},
                "camera_num01_highrise_for_env": {"human_ids": []},
            },
            "storage_area": {
                "camera_num00_storage_area": {"human_ids": []},
            },
            "num00_rotaryPipeAutomaticWeldingMachine": {
                "camera_num00_rotaryPipeAutomaticWeldingMachine_part_01_station": {"human_ids": []},
                "camera_num00_rotaryPipeAutomaticWeldingMachine_part_02_station": {"human_ids": []},
            },
            "num01_weldingRobot": {
                "camera_num01_weldingRobot_part02_robot_arm_and_base": {"human_ids": []},
            },
            "num02_rollerbedCNCPipeIntersectionCuttingMachine": {
                "camera_num02_rollerbedCNCPipeIntersectionCuttingMachine_part01_station": {"human_ids": []},
            },
            "num04_groovingMachineLarge": {
                "camera_num04_groovingMachineLarge_part01_large_fixed_base": {"human_ids": []},
            },
            "num08_workbench": {
                "camera_num08_workbench": {"human_ids": []},
            },
        },
    },
    "human_subtask_recognition": {
        "input": {
            "images": ImageSampleTemplate,
            "human_keys": ["num_00_NormalHuman"],
            "human_task": ["logistic_for_pipe_cutting"],
            "human_task_id": [1],
        },
        "output_label": {
            "human_subtask": ["go_to_material"],
            "human_subtask_id": [0],
            "human_subtask_done": [False],
        },
    },
    # record only — not used for training
    "env_state_action_dict": {},
}

CfgPerception = {
    "enabled": True,
    "mode": "collect",
    "output_dir": str(_DEFAULT_OUTPUT_DIR),
    "save_interval": 1,
    # 6 episodes → train/val/test = 4 / 1 / 1（每 episode 约 3000+ 步，总量已够）
    "max_episodes": 6,
    "max_steps_per_episode": None,
    "save_images": True,
    "image_format": "jpg",
    "image_quality": 90,
    "save_env_record": True,
    "checkpoint_path": None,
    "use_constraint_propagation": True,
}

CfgPerceptionTraining = {
    "dataset_dir": str(_DEFAULT_OUTPUT_DIR),
    "output_dir": str(_DEFAULT_OUTPUT_DIR.parent / "perception_runs"),
    "run_name": "perception_baseline",
    # id | subtask
    "task": "subtask",
    "batch_size": 16,
    "num_epochs": 20,
    "learning_rate": 1e-4,
    "weight_decay": 1e-5,
    # 6 episodes → 4 / 1 / 1
    "train_ratio": 4 / 6,
    "val_ratio": 1 / 6,
    "test_ratio": 1 / 6,
    "split_seed": 42,
    "num_workers": 4,
    "device": "cuda:0",
    "image_size": 224,
}
