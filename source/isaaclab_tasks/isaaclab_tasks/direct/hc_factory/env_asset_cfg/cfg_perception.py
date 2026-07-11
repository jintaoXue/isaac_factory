"""Perception data collection & training configuration."""

from pathlib import Path

from .cfg_process_task_gallery import CfgProcessTaskGalleryInAll

# agent column index in subtasks row: human=0, gantry=1, machine=2, robot=3
AGENT_COL_HUMAN = 0
AGENT_COL_GANTRY = 1
AGENT_COL_MACHINE = 2
AGENT_COL_ROBOT = 3

_DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parent.parent / "output" / "perception_dataset"
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
#
# 同一 subtasks 行里，不同列可以有不同名字；不能用一个全局同义词表强行对齐。
# 仅当 (human_subtask, partner_col, partner_subtask) 匹配时，
# 且 partner 列 finished=True，才推断 human 当前 subtask 为 done。
#
# logistic_for_pipe_cutting have_AGV 示例:
#   row2: human=control_gantry,  gantry=carry_to_robot        -> 耦合
#   row4: human=go_to_goal_area, gantry=move_to_goal_area     -> 不耦合（并行到达）
#   row6: human=control_gantry,  gantry=carry_to_goal_area    -> 耦合
#
# only_have_gantry:
#   row2: human=go_to_goal_area, gantry=carry_to_goal_area    -> 不耦合
#
# pipe_cutting:
#   row1: human=control_machine, machine=process              -> 耦合
#   row2: human=wait,            gantry=finding_free_gantry   -> 不耦合
#   row3: human=control_gantry,  gantry=go_to_processing_machine -> 耦合
#   row5: human=control_gantry,  gantry=carry_to_goal_area    -> 耦合
# ---------------------------------------------------------------------------
CoupledDoneRules: list[tuple[str, int, str]] = [
    ("control_gantry", AGENT_COL_GANTRY, "carry_to_robot"),
    ("control_gantry", AGENT_COL_GANTRY, "carry_to_goal_area"),
    ("control_gantry", AGENT_COL_GANTRY, "go_to_processing_machine"),
    ("control_machine", AGENT_COL_MACHINE, "process"),
]

# 以下同名/异名组合为并行任务，伙伴 finished 不能推断 human done
IndependentSubtaskRows: list[tuple[str, int, str]] = [
    ("go_to_material", AGENT_COL_GANTRY, "go_to_material"),
    ("go_to_material", AGENT_COL_ROBOT, "go_to_material"),
    ("go_to_goal_area", AGENT_COL_GANTRY, "move_to_goal_area"),
    ("go_to_goal_area", AGENT_COL_GANTRY, "carry_to_goal_area"),
    ("go_to_goal_area", AGENT_COL_ROBOT, "carry_to_goal_area"),
    ("wait", AGENT_COL_GANTRY, "finding_free_gantry"),
]

# meta.jsonl：一行 = 一个 working human 的一个样本（free 不写入）
PerceptionSampleTemplate = {
    "episode_id": 0,
    "step_id": 0,
    "time_step": 0,
    "env_id": 0,
    "human_key": "num_00_NormalHuman",
    "human_index": 0,
    "product_index": 0,
    "task": "logistic_for_pipe_cutting",
    "subtask_index": 0,
    "subtask_name": "go_to_material",
    "subtask_done": False,
    "area_id": 202,
    # 同一步各 human 样本共享相机路径
    "camera_paths": {},
    # 该 human 对应 product 的 task-record 信号（训练可选输入）
    "agent_signal": {
        "subtask_index": 0,
        "num_subtasks": 9,
        "ongoing_row": ["go_to_material", "go_to_material", "wait", "go_to_material"],
        "finished": [False, False, False, False],
    },
    "text_context": "",
}

CfgPerception = {
    "enabled": True,
    # collect: 仿真中采集 | infer: 仿真中推理 | off: 关闭
    "mode": "collect",
    "output_dir": str(_DEFAULT_OUTPUT_DIR),
    # 每 N 个 env step 存一帧（与 camera_capture_interval 对齐时可设为相同值）
    "save_interval": 1,
    "max_episodes": 200,
    "max_steps_per_episode": None,
    "save_images": True,
    "image_format": "jpg",
    "image_quality": 90,
    "build_text_context": False,
    # infer 模式加载的 checkpoint
    "checkpoint_path": None,
    "use_constraint_propagation": True,
}

CfgPerceptionTraining = {
    "dataset_dir": str(_DEFAULT_OUTPUT_DIR),
    "output_dir": str(_DEFAULT_OUTPUT_DIR.parent / "perception_runs"),
    "run_name": "subtask_baseline",
    "batch_size": 16,
    "num_epochs": 20,
    "learning_rate": 1e-4,
    "weight_decay": 1e-5,
    "val_ratio": 0.15,
    "num_workers": 4,
    "device": "cuda:0",
    # 训练目标：仅预测 human subtask 是否完成
    "predict_subtask_done": True,
    "use_agent_signals": True,
    "signal_dim": 6,
    "image_size": 224,
}
