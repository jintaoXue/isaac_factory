"""Perception data collection & training configuration."""

from pathlib import Path

from .cfg_process_subtask_gallery import CfgSubtaskPredefinedTimeGallery

# agent column index in subtasks row: human=0, gantry=1, machine=2, robot=3
AGENT_COL_HUMAN = 0
AGENT_COL_GANTRY = 1
AGENT_COL_MACHINE = 2
AGENT_COL_ROBOT = 3

_DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parent.parent / "output" / "perception_dataset"
)

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

# 单步采集样本模板（logger 按此结构写入 meta.jsonl）
PerceptionSampleTemplate = {
    "episode_id": 0,
    "step_id": 0,
    "time_step": 0,
    "env_id": 0,
    # 图像路径相对 episode 目录，如 cameras/step_000123/camera_xxx.jpg
    "camera_paths": {},
    # 结构化文本上下文（供训练 / VLM prompt）
    "text_context": "",
    # GT：每个 human 的 subtask 进度
    "human_labels": {
        # "num_00_NormalHuman": {
        #     "human_index": 0,
        #     "state": "free",
        #     "ongoing_task_record_index": None,
        #     "current_area_id": 56,
        #     "target_area_id": None,
        #     "task": None,
        #     "task_type": None,
        #     "subtask_index": None,
        #     "subtask_name": None,
        #     "subtask_done": None,
        #     "animation_pose": "idle",
        # }
    },
    # 可直接读取的机器/机器人信号（训练时可作 privileged input）
    "agent_signals": {
        # product_index: {
        #     "task": "logistic_for_pipe_cutting",
        #     "subtask_index": 2,
        #     "ongoing_row": ["control_gantry", "carry_to_robot", "wait", "wait"],
        #     "finished": [False, True, True, False],
        # }
    },
    # 完整 task record（JSON-serializable 子集）
    "task_records": {},
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
    "serialize_task_records": True,
    "build_text_context": True,
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
    # 训练目标
    "predict_subtask_name": True,
    "predict_subtask_done": True,
    "use_agent_signals": True,
    "use_text_features": False,
    "temporal_window": 1,
    "image_size": 224,
    "backbone": "resnet18",
    "subtask_vocab": sorted(
        k for k in CfgSubtaskPredefinedTimeGallery if k not in ("done", "none")
    ),
}
