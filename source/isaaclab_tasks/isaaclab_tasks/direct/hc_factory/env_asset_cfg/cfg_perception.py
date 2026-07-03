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
    "enabled": False,
    # collect: 仿真中采集 | infer: 仿真中推理 | off: 关闭
    "mode": "off",
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
