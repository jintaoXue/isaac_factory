"""Human subtask perception: simulation data logging and offline training.

Modes
-----
- **collect**: hooked into env via ``env_state_action_dict`` (PerceptionManager.step/reset)
- **train / eval**: standalone ``python -m ...`` without Isaac Sim

Example
-------
Collect (enable in cfg_perception.py, set mode="collect", enabled=True), then train::

    python source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/src/perception.py train \\
        --dataset_dir output/perception_dataset
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

# Allow standalone training: python .../src/perception.py train
_PKG_ROOT = Path(__file__).resolve().parents[4]
if _PKG_ROOT.name == "isaaclab_tasks" and str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split

try:
    from ..env_asset_cfg.cfg_perception import (
        AGENT_COL_GANTRY,
        AGENT_COL_HUMAN,
        AGENT_COL_MACHINE,
        AGENT_COL_ROBOT,
        CfgPerception,
        CfgPerceptionTraining,
        PerceptionSampleTemplate,
    )
except ImportError:
    from isaaclab_tasks.direct.hc_factory.env_asset_cfg.cfg_perception import (  # type: ignore[no-redef]
        AGENT_COL_GANTRY,
        AGENT_COL_HUMAN,
        AGENT_COL_MACHINE,
        AGENT_COL_ROBOT,
        CfgPerception,
        CfgPerceptionTraining,
        PerceptionSampleTemplate,
    )

# ---------------------------------------------------------------------------
# Serialization & label extraction
# ---------------------------------------------------------------------------

_SUBTASK_SYNONYMS = {
    "carry_to_robot": "control_gantry",
    "move_to_goal_area": "control_gantry",
    "carry_to_goal_area": "go_to_goal_area",
    "finding_free_gantry": "control_gantry",
}


def to_serializable(obj: Any) -> Any:
    """Recursively convert env_state values to JSON-friendly types."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, dict):
        return {str(k): to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    return str(obj)


def _normalize_subtask_name(name: str | None) -> str | None:
    if name is None:
        return None
    return _SUBTASK_SYNONYMS.get(name, name)


def extract_human_labels(env_state_action_dict: dict) -> dict[str, dict]:
    """Ground-truth human subtask labels from env_state_action_dict."""
    labels: dict[str, dict] = {}
    ongoing_records = env_state_action_dict["progress"]["ongoing_task_records"]

    for human_key, human_state in env_state_action_dict["human"].items():
        label = {
            "human_index": human_state["key_variables"]["idx"],
            "state": human_state["state"],
            "ongoing_task_record_index": human_state["ongoing_task_record_index"],
            "current_area_id": human_state["current_area_id"],
            "target_area_id": human_state["target_area_id"],
            "subtask_time_counter": human_state.get("subtask_time_counter", 0),
            "task": None,
            "task_type": None,
            "product_index": None,
            "subtask_index": None,
            "subtask_name": None,
            "subtask_done": None,
            "animation_pose": None,
        }

        record_idx = human_state["ongoing_task_record_index"]
        if record_idx is not None and record_idx in ongoing_records:
            task_record = ongoing_records[record_idx]
            subtasks_dict = task_record["subtasks_dict"]
            ongoing_row = subtasks_dict["ongoing"]
            finished = subtasks_dict["finished"]
            label.update(
                {
                    "task": task_record.get("task"),
                    "task_type": task_record.get("task_type"),
                    "product_index": record_idx,
                    "subtask_index": subtasks_dict.get("ongoing_index"),
                    "subtask_name": _normalize_subtask_name(ongoing_row[AGENT_COL_HUMAN]),
                    "subtask_done": bool(finished[AGENT_COL_HUMAN]),
                }
            )
            if label["state"].startswith("working_"):
                label["animation_pose"] = _infer_animation_pose(label["subtask_name"])

        labels[human_key] = label
    return labels


def _infer_animation_pose(subtask_name: str | None) -> str | None:
    if subtask_name is None:
        return None
    if subtask_name in ("go_to_material", "go_to_goal_area", "go_to_processing_machine"):
        return "walk"
    if subtask_name in (
        "material_on_gantry",
        "control_gantry",
        "material_on_robot",
        "material_on_goal_area",
        "control_machine",
    ):
        return "operate"
    return "idle"


def extract_agent_signals(env_state_action_dict: dict) -> dict[str, dict]:
    """Privileged multi-agent subtask row signals from ongoing task records."""
    signals: dict[str, dict] = {}
    for product_index, task_record in env_state_action_dict["progress"]["ongoing_task_records"].items():
        subtasks_dict = task_record["subtasks_dict"]
        signals[str(product_index)] = {
            "task": task_record.get("task"),
            "task_type": task_record.get("task_type"),
            "human_key": task_record.get("human"),
            "robot_key": task_record.get("robot"),
            "subtask_index": subtasks_dict.get("ongoing_index"),
            "num_subtasks": subtasks_dict.get("num_subtasks"),
            "ongoing_row": list(subtasks_dict.get("ongoing", [])),
            "finished": [bool(x) for x in subtasks_dict.get("finished", [])],
            "material_start_area": subtasks_dict.get("material_start_area"),
            "material_goal_area": subtasks_dict.get("material_goal_area"),
        }
    return signals


def extract_task_records(env_state_action_dict: dict) -> dict:
    """JSON-serializable subset of ongoing task records."""
    records: dict[str, dict] = {}
    for product_index, task_record in env_state_action_dict["progress"]["ongoing_task_records"].items():
        subtasks_dict = task_record.get("subtasks_dict") or {}
        records[str(product_index)] = to_serializable(
            {
                "task": task_record.get("task"),
                "task_type": task_record.get("task_type"),
                "product": task_record.get("product"),
                "product_index": task_record.get("product_index"),
                "human": task_record.get("human"),
                "human_index": task_record.get("human_index"),
                "robot": task_record.get("robot"),
                "target_machine": task_record.get("target_machine"),
                "chosen_machine_workstation": task_record.get("chosen_machine_workstation"),
                "chosen_gantry_index": task_record.get("chosen_gantry_index"),
                "subtasks_dict": {
                    "ongoing_index": subtasks_dict.get("ongoing_index"),
                    "num_subtasks": subtasks_dict.get("num_subtasks"),
                    "ongoing": subtasks_dict.get("ongoing"),
                    "finished": subtasks_dict.get("finished"),
                    "material_start_area": subtasks_dict.get("material_start_area"),
                    "material_goal_area": subtasks_dict.get("material_goal_area"),
                },
            }
        )
    return records


def build_text_context(
    env_state_action_dict: dict,
    human_labels: dict,
    agent_signals: dict,
) -> str:
    """Compact textual context for multimodal models."""
    lines = [f"time_step={env_state_action_dict['time_step']}"]
    for human_key, label in human_labels.items():
        if label["state"] == "free":
            lines.append(f"{human_key}: free")
            continue
        lines.append(
            f"{human_key}: state={label['state']} task={label['task']} "
            f"subtask[{label['subtask_index']}]={label['subtask_name']} "
            f"done={label['subtask_done']} area={label['current_area_id']}"
        )
    for product_index, sig in agent_signals.items():
        lines.append(
            f"task_record[{product_index}]: {sig['task']} row={sig['ongoing_row']} finished={sig['finished']}"
        )
    return "\n".join(lines)


def apply_constraint_propagation(
    human_subtask_name: str | None,
    human_subtask_done: bool,
    agent_signals: dict[str, dict],
    human_key: str,
) -> tuple[str | None, bool]:
    """Use gantry/robot/machine finished flags to refine human subtask phase."""
    if human_subtask_name is None:
        return human_subtask_name, human_subtask_done

    for sig in agent_signals.values():
        if sig.get("human_key") != human_key:
            continue
        row = sig.get("ongoing_row") or []
        finished = sig.get("finished") or []
        if len(row) <= AGENT_COL_HUMAN:
            continue

        human_name = _normalize_subtask_name(row[AGENT_COL_HUMAN])
        if human_name != human_subtask_name:
            continue

        # coupled subtasks: gantry done -> human control_gantry done
        if human_subtask_name == "control_gantry" and len(row) > AGENT_COL_GANTRY:
            gantry_name = _normalize_subtask_name(row[AGENT_COL_GANTRY])
            if gantry_name in ("control_gantry", "carry_to_robot", "move_to_goal_area"):
                if finished[AGENT_COL_GANTRY]:
                    human_subtask_done = True

        if human_subtask_name == "go_to_goal_area" and len(row) > AGENT_COL_ROBOT:
            robot_name = _normalize_subtask_name(row[AGENT_COL_ROBOT])
            if robot_name == "go_to_goal_area" and finished[AGENT_COL_ROBOT]:
                human_subtask_done = True

        if human_subtask_name == "go_to_material" and len(row) > AGENT_COL_ROBOT:
            robot_name = _normalize_subtask_name(row[AGENT_COL_ROBOT])
            if robot_name == "go_to_material" and finished[AGENT_COL_ROBOT]:
                # independent columns – only weak hint, do not force done
                pass

        if len(row) > AGENT_COL_MACHINE and finished[AGENT_COL_MACHINE]:
            if row[AGENT_COL_MACHINE] not in ("wait", "done", "none"):
                pass

    return human_subtask_name, human_subtask_done


# ---------------------------------------------------------------------------
# Data collection (simulation)
# ---------------------------------------------------------------------------


class PerceptionLogger:
    """Write perception samples to disk during simulation."""

    def __init__(self, env_id: int, cfg: dict):
        self.env_id = env_id
        self.cfg = cfg
        self.output_dir = Path(cfg["output_dir"])
        self.episode_id = 0
        self.step_id = 0
        self._episode_dir: Path | None = None
        self._meta_path: Path | None = None

    def reset(self) -> None:
        if self.episode_id >= self.cfg["max_episodes"]:
            return
        self._episode_dir = self.output_dir / f"env_{self.env_id:02d}_episode_{self.episode_id:06d}"
        self._episode_dir.mkdir(parents=True, exist_ok=True)
        self._meta_path = self._episode_dir / "meta.jsonl"
        self.step_id = 0
        if self.episode_id == 0:
            manifest = {
                "env_id": self.env_id,
                "sample_template": PerceptionSampleTemplate,
                "cfg": to_serializable(self.cfg),
            }
            with (self.output_dir / "manifest.json").open("w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2, ensure_ascii=False)
        self.episode_id += 1

    def maybe_log(self, env_state_action_dict: dict) -> dict | None:
        if self._episode_dir is None or self._meta_path is None:
            return None
        if self.episode_id > self.cfg["max_episodes"]:
            return None

        time_step = env_state_action_dict["time_step"]
        if time_step % self.cfg["save_interval"] != 0:
            return None

        max_steps = self.cfg.get("max_steps_per_episode")
        if max_steps is not None and self.step_id >= max_steps:
            return None

        human_labels = extract_human_labels(env_state_action_dict)
        agent_signals = extract_agent_signals(env_state_action_dict)
        task_records = extract_task_records(env_state_action_dict) if self.cfg["serialize_task_records"] else {}
        text_context = build_text_context(env_state_action_dict, human_labels, agent_signals) if self.cfg[
            "build_text_context"
        ] else ""

        camera_paths = self._save_cameras(env_state_action_dict, self.step_id)

        sample = copy.deepcopy(PerceptionSampleTemplate)
        sample.update(
            {
                "episode_id": self.episode_id - 1,
                "step_id": self.step_id,
                "time_step": time_step,
                "env_id": self.env_id,
                "camera_paths": camera_paths,
                "text_context": text_context,
                "human_labels": human_labels,
                "agent_signals": agent_signals,
                "task_records": task_records,
            }
        )

        with self._meta_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(to_serializable(sample), ensure_ascii=False) + "\n")

        self.step_id += 1
        return sample

    def _save_cameras(self, env_state_action_dict: dict, step_id: int) -> dict[str, str]:
        if not self.cfg.get("save_images", True):
            return {}

        assert self._episode_dir is not None
        step_dir = self._episode_dir / "cameras" / f"step_{step_id:06d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        rel_paths: dict[str, str] = {}
        ext = self.cfg.get("image_format", "jpg")

        for camera_key, camera_state in env_state_action_dict.get("camera", {}).items():
            rgb = camera_state.get("rgb")
            if rgb is None or not camera_state.get("is_initialized", False):
                continue
            if isinstance(rgb, torch.Tensor):
                rgb_np = rgb.detach().cpu().numpy()
            else:
                rgb_np = np.asarray(rgb)
            if rgb_np.size == 0:
                continue

            filename = f"{camera_key}.{ext}"
            out_path = step_dir / filename
            if ext.lower() in ("jpg", "jpeg"):
                Image.fromarray(rgb_np.astype(np.uint8)).save(
                    out_path, format="JPEG", quality=int(self.cfg.get("image_quality", 90))
                )
            else:
                Image.fromarray(rgb_np.astype(np.uint8)).save(out_path)

            rel_paths[camera_key] = str(out_path.relative_to(self._episode_dir))
        return rel_paths


class PerceptionManager:
    """Perception hook for HcSingleEnv – reads/writes via env_state_action_dict."""

    def __init__(self, env_id: int, cuda_device: torch.device, cfg: dict | None = None):
        self.env_id = env_id
        self.cuda_device = cuda_device
        self.cfg = copy.deepcopy(cfg or CfgPerception)
        self.mode = self.cfg.get("mode", "off")
        self.enabled = bool(self.cfg.get("enabled", False))
        self._logger: PerceptionLogger | None = None
        self._model: MultimodalSubtaskModel | None = None
        self._subtask_vocab: list[str] = list(CfgPerceptionTraining["subtask_vocab"])
        self._subtask_to_idx = {name: i for i, name in enumerate(self._subtask_vocab)}

        if self.enabled and self.mode == "collect":
            self._logger = PerceptionLogger(env_id, self.cfg)
        if self.enabled and self.mode == "infer":
            self._load_infer_model()

    def _load_infer_model(self) -> None:
        ckpt_path = self.cfg.get("checkpoint_path")
        if not ckpt_path:
            print("[WARN] Perception infer mode enabled but checkpoint_path is None.")
            return
        path = Path(ckpt_path)
        if not path.is_file():
            print(f"[WARN] Perception checkpoint not found: {path}")
            return
        payload = torch.load(path, map_location=self.cuda_device, weights_only=False)
        self._subtask_vocab = payload.get("subtask_vocab", self._subtask_vocab)
        self._subtask_to_idx = {name: i for i, name in enumerate(self._subtask_vocab)}
        train_cfg = payload.get("train_cfg", CfgPerceptionTraining)
        self._model = MultimodalSubtaskModel(
            num_cameras=payload.get("num_cameras", 1),
            num_subtasks=len(self._subtask_vocab),
            signal_dim=payload.get("signal_dim", 8),
            image_size=train_cfg.get("image_size", 224),
            backbone=train_cfg.get("backbone", "resnet18"),
        ).to(self.cuda_device)
        self._model.load_state_dict(payload["model_state_dict"])
        self._model.eval()
        print(f"[INFO] Perception model loaded from {path}")

    def reset(self, env_state_action_dict: dict) -> dict:
        env_state_action_dict.setdefault("perception", {})
        env_state_action_dict["perception"]["predictions"] = {}
        env_state_action_dict["perception"]["last_sample"] = None

        if self.enabled and self.mode == "collect" and self._logger is not None:
            self._logger.reset()
        return env_state_action_dict

    def step(self, env_state_action_dict: dict) -> dict:
        if not self.enabled or self.mode == "off":
            return env_state_action_dict

        env_state_action_dict.setdefault("perception", {})

        if self.mode == "collect" and self._logger is not None:
            sample = self._logger.maybe_log(env_state_action_dict)
            env_state_action_dict["perception"]["last_sample"] = sample

        elif self.mode == "infer" and self._model is not None:
            predictions = self._infer(env_state_action_dict)
            env_state_action_dict["perception"]["predictions"] = predictions

        return env_state_action_dict

    def _infer(self, env_state_action_dict: dict) -> dict[str, dict]:
        agent_signals = extract_agent_signals(env_state_action_dict)
        predictions: dict[str, dict] = {}

        camera_tensors = self._cameras_to_tensor(env_state_action_dict)
        signal_tensor = self._signals_to_tensor(agent_signals)

        with torch.no_grad():
            subtask_logits, done_logit = self._model(camera_tensors.unsqueeze(0), signal_tensor.unsqueeze(0))
            subtask_idx = int(subtask_logits.argmax(dim=-1).item())
            subtask_name = self._subtask_vocab[subtask_idx]
            subtask_done = bool(torch.sigmoid(done_logit).item() > 0.5)

        for human_key in env_state_action_dict["human"]:
            name, done = subtask_name, subtask_done
            if self.cfg.get("use_constraint_propagation", True):
                name, done = apply_constraint_propagation(name, done, agent_signals, human_key)
            predictions[human_key] = {
                "subtask_name": name,
                "subtask_done": done,
                "subtask_index_pred": subtask_idx,
            }
        return predictions

    def _cameras_to_tensor(self, env_state_action_dict: dict) -> torch.Tensor:
        frames: list[torch.Tensor] = []
        image_size = CfgPerceptionTraining.get("image_size", 224)
        for camera_state in env_state_action_dict.get("camera", {}).values():
            rgb = camera_state.get("rgb")
            if rgb is None:
                continue
            t = rgb.float() / 255.0 if rgb.dtype == torch.uint8 else rgb.float()
            if t.dim() == 3:
                t = t.permute(2, 0, 1)
            t = F.interpolate(t.unsqueeze(0), size=(image_size, image_size), mode="bilinear", align_corners=False)
            frames.append(t.squeeze(0))
        if not frames:
            return torch.zeros(3, image_size, image_size, device=self.cuda_device)
        stacked = torch.stack(frames[: self._model.num_cameras], dim=0)  # type: ignore[union-attr]
        while stacked.shape[0] < self._model.num_cameras:  # type: ignore[union-attr]
            stacked = torch.cat([stacked, stacked[:1]], dim=0)
        return stacked

    def _signals_to_tensor(self, agent_signals: dict) -> torch.Tensor:
        if not agent_signals:
            return torch.zeros(8, device=self.cuda_device)
        sig = next(iter(agent_signals.values()))
        finished = (sig.get("finished") or [False] * 4) + [False] * 4
        row = sig.get("ongoing_row") or [""] * 4
        vec = [
            float(sig.get("subtask_index") or 0),
            float(sig.get("num_subtasks") or 0),
            float(finished[AGENT_COL_HUMAN]),
            float(finished[AGENT_COL_GANTRY]),
            float(finished[AGENT_COL_MACHINE]),
            float(finished[AGENT_COL_ROBOT]),
            float(row[AGENT_COL_HUMAN] == "wait"),
            float(row[AGENT_COL_GANTRY] == "wait"),
        ]
        return torch.tensor(vec, dtype=torch.float32, device=self.cuda_device)


# ---------------------------------------------------------------------------
# Dataset & model (offline training)
# ---------------------------------------------------------------------------


def _load_samples_from_dataset(dataset_dir: Path) -> list[dict]:
    samples: list[dict] = []
    for meta_path in sorted(dataset_dir.glob("env_*_episode_*/meta.jsonl")):
        episode_dir = meta_path.parent
        with meta_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                sample = json.loads(line)
                sample["_episode_dir"] = str(episode_dir)
                samples.append(sample)
    return samples


def _signals_vector(agent_signals: dict, product_index: int | None) -> list[float]:
    if product_index is None:
        return [0.0] * 8
    sig = agent_signals.get(str(product_index))
    if sig is None:
        return [0.0] * 8
    finished = (sig.get("finished") or [False] * 4) + [False] * 4
    row = sig.get("ongoing_row") or [""] * 4
    return [
        float(sig.get("subtask_index") or 0),
        float(sig.get("num_subtasks") or 0),
        float(finished[AGENT_COL_HUMAN]),
        float(finished[AGENT_COL_GANTRY]),
        float(finished[AGENT_COL_MACHINE]),
        float(finished[AGENT_COL_ROBOT]),
        float(row[AGENT_COL_HUMAN] == "wait"),
        float(row[AGENT_COL_GANTRY] == "wait"),
    ]


class PerceptionDataset(Dataset):
    """Load logged perception episodes for subtask training."""

    def __init__(
        self,
        dataset_dir: str | Path,
        subtask_vocab: list[str],
        image_size: int = 224,
        use_agent_signals: bool = True,
        camera_keys: list[str] | None = None,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.subtask_vocab = subtask_vocab
        self.subtask_to_idx = {n: i for i, n in enumerate(subtask_vocab)}
        self.image_size = image_size
        self.use_agent_signals = use_agent_signals
        self.samples = _load_samples_from_dataset(self.dataset_dir)
        if not self.samples:
            raise FileNotFoundError(f"No perception samples found under {self.dataset_dir}")

        if camera_keys is None:
            camera_keys = sorted(
                {k for s in self.samples for k in s.get("camera_paths", {}).keys()}
            )
        self.camera_keys = camera_keys
        self.num_cameras = max(1, len(camera_keys))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        episode_dir = Path(sample["_episode_dir"])
        images = []
        for cam_key in self.camera_keys:
            rel = sample.get("camera_paths", {}).get(cam_key)
            if rel is None:
                images.append(torch.zeros(3, self.image_size, self.image_size))
                continue
            img = Image.open(episode_dir / rel).convert("RGB")
            t = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
            t = F.interpolate(t.unsqueeze(0), size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
            images.append(t.squeeze(0))
        while len(images) < self.num_cameras:
            images.append(images[0].clone())
        images = torch.stack(images[: self.num_cameras], dim=0)

        # use first working human in sample
        subtask_name = "wait"
        subtask_done = 0
        product_index = None
        for label in sample.get("human_labels", {}).values():
            if label.get("state") != "free" and label.get("subtask_name"):
                subtask_name = label["subtask_name"]
                subtask_done = int(bool(label.get("subtask_done")))
                product_index = label.get("product_index")
                break

        subtask_idx = self.subtask_to_idx.get(subtask_name, self.subtask_to_idx.get("wait", 0))
        signals = torch.tensor(
            _signals_vector(sample.get("agent_signals", {}), product_index),
            dtype=torch.float32,
        )

        return {
            "images": images,
            "signals": signals,
            "subtask_idx": torch.tensor(subtask_idx, dtype=torch.long),
            "subtask_done": torch.tensor(subtask_done, dtype=torch.float32),
            "subtask_name": subtask_name,
        }


class MultimodalSubtaskModel(nn.Module):
    """Simple multi-camera + signal baseline for subtask classification."""

    def __init__(
        self,
        num_cameras: int,
        num_subtasks: int,
        signal_dim: int = 8,
        image_size: int = 224,
        backbone: str = "resnet18",
    ):
        super().__init__()
        self.num_cameras = num_cameras
        from torchvision import models

        weights = models.ResNet18_Weights.DEFAULT
        resnet = models.resnet18(weights=weights)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        feat_dim = 512
        self.cam_encoder = nn.Linear(feat_dim, 256)
        self.signal_encoder = nn.Linear(signal_dim, 64)
        fusion_dim = 256 * num_cameras + 64
        self.head_subtask = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_subtasks),
        )
        self.head_done = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, images: torch.Tensor, signals: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # images: (B, N, 3, H, W)
        b, n, c, h, w = images.shape
        feats = []
        for i in range(n):
            x = images[:, i]
            f = self.backbone(x).flatten(1)
            feats.append(self.cam_encoder(f))
        fused = torch.cat(feats, dim=-1)
        sig = self.signal_encoder(signals)
        fused = torch.cat([fused, sig], dim=-1)
        return self.head_subtask(fused), self.head_done(fused).squeeze(-1)


class PerceptionTrainer:
    """Offline trainer for subtask perception baseline."""

    def __init__(self, train_cfg: dict | None = None):
        self.cfg = copy.deepcopy(train_cfg or CfgPerceptionTraining)
        self.device = torch.device(self.cfg["device"] if torch.cuda.is_available() else "cpu")

    def train(self) -> Path:
        dataset = PerceptionDataset(
            self.cfg["dataset_dir"],
            subtask_vocab=self.cfg["subtask_vocab"],
            image_size=self.cfg["image_size"],
            use_agent_signals=self.cfg["use_agent_signals"],
        )
        val_len = max(1, int(len(dataset) * self.cfg["val_ratio"]))
        train_len = len(dataset) - val_len
        train_set, val_set = random_split(
            dataset, [train_len, val_len], generator=torch.Generator().manual_seed(42)
        )

        train_loader = DataLoader(
            train_set,
            batch_size=self.cfg["batch_size"],
            shuffle=True,
            num_workers=self.cfg["num_workers"],
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_set,
            batch_size=self.cfg["batch_size"],
            shuffle=False,
            num_workers=self.cfg["num_workers"],
            pin_memory=True,
        )

        model = MultimodalSubtaskModel(
            num_cameras=dataset.num_cameras,
            num_subtasks=len(self.cfg["subtask_vocab"]),
            signal_dim=8,
            image_size=self.cfg["image_size"],
            backbone=self.cfg["backbone"],
        ).to(self.device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.cfg["learning_rate"],
            weight_decay=self.cfg["weight_decay"],
        )

        run_dir = Path(self.cfg["output_dir"]) / self.cfg["run_name"]
        run_dir.mkdir(parents=True, exist_ok=True)
        best_acc = 0.0
        history: list[dict] = []

        for epoch in range(self.cfg["num_epochs"]):
            model.train()
            train_loss = 0.0
            for batch in train_loader:
                images = batch["images"].to(self.device)
                signals = batch["signals"].to(self.device)
                subtask_tgt = batch["subtask_idx"].to(self.device)
                done_tgt = batch["subtask_done"].to(self.device)

                subtask_logits, done_logit = model(images, signals)
                loss_cls = F.cross_entropy(subtask_logits, subtask_tgt)
                loss_done = F.binary_cross_entropy_with_logits(done_logit, done_tgt)
                loss = loss_cls + loss_done

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += float(loss.item())

            metrics = self._evaluate(model, val_loader)
            metrics["epoch"] = epoch
            metrics["train_loss"] = train_loss / max(1, len(train_loader))
            history.append(metrics)
            print(
                f"[epoch {epoch:03d}] train_loss={metrics['train_loss']:.4f} "
                f"val_subtask_acc={metrics['subtask_acc']:.3f} val_done_acc={metrics['done_acc']:.3f}"
            )

            ckpt = {
                "model_state_dict": model.state_dict(),
                "subtask_vocab": self.cfg["subtask_vocab"],
                "num_cameras": dataset.num_cameras,
                "signal_dim": 8,
                "train_cfg": self.cfg,
            }
            torch.save(ckpt, run_dir / "last.pt")
            if metrics["subtask_acc"] >= best_acc:
                best_acc = metrics["subtask_acc"]
                torch.save(ckpt, run_dir / "best.pt")

        with (run_dir / "history.json").open("w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        print(f"[INFO] Training done. Checkpoints -> {run_dir}")
        return run_dir

    @torch.no_grad()
    def _evaluate(self, model: nn.Module, loader: DataLoader) -> dict:
        model.eval()
        correct_subtask = 0
        correct_done = 0
        total = 0
        for batch in loader:
            images = batch["images"].to(self.device)
            signals = batch["signals"].to(self.device)
            subtask_logits, done_logit = model(images, signals)
            pred_subtask = subtask_logits.argmax(dim=-1)
            pred_done = (torch.sigmoid(done_logit) > 0.5).long()
            correct_subtask += int((pred_subtask == batch["subtask_idx"].to(self.device)).sum().item())
            correct_done += int((pred_done == batch["subtask_done"].long().to(self.device)).sum().item())
            total += images.shape[0]
        return {
            "subtask_acc": correct_subtask / max(1, total),
            "done_acc": correct_done / max(1, total),
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HRTPA perception training / evaluation")
    parser.add_argument("command", choices=["train", "eval"], help="train on logged dataset or evaluate checkpoint")
    parser.add_argument("--dataset_dir", type=str, default=CfgPerceptionTraining["dataset_dir"])
    parser.add_argument("--output_dir", type=str, default=CfgPerceptionTraining["output_dir"])
    parser.add_argument("--run_name", type=str, default=CfgPerceptionTraining["run_name"])
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=CfgPerceptionTraining["num_epochs"])
    parser.add_argument("--batch_size", type=int, default=CfgPerceptionTraining["batch_size"])
    parser.add_argument("--device", type=str, default=CfgPerceptionTraining["device"])
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    train_cfg = copy.deepcopy(CfgPerceptionTraining)
    train_cfg.update(
        {
            "dataset_dir": args.dataset_dir,
            "output_dir": args.output_dir,
            "run_name": args.run_name,
            "num_epochs": args.epochs,
            "batch_size": args.batch_size,
            "device": args.device,
        }
    )

    trainer = PerceptionTrainer(train_cfg)
    if args.command == "train":
        trainer.train()
    elif args.command == "eval":
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required for eval")
        # quick eval using trainer utilities
        dataset = PerceptionDataset(args.dataset_dir, subtask_vocab=train_cfg["subtask_vocab"])
        loader = DataLoader(dataset, batch_size=train_cfg["batch_size"], shuffle=False)
        payload = torch.load(args.checkpoint, map_location=trainer.device, weights_only=False)
        model = MultimodalSubtaskModel(
            num_cameras=payload["num_cameras"],
            num_subtasks=len(payload["subtask_vocab"]),
            signal_dim=payload.get("signal_dim", 8),
        ).to(trainer.device)
        model.load_state_dict(payload["model_state_dict"])
        metrics = trainer._evaluate(model, loader)
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
