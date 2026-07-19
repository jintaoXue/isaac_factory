"""Perception: dual-task collect / offline train (human-id + human-subtask)."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

_HC_FACTORY = Path(__file__).resolve().parents[1]
_CFG_DIR = _HC_FACTORY / "env_asset_cfg"
_PERCEPTION_CFG_DIR = _CFG_DIR / "perception"


def _load_modules_offline() -> tuple[Any, Any, Any]:
    """Load perception/camera/human cfg without isaaclab (CLI train/eval)."""
    pkg = "hc_factory_env_asset_cfg"
    for name in ("cfg_machine", "cfg_process_subtask_gallery", "cfg_process_task_gallery"):
        full = f"{pkg}.{name}"
        if full in sys.modules:
            continue
        path = _CFG_DIR / f"{name}.py"
        spec = importlib.util.spec_from_file_location(full, path)
        mod = importlib.util.module_from_spec(spec)
        mod.__package__ = pkg
        sys.modules[full] = mod
        assert spec.loader is not None
        spec.loader.exec_module(mod)

    loaded = {}
    for name, path, package in (
        ("cfg_perception", _PERCEPTION_CFG_DIR / "cfg_perception.py", f"{pkg}.perception"),
        ("cfg_camera", _PERCEPTION_CFG_DIR / "cfg_camera.py", f"{pkg}.perception"),
        ("cfg_human", _CFG_DIR / "cfg_human.py", pkg),
    ):
        full = f"{package}.{name}" if name != "cfg_human" else f"{pkg}.{name}"
        if full not in sys.modules:
            spec = importlib.util.spec_from_file_location(full, path)
            mod = importlib.util.module_from_spec(spec)
            mod.__package__ = package if name != "cfg_human" else pkg
            sys.modules[full] = mod
            assert spec.loader is not None
            # cfg_human imports HcVectorEnvCfg — only needed under Isaac; stub fields for CLI.
            if name == "cfg_human":
                try:
                    spec.loader.exec_module(mod)
                except Exception:
                    # Minimal offline stub: HumanIdVocab only.
                    mod.HumanIdVocab = ["00", "01", "02", "03", "04"]
            else:
                spec.loader.exec_module(mod)
        loaded[name] = sys.modules[full]
    return loaded["cfg_perception"], loaded["cfg_camera"], loaded["cfg_human"]


try:
    from ..env_asset_cfg.cfg_human import HumanIdVocab
    from ..env_asset_cfg.perception.cfg_camera import (
        CAMERA_POSES,
        CfgCamera,
        camera_detects_human_id,
        visible_human_ids_for_camera,
    )
    from ..env_asset_cfg.perception.cfg_perception import (
        AGENT_COL_HUMAN,
        CfgPerception,
        CfgPerceptionTraining,
        HumanSubtaskVocab,
        PerceptionSampleTemplate,
        TaskLabelToIndex,
    )
except ImportError:
    _p, _c, _h = _load_modules_offline()
    AGENT_COL_HUMAN = _p.AGENT_COL_HUMAN
    CfgPerception = _p.CfgPerception
    CfgPerceptionTraining = _p.CfgPerceptionTraining
    HumanSubtaskVocab = _p.HumanSubtaskVocab
    PerceptionSampleTemplate = _p.PerceptionSampleTemplate
    TaskLabelToIndex = _p.TaskLabelToIndex
    CAMERA_POSES = _c.CAMERA_POSES
    CfgCamera = _c.CfgCamera
    camera_detects_human_id = _c.camera_detects_human_id
    visible_human_ids_for_camera = _c.visible_human_ids_for_camera
    HumanIdVocab = getattr(_h, "HumanIdVocab", ["00", "01", "02", "03", "04"])

_SUBTASK_TO_ID = {n: i for i, n in enumerate(HumanSubtaskVocab)}
_HUMAN_ID_TO_IDX = {h: i for i, h in enumerate(HumanIdVocab)}


def to_serializable(obj: Any) -> Any:
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
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


def _xy_from_position(pos) -> tuple[float, float] | None:
    if pos is None:
        return None
    if isinstance(pos, torch.Tensor):
        t = pos.detach().cpu().reshape(-1)
        if t.numel() < 2:
            return None
        return float(t[0]), float(t[1])
    arr = np.asarray(pos).reshape(-1)
    if arr.size < 2:
        return None
    return float(arr[0]), float(arr[1])


def human_xy_by_id(env: dict) -> dict[str, tuple[float, float]]:
    """Map human id '00'.. from rigid_prims."""
    out: dict[str, tuple[float, float]] = {}
    for key, prim in env.get("rigid_prims", {}).items():
        if "NormalHuman" not in key and "human" not in key.lower():
            continue
        # num_00_NormalHuman -> 00
        parts = key.split("_")
        hid = None
        for p in parts:
            if p.isdigit() and len(p) == 2:
                hid = p
                break
        if hid is None:
            continue
        xy = _xy_from_position(prim.get("position"))
        if xy is not None:
            out[hid] = xy
    return out


def extract_human_labels(env: dict) -> dict[str, dict]:
    ongoing = env["progress"]["ongoing_task_records"]
    labels = {}
    for human_key, hs in env["human"].items():
        label = {
            "human_index": hs["key_variables"]["idx"],
            "state": hs["state"],
            "current_area_id": hs.get("current_area_id"),
            "target_area_id": hs.get("target_area_id"),
            "task": None,
            "subtask_index": None,
            "subtask_name": None,
            "subtask_done": None,
        }
        rid = hs.get("ongoing_task_record_index")
        if rid is not None and rid in ongoing:
            tr, sd = ongoing[rid], ongoing[rid]["subtasks_dict"]
            row, fin = sd["ongoing"], sd["finished"]
            label.update(
                task=tr.get("task"),
                subtask_index=sd.get("ongoing_index"),
                subtask_name=row[AGENT_COL_HUMAN],
                subtask_done=bool(fin[AGENT_COL_HUMAN]),
            )
        labels[human_key] = label
    return labels


def _empty_nested_images() -> dict:
    return {
        machine: {cam: {"image": None} for cam in cams}
        for machine, cams in CAMERA_POSES.items()
    }


def _empty_id_labels() -> dict:
    return {
        machine: {cam: {"human_ids": []} for cam in cams}
        for machine, cams in CAMERA_POSES.items()
    }


def build_id_labels(human_xy: dict[str, tuple[float, float]]) -> dict:
    labels = _empty_id_labels()
    for machine, cams in CAMERA_POSES.items():
        for cam in cams:
            ids = visible_human_ids_for_camera(cam, human_xy) if camera_detects_human_id(cam) else []
            labels[machine][cam]["human_ids"] = ids
    return labels


def build_subtask_block(labels: dict[str, dict], images_nested: dict) -> dict:
    working = sorted(
        ((hk, lb) for hk, lb in labels.items() if lb.get("state") != "free" and lb.get("subtask_name")),
        key=lambda x: x[1]["human_index"],
    )
    tasks, task_ids, keys = [], [], []
    subs, sub_ids, dones = [], [], []
    for hk, lb in working:
        task = lb.get("task") or "none"
        sub = lb["subtask_name"]
        keys.append(hk)
        tasks.append(task)
        task_ids.append(int(TaskLabelToIndex.get(task, 0)))
        subs.append(sub)
        sub_ids.append(int(_SUBTASK_TO_ID.get(sub, -1)))
        dones.append(bool(lb.get("subtask_done")))
    return {
        "input": {"images": images_nested, "human_keys": keys, "human_task": tasks, "human_task_id": task_ids},
        "output_label": {
            "human_subtask": subs,
            "human_subtask_id": sub_ids,
            "human_subtask_done": dones,
        },
    }


def slim_env_record(env: dict) -> dict:
    """Small serializable snapshot (no RGB / big tensors)."""
    humans = {}
    for k, hs in env.get("human", {}).items():
        humans[k] = {
            "state": hs.get("state"),
            "current_area_id": hs.get("current_area_id"),
            "target_area_id": hs.get("target_area_id"),
            "ongoing_task_record_index": hs.get("ongoing_task_record_index"),
        }
    prims = {}
    for k, prim in env.get("rigid_prims", {}).items():
        xy = _xy_from_position(prim.get("position"))
        if xy is not None:
            prims[k] = {"xy": list(xy)}
    return {
        "time_step": env.get("time_step"),
        "human": humans,
        "rigid_prims_xy": prims,
    }


def build_step_sample(
    env: dict,
    env_id: int,
    episode_id: int,
    step_id: int,
    images_nested: dict,
    *,
    save_env_record: bool = True,
) -> dict:
    labels = extract_human_labels(env)
    human_xy = human_xy_by_id(env)
    id_labels = build_id_labels(human_xy)
    # pick a scene-level task tag from first working human (else none)
    scene_task = "none"
    for lb in labels.values():
        if lb.get("task"):
            scene_task = lb["task"]
            break
    subtask_block = build_subtask_block(labels, images_nested)
    return {
        "episode_id": episode_id,
        "step_id": step_id,
        "time_step": env.get("time_step"),
        "env_id": env_id,
        "human_id_recognition": {
            "input": {"task": scene_task, "images": images_nested},
            "output_label": id_labels,
        },
        "human_subtask_recognition": subtask_block,
        "env_state_action_dict": slim_env_record(env) if save_env_record else {},
    }


def _rgb_numpy(rgb) -> np.ndarray | None:
    if rgb is None:
        return None
    arr = rgb.detach().cpu().numpy() if isinstance(rgb, torch.Tensor) else np.asarray(rgb)
    return arr if arr.size else None


def _resize_rgb(rgb, size: int) -> torch.Tensor:
    t = rgb.float() / 255.0 if isinstance(rgb, torch.Tensor) and rgb.dtype == torch.uint8 else torch.as_tensor(rgb).float()
    if t.dim() == 3 and t.shape[-1] == 3:
        t = t.permute(2, 0, 1)
    return F.interpolate(t.unsqueeze(0), size=(size, size), mode="bilinear", align_corners=False).squeeze(0)


def _cameras_enabled() -> bool:
    try:
        import carb

        return bool(carb.settings.get_settings().get("/isaaclab/cameras_enabled"))
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Sim collect / infer
# ---------------------------------------------------------------------------


class PerceptionLogger:
    def __init__(self, env_id: int, cfg: dict):
        self.env_id, self.cfg = env_id, cfg
        self.output_dir = Path(cfg["output_dir"])
        self._episode_num: int | None = None
        self.step_id = 0
        self._ep_dir: Path | None = None

    def reset(self, env: dict) -> None:
        episode_num = int(env.get("episode_num", 0))
        self._episode_num = episode_num
        if episode_num >= self.cfg["max_episodes"]:
            self._ep_dir = None
            self.step_id = 0
            return
        self._ep_dir = self.output_dir / f"env_{self.env_id:02d}_episode_{episode_num:06d}"
        self._ep_dir.mkdir(parents=True, exist_ok=True)
        if episode_num == 0:
            (self.output_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "env_id": self.env_id,
                        "sample_template": to_serializable(PerceptionSampleTemplate),
                        "cfg": to_serializable(self.cfg),
                        "human_id_vocab": HumanIdVocab,
                        "human_subtask_vocab": HumanSubtaskVocab,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        self.step_id = 0

    def maybe_log(self, env: dict) -> dict | None:
        episode_num = int(env.get("episode_num", self._episode_num if self._episode_num is not None else 0))
        if not self._ep_dir or episode_num >= self.cfg["max_episodes"]:
            return None
        if env["time_step"] % self.cfg["save_interval"]:
            return None
        if self.cfg.get("max_steps_per_episode") is not None and self.step_id >= self.cfg["max_steps_per_episode"]:
            return None

        images_nested = self._save_images(env)
        sample = build_step_sample(
            env,
            self.env_id,
            episode_num,
            self.step_id,
            images_nested,
            save_env_record=bool(self.cfg.get("save_env_record", True)),
        )
        with (self._ep_dir / "meta.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(to_serializable(sample), ensure_ascii=False) + "\n")

        self._print_step(sample)
        self.step_id += 1
        return sample

    def _save_images(self, env: dict) -> dict:
        nested = _empty_nested_images()
        if not self.cfg.get("save_images", True):
            return nested
        assert self._ep_dir
        step_dir = self._ep_dir / "cameras" / f"step_{self.step_id:06d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        ext = self.cfg.get("image_format", "jpg")
        cam_states = env.get("camera", {})
        for machine, cams in CAMERA_POSES.items():
            for cam in cams:
                st = cam_states.get(cam) or cam_states.get(f"num_00_{cam}")
                if not st or not st.get("is_initialized"):
                    continue
                arr = _rgb_numpy(st.get("rgb"))
                if arr is None:
                    continue
                out = step_dir / f"{cam}.{ext}"
                kw = {"format": "JPEG", "quality": int(self.cfg.get("image_quality", 90))} if ext.lower() in ("jpg", "jpeg") else {}
                Image.fromarray(arr.astype(np.uint8)).save(out, **kw)
                nested[machine][cam]["image"] = str(out.relative_to(self._ep_dir))
        return nested

    def _print_step(self, sample: dict) -> None:
        id_labels = sample["human_id_recognition"]["output_label"]
        visible = {
            cam: lab["human_ids"]
            for m in id_labels.values()
            for cam, lab in m.items()
            if lab["human_ids"]
        }
        sub = sample["human_subtask_recognition"]
        keys = sub["input"]["human_keys"]
        names = sub["output_label"]["human_subtask"]
        dones = sub["output_label"]["human_subtask_done"]
        print(
            f"[perception] env={sample['env_id']} ep={sample['episode_id']} "
            f"step={sample['step_id']} visible={visible or '{}'} "
            f"working={list(zip(keys, names, dones))}"
        )


class PerceptionManager:
    def __init__(self, env_id: int, cuda_device: torch.device, cfg: dict | None = None):
        self.env_id, self.cuda_device = env_id, cuda_device
        self.cfg = copy.deepcopy(cfg or CfgPerception)
        cfg_enabled = bool(self.cfg.get("enabled", False))
        self.enabled = cfg_enabled and _cameras_enabled()
        if cfg_enabled and not self.enabled:
            print(f"[INFO] Perception disabled for env_{env_id:02d}: --enable_cameras not set.")
        self.mode = self.cfg.get("mode", "off") if self.enabled else "off"
        self._logger = PerceptionLogger(env_id, self.cfg) if self.enabled and self.mode == "collect" else None

    def reset(self, env: dict) -> dict:
        env.setdefault("perception", {})["predictions"] = {} 
        env["perception"]["last_sample"] = None
        if self._logger:
            self._logger.reset(env)
        return env

    def step(self, env: dict) -> dict:
        if not self.enabled or self.mode == "off":
            return env
        env.setdefault("perception", {})
        if self.mode == "collect" and self._logger:
            env["perception"]["last_sample"] = self._logger.maybe_log(env)
        return env


# ---------------------------------------------------------------------------
# Offline datasets / models
# ---------------------------------------------------------------------------


def _iter_meta_rows(dataset_dir: Path):
    for meta in sorted(dataset_dir.glob("env_*_episode_*/meta.jsonl")):
        ep = meta.parent
        for line in meta.read_text(encoding="utf-8").splitlines():
            if line.strip():
                yield json.loads(line), ep


def _load_image(ep: Path, rel: str | None, size: int) -> torch.Tensor:
    if not rel:
        return torch.zeros(3, size, size)
    return _resize_rgb(np.array(Image.open(ep / rel).convert("RGB")), size)


class HumanIdDataset(Dataset):
    """One item = one camera frame → multi-hot over HumanIdVocab."""

    def __init__(
        self,
        dataset_dir: str | Path,
        image_size: int = 224,
        episode_dirs: set[Path] | None = None,
    ):
        self.image_size = image_size
        self.items: list[dict] = []
        for row, ep in _iter_meta_rows(Path(dataset_dir)):
            if episode_dirs is not None and ep.resolve() not in episode_dirs:
                continue
            if "human_id_recognition" not in row:
                continue
            block = row["human_id_recognition"]
            images = block["input"]["images"]
            labels = block["output_label"]
            for machine, cams in images.items():
                for cam, payload in cams.items():
                    if not camera_detects_human_id(cam):
                        continue
                    rel = payload.get("image") if isinstance(payload, dict) else None
                    if not rel:
                        continue
                    ids = labels.get(machine, {}).get(cam, {}).get("human_ids", [])
                    vec = torch.zeros(len(HumanIdVocab), dtype=torch.float32)
                    for hid in ids:
                        if hid in _HUMAN_ID_TO_IDX:
                            vec[_HUMAN_ID_TO_IDX[hid]] = 1.0
                    self.items.append({"_episode_dir": ep, "image": rel, "target": vec, "camera": cam})
        if not self.items:
            raise FileNotFoundError(f"No human-id samples under {dataset_dir}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict:
        it = self.items[i]
        return {
            "image": _load_image(Path(it["_episode_dir"]), it["image"], self.image_size),
            "target": it["target"],
            "camera": it["camera"],
        }


class HumanSubtaskDataset(Dataset):
    """One item = one working human at a step (shared multi-cam images)."""

    def __init__(
        self,
        dataset_dir: str | Path,
        image_size: int = 224,
        episode_dirs: set[Path] | None = None,
    ):
        self.image_size = image_size
        self.camera_order = [cam for _, cams in CAMERA_POSES.items() for cam in cams]
        self.items: list[dict] = []
        for row, ep in _iter_meta_rows(Path(dataset_dir)):
            if episode_dirs is not None and ep.resolve() not in episode_dirs:
                continue
            if "human_subtask_recognition" not in row:
                continue
            block = row["human_subtask_recognition"]
            images = block["input"]["images"]
            keys = block["input"].get("human_keys") or []
            tasks = block["input"].get("human_task_id") or []
            subs = block["output_label"].get("human_subtask_id") or []
            dones = block["output_label"].get("human_subtask_done") or []
            for i, key in enumerate(keys):
                sid = int(subs[i]) if i < len(subs) else -1
                if sid < 0 or sid >= len(HumanSubtaskVocab):
                    continue
                self.items.append(
                    {
                        "_episode_dir": ep,
                        "images": images,
                        "human_key": key,
                        "task_id": int(tasks[i]) if i < len(tasks) else 0,
                        "subtask_id": sid,
                        "subtask_done": float(bool(dones[i])) if i < len(dones) else 0.0,
                    }
                )
        if not self.items:
            raise FileNotFoundError(f"No human-subtask samples under {dataset_dir}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict:
        it = self.items[i]
        ep = Path(it["_episode_dir"])
        frames = []
        for cam in self.camera_order:
            rel = None
            for machine, cams in it["images"].items():
                if cam in cams and isinstance(cams[cam], dict):
                    rel = cams[cam].get("image")
                    break
            frames.append(_load_image(ep, rel, self.image_size))
        return {
            "images": torch.stack(frames),
            "task_id": torch.tensor(it["task_id"], dtype=torch.long),
            "subtask_id": torch.tensor(it["subtask_id"], dtype=torch.long),
            "subtask_done": torch.tensor(it["subtask_done"], dtype=torch.float32),
        }


class HumanIdModel(nn.Module):
    def __init__(self, num_ids: int = len(HumanIdVocab)):
        super().__init__()
        from torchvision import models

        self.backbone = nn.Sequential(*list(models.resnet18(weights=models.ResNet18_Weights.DEFAULT).children())[:-1])
        self.head = nn.Linear(512, num_ids)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(image).flatten(1))


class HumanSubtaskModel(nn.Module):
    def __init__(self, num_cameras: int, num_subtasks: int = len(HumanSubtaskVocab), num_tasks: int = 16):
        super().__init__()
        self.num_cameras = num_cameras
        from torchvision import models

        self.backbone = nn.Sequential(*list(models.resnet18(weights=models.ResNet18_Weights.DEFAULT).children())[:-1])
        self.cam_enc = nn.Linear(512, 128)
        self.task_emb = nn.Embedding(num_tasks, 32)
        dim = 128 * num_cameras + 32
        self.head_sub = nn.Sequential(nn.Linear(dim, 128), nn.ReLU(True), nn.Linear(128, num_subtasks))
        self.head_done = nn.Sequential(nn.Linear(dim, 64), nn.ReLU(True), nn.Linear(64, 1))

    def forward(self, images: torch.Tensor, task_id: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feats = [self.cam_enc(self.backbone(images[:, i]).flatten(1)) for i in range(images.shape[1])]
        fused = torch.cat([*feats, self.task_emb(task_id.clamp(min=0))], dim=-1)
        return self.head_sub(fused), self.head_done(fused).squeeze(-1)


class PerceptionTrainer:
    def __init__(self, cfg: dict | None = None):
        self.cfg = copy.deepcopy(cfg or CfgPerceptionTraining)
        self.device = torch.device(self.cfg["device"] if torch.cuda.is_available() else "cpu")

    def train(self) -> Path:
        task = self.cfg.get("task", "subtask")
        if task == "id":
            return self._train_id()
        if task == "subtask":
            return self._train_subtask()
        raise ValueError(f"Unknown training task: {task} (use id|subtask)")

    def _episode_split(self) -> tuple[set[Path], set[Path], set[Path]]:
        episodes = sorted(Path(self.cfg["dataset_dir"]).glob("env_*_episode_*"))
        if not episodes:
            raise FileNotFoundError(f"No episodes under {self.cfg['dataset_dir']}")
        order = list(episodes)
        random.Random(self.cfg.get("split_seed", 42)).shuffle(order)
        n = len(order)
        n_train = int(round(n * self.cfg.get("train_ratio", 0.7)))
        n_val = int(round(n * self.cfg.get("val_ratio", 0.15)))
        n_train = min(max(n_train, 1), n) if n else 0
        n_val = min(n_val, max(0, n - n_train))
        train_eps = {p.resolve() for p in order[:n_train]}
        val_eps = {p.resolve() for p in order[n_train : n_train + n_val]}
        test_eps = {p.resolve() for p in order[n_train + n_val :]}
        if not val_eps and len(train_eps) > 1:
            moved = next(iter(train_eps))
            train_eps.remove(moved)
            val_eps.add(moved)
        return train_eps, val_eps, test_eps

    def _split(self, ds_cls: type[Dataset]):
        train_eps, val_eps, _ = self._episode_split()
        if not val_eps:
            val_eps = train_eps
        train_set = ds_cls(self.cfg["dataset_dir"], self.cfg["image_size"], train_eps)
        val_set = ds_cls(self.cfg["dataset_dir"], self.cfg["image_size"], val_eps)
        return train_set, val_set

    def _train_id(self) -> Path:
        train_set, val_set = self._split(HumanIdDataset)
        train_ld = DataLoader(train_set, self.cfg["batch_size"], shuffle=True, num_workers=self.cfg["num_workers"])
        val_ld = DataLoader(val_set, self.cfg["batch_size"], num_workers=self.cfg["num_workers"])
        model = HumanIdModel().to(self.device)
        opt = torch.optim.AdamW(model.parameters(), lr=self.cfg["learning_rate"], weight_decay=self.cfg["weight_decay"])
        run_dir = Path(self.cfg["output_dir"]) / f"{self.cfg['run_name']}_id"
        run_dir.mkdir(parents=True, exist_ok=True)
        best, history = 0.0, []
        for epoch in range(self.cfg["num_epochs"]):
            model.train()
            loss_sum = 0.0
            for b in train_ld:
                logits = model(b["image"].to(self.device))
                loss = F.binary_cross_entropy_with_logits(logits, b["target"].to(self.device))
                opt.zero_grad()
                loss.backward()
                opt.step()
                loss_sum += loss.item()
            m = self._eval_id(model, val_ld) | {"epoch": epoch, "train_loss": loss_sum / max(1, len(train_ld))}
            history.append(m)
            print(
                f"[id epoch {epoch:03d}] loss={m['train_loss']:.4f} "
                f"elem_acc={m['elem_acc']:.3f} "
                f"empty_acc={m['empty_elem_acc']:.3f}(n={m['empty_n']}) "
                f"occ_acc={m['occupied_elem_acc']:.3f}(n={m['occupied_n']})"
            )
            ckpt = {"model_state_dict": model.state_dict(), "task": "id", "train_cfg": self.cfg}
            torch.save(ckpt, run_dir / "last.pt")
            # prefer occupied-frame accuracy for checkpoint selection
            score = m["occupied_elem_acc"] if m["occupied_n"] > 0 else m["elem_acc"]
            if score >= best:
                best = score
                torch.save(ckpt, run_dir / "best.pt")
        (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(f"[INFO] Checkpoints -> {run_dir}")
        return run_dir

    @torch.no_grad()
    def _eval_id(self, model: nn.Module, loader: DataLoader) -> dict:
        """Element-wise multi-label acc, split by empty vs occupied frames."""
        model.eval()
        correct = total = 0
        empty_ok = empty_elem = 0
        occ_ok = occ_elem = 0
        empty_frames = occ_frames = 0
        for b in loader:
            pred = (torch.sigmoid(model(b["image"].to(self.device))) > 0.5).float()
            gt = b["target"].to(self.device)
            match = pred == gt
            correct += int(match.sum().item())
            total += int(gt.numel())
            is_empty = gt.sum(dim=-1) == 0
            n_empty = int(is_empty.sum().item())
            n_occ = int((~is_empty).sum().item())
            if n_empty:
                empty_ok += int(match[is_empty].sum().item())
                empty_elem += n_empty * gt.shape[-1]
                empty_frames += n_empty
            if n_occ:
                occ_ok += int(match[~is_empty].sum().item())
                occ_elem += n_occ * gt.shape[-1]
                occ_frames += n_occ
        return {
            "elem_acc": correct / max(1, total),
            "empty_elem_acc": empty_ok / max(1, empty_elem),
            "occupied_elem_acc": occ_ok / max(1, occ_elem),
            "empty_n": empty_frames,
            "occupied_n": occ_frames,
        }

    def _train_subtask(self) -> Path:
        train_set, val_set = self._split(HumanSubtaskDataset)
        train_ld = DataLoader(train_set, self.cfg["batch_size"], shuffle=True, num_workers=self.cfg["num_workers"])
        val_ld = DataLoader(val_set, self.cfg["batch_size"], num_workers=self.cfg["num_workers"])
        model = HumanSubtaskModel(len(train_set.camera_order)).to(self.device)
        opt = torch.optim.AdamW(model.parameters(), lr=self.cfg["learning_rate"], weight_decay=self.cfg["weight_decay"])
        run_dir = Path(self.cfg["output_dir"]) / f"{self.cfg['run_name']}_subtask"
        run_dir.mkdir(parents=True, exist_ok=True)
        best, history = 0.0, []
        for epoch in range(self.cfg["num_epochs"]):
            model.train()
            loss_sum = 0.0
            for b in train_ld:
                imgs = b["images"].to(self.device)
                tid = b["task_id"].to(self.device)
                sub_logits, done_logit = model(imgs, tid)
                loss = F.cross_entropy(sub_logits, b["subtask_id"].to(self.device))
                loss = loss + F.binary_cross_entropy_with_logits(done_logit, b["subtask_done"].to(self.device))
                opt.zero_grad()
                loss.backward()
                opt.step()
                loss_sum += loss.item()
            m = self._eval_subtask(model, val_ld) | {"epoch": epoch, "train_loss": loss_sum / max(1, len(train_ld))}
            history.append(m)
            print(
                f"[subtask epoch {epoch:03d}] loss={m['train_loss']:.4f} "
                f"sub_acc={m['subtask_acc']:.3f} done_acc={m['done_acc']:.3f}"
            )
            ckpt = {
                "model_state_dict": model.state_dict(),
                "task": "subtask",
                "num_cameras": len(train_set.camera_order),
                "train_cfg": self.cfg,
            }
            torch.save(ckpt, run_dir / "last.pt")
            if m["subtask_acc"] >= best:
                best = m["subtask_acc"]
                torch.save(ckpt, run_dir / "best.pt")
        (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(f"[INFO] Checkpoints -> {run_dir}")
        return run_dir

    @torch.no_grad()
    def _eval_subtask(self, model: nn.Module, loader: DataLoader) -> dict:
        model.eval()
        sub_ok = done_ok = n = 0
        for b in loader:
            sub_logits, done_logit = model(b["images"].to(self.device), b["task_id"].to(self.device))
            sub_pred = sub_logits.argmax(dim=-1)
            done_pred = (torch.sigmoid(done_logit) > 0.5).long()
            sub_ok += int((sub_pred == b["subtask_id"].to(self.device)).sum().item())
            done_ok += int((done_pred == b["subtask_done"].to(self.device).long()).sum().item())
            n += b["subtask_id"].shape[0]
        return {"subtask_acc": sub_ok / max(1, n), "done_acc": done_ok / max(1, n)}


def main() -> None:
    p = argparse.ArgumentParser(description="HRTPA perception train/eval")
    p.add_argument("command", choices=["train", "eval"])
    p.add_argument("--task", choices=["id", "subtask"], default=CfgPerceptionTraining.get("task", "subtask"))
    p.add_argument("--dataset_dir", default=CfgPerceptionTraining["dataset_dir"])
    p.add_argument("--output_dir", default=CfgPerceptionTraining["output_dir"])
    p.add_argument("--run_name", default=CfgPerceptionTraining["run_name"])
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--epochs", type=int, default=CfgPerceptionTraining["num_epochs"])
    p.add_argument("--batch_size", type=int, default=CfgPerceptionTraining["batch_size"])
    p.add_argument("--device", default=CfgPerceptionTraining["device"])
    args = p.parse_args()

    cfg = copy.deepcopy(CfgPerceptionTraining) | {
        "task": args.task,
        "dataset_dir": args.dataset_dir,
        "output_dir": args.output_dir,
        "run_name": args.run_name,
        "num_epochs": args.epochs,
        "batch_size": args.batch_size,
        "device": args.device,
    }
    trainer = PerceptionTrainer(cfg)
    if args.command == "train":
        trainer.train()
        return

    if not args.checkpoint:
        raise ValueError("--checkpoint required for eval")
    payload = torch.load(args.checkpoint, map_location=trainer.device, weights_only=False)
    _, _, test_eps = trainer._episode_split()
    eval_eps = test_eps or None
    task = payload.get("task", args.task)
    if task == "id":
        model = HumanIdModel().to(trainer.device)
        model.load_state_dict(payload["model_state_dict"])
        metrics = trainer._eval_id(
            model,
            DataLoader(
                HumanIdDataset(args.dataset_dir, trainer.cfg["image_size"], eval_eps),
                args.batch_size,
            ),
        )
    else:
        n_cam = payload.get("num_cameras", len([c for _, cams in CAMERA_POSES.items() for c in cams]))
        model = HumanSubtaskModel(n_cam).to(trainer.device)
        model.load_state_dict(payload["model_state_dict"])
        metrics = trainer._eval_subtask(
            model,
            DataLoader(
                HumanSubtaskDataset(args.dataset_dir, trainer.cfg["image_size"], eval_eps),
                args.batch_size,
            ),
        )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
