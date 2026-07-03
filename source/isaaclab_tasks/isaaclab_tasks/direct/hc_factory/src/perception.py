"""Human subtask perception: sim logging (collect/infer) and offline training."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

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
        CoupledDoneRules,
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
        CoupledDoneRules,
        PerceptionSampleTemplate,
    )

_COUPLED_DONE = set(CoupledDoneRules)
_SKIP_PARTNER = frozenset({"wait", "done", "none"})
_WALK = frozenset({"go_to_material", "go_to_goal_area", "go_to_processing_machine"})
_OPERATE = frozenset(
    {"material_on_gantry", "control_gantry", "material_on_robot", "material_on_goal_area", "control_machine"}
)
_TASK_RECORD_KEYS = (
    "task", "task_type", "product", "product_index", "human", "human_index", "robot",
    "target_machine", "chosen_machine_workstation", "chosen_gantry_index",
)
_SUBTASK_DICT_KEYS = (
    "ongoing_index", "num_subtasks", "ongoing", "finished", "material_start_area", "material_goal_area",
)


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


def _animation_pose(name: str | None) -> str | None:
    if name is None:
        return None
    if name in _WALK:
        return "walk"
    if name in _OPERATE:
        return "operate"
    return "idle"


def _signals_vec(sig: dict | None) -> list[float]:
    if not sig:
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


def _agent_signal(task_record: dict) -> dict:
    sd = task_record["subtasks_dict"]
    return {
        "task": task_record.get("task"),
        "task_type": task_record.get("task_type"),
        "human_key": task_record.get("human"),
        "robot_key": task_record.get("robot"),
        "subtask_index": sd.get("ongoing_index"),
        "num_subtasks": sd.get("num_subtasks"),
        "ongoing_row": list(sd.get("ongoing", [])),
        "finished": [bool(x) for x in sd.get("finished", [])],
        "material_start_area": sd.get("material_start_area"),
        "material_goal_area": sd.get("material_goal_area"),
    }


def extract_perception_state(env: dict, *, include_task_records: bool = True) -> tuple[dict, dict, dict]:
    """Extract human_labels, agent_signals, task_records from env_state_action_dict."""
    ongoing = env["progress"]["ongoing_task_records"]
    human_labels, agent_signals, task_records = {}, {}, {}

    for human_key, hs in env["human"].items():
        label = {
            "human_index": hs["key_variables"]["idx"],
            "state": hs["state"],
            "ongoing_task_record_index": hs["ongoing_task_record_index"],
            "current_area_id": hs["current_area_id"],
            "target_area_id": hs["target_area_id"],
            "subtask_time_counter": hs.get("subtask_time_counter", 0),
            "task": None, "task_type": None, "product_index": None,
            "subtask_index": None, "subtask_name": None, "subtask_done": None, "animation_pose": None,
        }
        rid = hs["ongoing_task_record_index"]
        if rid is not None and rid in ongoing:
            tr, sd = ongoing[rid], ongoing[rid]["subtasks_dict"]
            row, fin = sd["ongoing"], sd["finished"]
            label.update(
                task=tr.get("task"), task_type=tr.get("task_type"), product_index=rid,
                subtask_index=sd.get("ongoing_index"), subtask_name=row[AGENT_COL_HUMAN],
                subtask_done=bool(fin[AGENT_COL_HUMAN]),
            )
            if label["state"].startswith("working_"):
                label["animation_pose"] = _animation_pose(label["subtask_name"])
        human_labels[human_key] = label

    for pid, tr in ongoing.items():
        key = str(pid)
        agent_signals[key] = _agent_signal(tr)
        if include_task_records:
            sd = tr.get("subtasks_dict") or {}
            task_records[key] = to_serializable(
                {k: tr.get(k) for k in _TASK_RECORD_KEYS}
                | {"subtasks_dict": {k: sd.get(k) for k in _SUBTASK_DICT_KEYS}}
            )
    return human_labels, agent_signals, task_records


def build_text_context(env: dict, human_labels: dict, agent_signals: dict) -> str:
    lines = [f"time_step={env['time_step']}"]
    for k, lb in human_labels.items():
        if lb["state"] == "free":
            lines.append(f"{k}: free")
        else:
            lines.append(
                f"{k}: state={lb['state']} task={lb['task']} "
                f"subtask[{lb['subtask_index']}]={lb['subtask_name']} done={lb['subtask_done']} area={lb['current_area_id']}"
            )
    for pid, sig in agent_signals.items():
        lines.append(f"task_record[{pid}]: {sig['task']} row={sig['ongoing_row']} finished={sig['finished']}")
    return "\n".join(lines)


def apply_constraint_propagation(
    human_subtask: str | None, human_done: bool, agent_signals: dict, human_key: str,
) -> tuple[str | None, bool]:
    if not human_subtask:
        return human_subtask, human_done
    for sig in agent_signals.values():
        if sig.get("human_key") != human_key:
            continue
        row, fin = sig.get("ongoing_row") or [], sig.get("finished") or []
        if len(row) <= AGENT_COL_HUMAN or row[AGENT_COL_HUMAN] != human_subtask:
            continue
        for col in (AGENT_COL_GANTRY, AGENT_COL_MACHINE, AGENT_COL_ROBOT):
            if len(row) <= col or len(fin) <= col or row[col] in _SKIP_PARTNER:
                continue
            if (human_subtask, col, row[col]) in _COUPLED_DONE and fin[col]:
                human_done = True
    return human_subtask, human_done


def _rgb_numpy(rgb) -> np.ndarray | None:
    if rgb is None:
        return None
    arr = rgb.detach().cpu().numpy() if isinstance(rgb, torch.Tensor) else np.asarray(rgb)
    return arr if arr.size else None


def _resize_rgb_tensor(rgb, size: int, device: torch.device | None = None) -> torch.Tensor:
    t = rgb.float() / 255.0 if isinstance(rgb, torch.Tensor) and rgb.dtype == torch.uint8 else torch.as_tensor(rgb).float()
    if t.dim() == 3 and t.shape[-1] == 3:
        t = t.permute(2, 0, 1)
    t = F.interpolate(t.unsqueeze(0), size=(size, size), mode="bilinear", align_corners=False).squeeze(0)
    return t.to(device) if device else t


def _stack_camera_tensors(camera_states: dict, num_cameras: int, size: int, device: torch.device) -> torch.Tensor:
    frames = [_resize_rgb_tensor(s["rgb"], size, device) for s in camera_states.values() if s.get("rgb") is not None]
    if not frames:
        return torch.zeros(3, size, size, device=device)
    stacked = torch.stack(frames[:num_cameras])
    while stacked.shape[0] < num_cameras:
        stacked = torch.cat([stacked, stacked[:1]])
    return stacked


# ---------------------------------------------------------------------------
# Simulation hook
# ---------------------------------------------------------------------------


class PerceptionManager:
    def __init__(self, env_id: int, cuda_device: torch.device, cfg: dict | None = None):
        self.env_id, self.cuda_device = env_id, cuda_device
        self.cfg = copy.deepcopy(cfg or CfgPerception)
        self.enabled = bool(self.cfg.get("enabled", False))
        self.mode = self.cfg.get("mode", "off")
        self._logger = PerceptionLogger(env_id, self.cfg) if self.enabled and self.mode == "collect" else None
        self._model: MultimodalSubtaskModel | None = None
        self._vocab: list[str] = list(CfgPerceptionTraining["subtask_vocab"])
        if self.enabled and self.mode == "infer":
            self._load_checkpoint()

    def _load_checkpoint(self) -> None:
        path = self.cfg.get("checkpoint_path")
        if not path or not Path(path).is_file():
            print(f"[WARN] Perception checkpoint missing: {path}")
            return
        payload = torch.load(path, map_location=self.cuda_device, weights_only=False)
        self._vocab = payload.get("subtask_vocab", self._vocab)
        tc = payload.get("train_cfg", CfgPerceptionTraining)
        self._model = MultimodalSubtaskModel(
            payload.get("num_cameras", 1), len(self._vocab), payload.get("signal_dim", 8),
        ).to(self.cuda_device)
        self._model.load_state_dict(payload["model_state_dict"])
        self._model.eval()
        print(f"[INFO] Perception model loaded: {path}")

    def reset(self, env: dict) -> dict:
        env.setdefault("perception", {})["predictions"], env["perception"]["last_sample"] = {}, None
        if self._logger:
            self._logger.reset()
        return env

    def step(self, env: dict) -> dict:
        if not self.enabled or self.mode == "off":
            return env
        env.setdefault("perception", {})
        if self.mode == "collect" and self._logger:
            env["perception"]["last_sample"] = self._logger.maybe_log(env)
        elif self.mode == "infer" and self._model:
            env["perception"]["predictions"] = self._infer(env)
        return env

    def _infer(self, env: dict) -> dict[str, dict]:
        signals = extract_perception_state(env, include_task_records=False)[1]
        size = CfgPerceptionTraining["image_size"]
        imgs = _stack_camera_tensors(env.get("camera", {}), self._model.num_cameras, size, self.cuda_device)
        sig = torch.tensor(_signals_vec(next(iter(signals.values()), None)), device=self.cuda_device)
        with torch.no_grad():
            logits, done_logit = self._model(imgs.unsqueeze(0), sig.unsqueeze(0))
            idx = int(logits.argmax(-1).item())
            name, done = self._vocab[idx], bool(torch.sigmoid(done_logit).item() > 0.5)
        preds = {}
        for hk in env["human"]:
            n, d = (apply_constraint_propagation(name, done, signals, hk)
                    if self.cfg.get("use_constraint_propagation", True) else (name, done))
            preds[hk] = {"subtask_name": n, "subtask_done": d, "subtask_index_pred": idx}
        return preds


class PerceptionLogger:
    def __init__(self, env_id: int, cfg: dict):
        self.env_id, self.cfg = env_id, cfg
        self.output_dir = Path(cfg["output_dir"])
        self.episode_id = self.step_id = 0
        self._ep_dir: Path | None = None

    def reset(self) -> None:
        if self.episode_id >= self.cfg["max_episodes"]:
            return
        self._ep_dir = self.output_dir / f"env_{self.env_id:02d}_episode_{self.episode_id:06d}"
        self._ep_dir.mkdir(parents=True, exist_ok=True)
        if self.episode_id == 0:
            (self.output_dir / "manifest.json").write_text(
                json.dumps({"env_id": self.env_id, "sample_template": PerceptionSampleTemplate,
                            "cfg": to_serializable(self.cfg)}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        self.step_id = 0
        self.episode_id += 1

    def maybe_log(self, env: dict) -> dict | None:
        if not self._ep_dir or self.episode_id > self.cfg["max_episodes"]:
            return None
        if env["time_step"] % self.cfg["save_interval"] or (
            self.cfg.get("max_steps_per_episode") is not None and self.step_id >= self.cfg["max_steps_per_episode"]
        ):
            return None

        labels, signals, records = extract_perception_state(env, include_task_records=self.cfg["serialize_task_records"])
        sample = {**PerceptionSampleTemplate, "episode_id": self.episode_id - 1, "step_id": self.step_id,
                  "time_step": env["time_step"], "env_id": self.env_id,
                  "camera_paths": self._save_images(env), "human_labels": labels, "agent_signals": signals,
                  "task_records": records,
                  "text_context": build_text_context(env, labels, signals) if self.cfg["build_text_context"] else ""}
        with (self._ep_dir / "meta.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(to_serializable(sample), ensure_ascii=False) + "\n")
        self.step_id += 1
        return sample

    def _save_images(self, env: dict) -> dict[str, str]:
        if not self.cfg.get("save_images", True):
            return {}
        assert self._ep_dir
        step_dir = self._ep_dir / "cameras" / f"step_{self.step_id:06d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        ext, paths = self.cfg.get("image_format", "jpg"), {}
        for key, st in env.get("camera", {}).items():
            arr = _rgb_numpy(st.get("rgb"))
            if arr is None or not st.get("is_initialized"):
                continue
            out = step_dir / f"{key}.{ext}"
            kw = {"format": "JPEG", "quality": int(self.cfg.get("image_quality", 90))} if ext.lower() in ("jpg", "jpeg") else {}
            Image.fromarray(arr.astype(np.uint8)).save(out, **kw)
            paths[key] = str(out.relative_to(self._ep_dir))
        return paths


# ---------------------------------------------------------------------------
# Offline training
# ---------------------------------------------------------------------------


def _load_samples(dataset_dir: Path) -> list[dict]:
    samples = []
    for meta in sorted(dataset_dir.glob("env_*_episode_*/meta.jsonl")):
        ep = meta.parent
        for line in meta.read_text(encoding="utf-8").splitlines():
            if line.strip():
                samples.append(json.loads(line) | {"_episode_dir": str(ep)})
    if not samples:
        raise FileNotFoundError(f"No samples under {dataset_dir}")
    return samples


class PerceptionDataset(Dataset):
    def __init__(self, dataset_dir: str | Path, subtask_vocab: list[str], image_size: int = 224):
        self.subtask_to_idx = {n: i for i, n in enumerate(subtask_vocab)}
        self.image_size = image_size
        self.samples = _load_samples(Path(dataset_dir))
        self.camera_keys = sorted({k for s in self.samples for k in s.get("camera_paths", {})})
        self.num_cameras = max(1, len(self.camera_keys))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> dict:
        s, ep = self.samples[i], Path(self.samples[i]["_episode_dir"])
        imgs = []
        for ck in self.camera_keys:
            rel = s.get("camera_paths", {}).get(ck)
            imgs.append(_resize_rgb_tensor(np.array(Image.open(ep / rel).convert("RGB"))) if rel
                         else torch.zeros(3, self.image_size, self.image_size))
        while len(imgs) < self.num_cameras:
            imgs.append(imgs[0].clone())
        imgs = torch.stack(imgs[: self.num_cameras])

        name, done, pid = "wait", 0, None
        for lb in s.get("human_labels", {}).values():
            if lb.get("state") != "free" and lb.get("subtask_name"):
                name, done, pid = lb["subtask_name"], int(bool(lb.get("subtask_done"))), lb.get("product_index")
                break
        sig = s.get("agent_signals", {}).get(str(pid)) if pid is not None else None
        return {
            "images": imgs,
            "signals": torch.tensor(_signals_vec(sig), dtype=torch.float32),
            "subtask_idx": torch.tensor(self.subtask_to_idx.get(name, self.subtask_to_idx.get("wait", 0)), dtype=torch.long),
            "subtask_done": torch.tensor(done, dtype=torch.float32),
        }


class MultimodalSubtaskModel(nn.Module):
    def __init__(self, num_cameras: int, num_subtasks: int, signal_dim: int = 8):
        super().__init__()
        self.num_cameras = num_cameras
        from torchvision import models
        self.backbone = nn.Sequential(*list(models.resnet18(weights=models.ResNet18_Weights.DEFAULT).children())[:-1])
        self.cam_enc = nn.Linear(512, 256)
        self.sig_enc = nn.Linear(signal_dim, 64)
        dim = 256 * num_cameras + 64
        self.head_cls = nn.Sequential(nn.Linear(dim, 256), nn.ReLU(True), nn.Dropout(0.2), nn.Linear(256, num_subtasks))
        self.head_done = nn.Sequential(nn.Linear(dim, 64), nn.ReLU(True), nn.Linear(64, 1))

    def forward(self, images: torch.Tensor, signals: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feats = [self.cam_enc(self.backbone(images[:, i]).flatten(1)) for i in range(images.shape[1])]
        fused = torch.cat([torch.cat(feats, dim=-1), self.sig_enc(signals)], dim=-1)
        return self.head_cls(fused), self.head_done(fused).squeeze(-1)


class PerceptionTrainer:
    def __init__(self, cfg: dict | None = None):
        self.cfg = copy.deepcopy(cfg or CfgPerceptionTraining)
        self.device = torch.device(self.cfg["device"] if torch.cuda.is_available() else "cpu")

    def train(self) -> Path:
        ds = PerceptionDataset(self.cfg["dataset_dir"], self.cfg["subtask_vocab"], self.cfg["image_size"])
        n_val = max(1, int(len(ds) * self.cfg["val_ratio"]))
        train_set, val_set = random_split(ds, [len(ds) - n_val, n_val], generator=torch.Generator().manual_seed(42))
        train_ld = DataLoader(train_set, self.cfg["batch_size"], shuffle=True,
                              num_workers=self.cfg["num_workers"], pin_memory=True)
        val_ld = DataLoader(val_set, self.cfg["batch_size"], num_workers=self.cfg["num_workers"], pin_memory=True)
        model = MultimodalSubtaskModel(ds.num_cameras, len(self.cfg["subtask_vocab"])).to(self.device)
        opt = torch.optim.AdamW(model.parameters(), lr=self.cfg["learning_rate"], weight_decay=self.cfg["weight_decay"])
        run_dir = Path(self.cfg["output_dir"]) / self.cfg["run_name"]
        run_dir.mkdir(parents=True, exist_ok=True)
        best, history = 0.0, []

        for epoch in range(self.cfg["num_epochs"]):
            model.train()
            loss_sum = 0.0
            for b in train_ld:
                img, sig = b["images"].to(self.device), b["signals"].to(self.device)
                logits, done = model(img, sig)
                loss = F.cross_entropy(logits, b["subtask_idx"].to(self.device)) + F.binary_cross_entropy_with_logits(
                    done, b["subtask_done"].to(self.device))
                opt.zero_grad(); loss.backward(); opt.step()
                loss_sum += loss.item()
            m = self.evaluate(model, val_ld) | {"epoch": epoch, "train_loss": loss_sum / max(1, len(train_ld))}
            history.append(m)
            print(f"[epoch {epoch:03d}] loss={m['train_loss']:.4f} subtask_acc={m['subtask_acc']:.3f} done_acc={m['done_acc']:.3f}")
            ckpt = {"model_state_dict": model.state_dict(), "subtask_vocab": self.cfg["subtask_vocab"],
                    "num_cameras": ds.num_cameras, "signal_dim": 8, "train_cfg": self.cfg}
            torch.save(ckpt, run_dir / "last.pt")
            if m["subtask_acc"] >= best:
                best = m["subtask_acc"]
                torch.save(ckpt, run_dir / "best.pt")
        (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(f"[INFO] Checkpoints -> {run_dir}")
        return run_dir

    @torch.no_grad()
    def evaluate(self, model: nn.Module, loader: DataLoader) -> dict:
        model.eval()
        ok_cls = ok_done = total = 0
        for b in loader:
            img, sig = b["images"].to(self.device), b["signals"].to(self.device)
            logits, done = model(img, sig)
            ok_cls += (logits.argmax(-1) == b["subtask_idx"].to(self.device)).sum().item()
            ok_done += ((torch.sigmoid(done) > 0.5).long() == b["subtask_done"].long().to(self.device)).sum().item()
            total += img.shape[0]
        return {"subtask_acc": ok_cls / max(1, total), "done_acc": ok_done / max(1, total)}


def main() -> None:
    p = argparse.ArgumentParser(description="HRTPA perception train/eval")
    p.add_argument("command", choices=["train", "eval"])
    p.add_argument("--dataset_dir", default=CfgPerceptionTraining["dataset_dir"])
    p.add_argument("--output_dir", default=CfgPerceptionTraining["output_dir"])
    p.add_argument("--run_name", default=CfgPerceptionTraining["run_name"])
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--epochs", type=int, default=CfgPerceptionTraining["num_epochs"])
    p.add_argument("--batch_size", type=int, default=CfgPerceptionTraining["batch_size"])
    p.add_argument("--device", default=CfgPerceptionTraining["device"])
    args = p.parse_args()

    cfg = copy.deepcopy(CfgPerceptionTraining) | {
        "dataset_dir": args.dataset_dir, "output_dir": args.output_dir, "run_name": args.run_name,
        "num_epochs": args.epochs, "batch_size": args.batch_size, "device": args.device,
    }
    trainer = PerceptionTrainer(cfg)
    if args.command == "train":
        trainer.train()
    else:
        if not args.checkpoint:
            raise ValueError("--checkpoint required for eval")
        payload = torch.load(args.checkpoint, map_location=trainer.device, weights_only=False)
        model = MultimodalSubtaskModel(payload["num_cameras"], len(payload["subtask_vocab"])).to(trainer.device)
        model.load_state_dict(payload["model_state_dict"])
        print(json.dumps(trainer.evaluate(model, DataLoader(
            PerceptionDataset(args.dataset_dir, cfg["subtask_vocab"]), args.batch_size)), indent=2))


if __name__ == "__main__":
    main()
