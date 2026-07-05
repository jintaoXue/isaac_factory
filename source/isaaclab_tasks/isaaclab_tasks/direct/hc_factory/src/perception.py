"""Human subtask perception: sim logging (collect/infer) and offline training."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

_HC_FACTORY = Path(__file__).resolve().parents[1]
_CFG_DIR = _HC_FACTORY / "env_asset_cfg"


def _load_cfg_perception_module():
    """Load cfg_perception without importing isaaclab_tasks (avoids Isaac Sim dependency)."""
    pkg = "hc_factory_env_asset_cfg"
    for name in (
        "cfg_machine",
        "cfg_process_subtask_gallery",
        "cfg_process_task_gallery",
        "cfg_perception",
    ):
        full = f"{pkg}.{name}"
        if full in sys.modules:
            continue
        path = _CFG_DIR / f"{name}.py"
        spec = importlib.util.spec_from_file_location(full, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load {path}")
        mod = importlib.util.module_from_spec(spec)
        mod.__package__ = pkg
        sys.modules[full] = mod
        spec.loader.exec_module(mod)
    return sys.modules[f"{pkg}.cfg_perception"]


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
    _cfg = _load_cfg_perception_module()
    AGENT_COL_GANTRY = _cfg.AGENT_COL_GANTRY
    AGENT_COL_HUMAN = _cfg.AGENT_COL_HUMAN
    AGENT_COL_MACHINE = _cfg.AGENT_COL_MACHINE
    AGENT_COL_ROBOT = _cfg.AGENT_COL_ROBOT
    CfgPerception = _cfg.CfgPerception
    CfgPerceptionTraining = _cfg.CfgPerceptionTraining
    CoupledDoneRules = _cfg.CoupledDoneRules
    PerceptionSampleTemplate = _cfg.PerceptionSampleTemplate

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split

_COUPLED_DONE = set(CoupledDoneRules)
_SKIP_PARTNER = frozenset({"wait", "done", "none"})
SIGNAL_DIM = 6  # human 列 finished/wait 不输入（为预测目标）


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



def _signals_vec(sig: dict | None) -> list[float]:
    """Partner + progress cues only; human-column finished/wait masked (prediction target)."""
    if not sig:
        return [0.0] * SIGNAL_DIM
    finished = (sig.get("finished") or [False] * 4) + [False] * 4
    row = sig.get("ongoing_row") or [""] * 4
    return [
        float(sig.get("subtask_index") or 0),
        float(sig.get("num_subtasks") or 0),
        float(finished[AGENT_COL_GANTRY]),
        float(finished[AGENT_COL_MACHINE]),
        float(finished[AGENT_COL_ROBOT]),
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


def _is_working_human(label: dict) -> bool:
    return label.get("state") != "free" and label.get("subtask_name") is not None


def _human_text(human_key: str, lb: dict, time_step: int) -> str:
    return (
        f"time_step={time_step} {human_key}: task={lb.get('task')} "
        f"subtask[{lb.get('subtask_index')}]={lb.get('subtask_name')} "
        f"done={lb.get('subtask_done')} area={lb.get('current_area_id')}"
    )


def build_human_sample(
    env: dict, env_id: int, episode_id: int, step_id: int,
    human_key: str, label: dict, camera_paths: dict, agent_signal: dict | None,
    *, text_context: str = "",
) -> dict:
    pid = label.get("product_index")
    return {
        "episode_id": episode_id,
        "step_id": step_id,
        "time_step": env["time_step"],
        "env_id": env_id,
        "human_key": human_key,
        "human_index": label["human_index"],
        "product_index": pid,
        "task": label.get("task"),
        "subtask_index": label.get("subtask_index"),
        "subtask_name": label.get("subtask_name"),
        "subtask_done": label.get("subtask_done"),
        "area_id": label.get("current_area_id"),
        "camera_paths": camera_paths,
        "agent_signal": agent_signal or {},
        "text_context": text_context,
    }


def extract_perception_state(env: dict) -> tuple[dict, dict]:
    """Return (human_labels, agent_signals) for logging / inference."""
    ongoing = env["progress"]["ongoing_task_records"]
    human_labels, agent_signals = {}, {}

    for human_key, hs in env["human"].items():
        label = {
            "human_index": hs["key_variables"]["idx"],
            "state": hs["state"],
            "ongoing_task_record_index": hs["ongoing_task_record_index"],
            "current_area_id": hs["current_area_id"],
            "target_area_id": hs["target_area_id"],
            "subtask_time_counter": hs.get("subtask_time_counter", 0),
            "task": None, "task_type": None, "product_index": None,
            "subtask_index": None, "subtask_name": None, "subtask_done": None,
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
        human_labels[human_key] = label

    for pid, tr in ongoing.items():
        agent_signals[str(pid)] = _agent_signal(tr)
    return human_labels, agent_signals


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
        self._model: SubtaskDoneModel | None = None
        if self.enabled and self.mode == "infer":
            self._load_checkpoint()

    def _load_checkpoint(self) -> None:
        path = self.cfg.get("checkpoint_path")
        if not path or not Path(path).is_file():
            print(f"[WARN] Perception checkpoint missing: {path}")
            return
        payload = torch.load(path, map_location=self.cuda_device, weights_only=False)
        self._model = SubtaskDoneModel(
            payload.get("num_cameras", 1), payload.get("signal_dim", SIGNAL_DIM),
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
        labels, signals = extract_perception_state(env)
        if not self._model:
            return {}
        size = CfgPerceptionTraining["image_size"]
        imgs = _stack_camera_tensors(env.get("camera", {}), self._model.num_cameras, size, self.cuda_device)
        use_sig = CfgPerceptionTraining.get("use_agent_signals", True)
        preds = {}
        for hk, lb in labels.items():
            if lb.get("state") == "free":
                continue
            pid = lb.get("product_index")
            sig = signals.get(str(pid)) if pid is not None else None
            sig_t = torch.tensor(_signals_vec(sig if use_sig else None), device=self.cuda_device)
            with torch.no_grad():
                done_logit = self._model(imgs.unsqueeze(0), sig_t.unsqueeze(0))
                done = bool(torch.sigmoid(done_logit).item() > 0.5)
            row = (sig or {}).get("ongoing_row") or []
            name = row[AGENT_COL_HUMAN] if len(row) > AGENT_COL_HUMAN else lb.get("subtask_name")
            if self.cfg.get("use_constraint_propagation", True):
                name, done = apply_constraint_propagation(name, done, signals, hk)
            preds[hk] = {"subtask_name": name, "subtask_done": done}
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

    def maybe_log(self, env: dict) -> list[dict] | None:
        if not self._ep_dir or self.episode_id > self.cfg["max_episodes"]:
            return None
        if env["time_step"] % self.cfg["save_interval"] or (
            self.cfg.get("max_steps_per_episode") is not None and self.step_id >= self.cfg["max_steps_per_episode"]
        ):
            return None

        labels, signals = extract_perception_state(env)
        working = {hk: lb for hk, lb in labels.items() if _is_working_human(lb)}
        if not working:
            return None

        camera_paths = self._save_images(env)
        episode_id = self.episode_id - 1
        written: list[dict] = []
        with (self._ep_dir / "meta.jsonl").open("a", encoding="utf-8") as f:
            for hk, lb in working.items():
                pid = lb.get("product_index")
                sig = signals.get(str(pid)) if pid is not None else None
                text = _human_text(hk, lb, env["time_step"]) if self.cfg.get("build_text_context") else ""
                sample = build_human_sample(
                    env, self.env_id, episode_id, self.step_id, hk, lb, camera_paths, sig, text_context=text,
                )
                f.write(json.dumps(to_serializable(sample), ensure_ascii=False) + "\n")
                written.append(sample)
        self.step_id += 1
        return written

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


def _legacy_row_to_sample(row: dict, episode_dir: str) -> list[dict]:
    """Expand old meta.jsonl rows (one step, many humans) into per-human samples."""
    out: list[dict] = []
    camera_paths = row.get("camera_paths", {})
    agent_signals = row.get("agent_signals", {})
    for hk, lb in row.get("human_labels", {}).items():
        if not _is_working_human(lb):
            continue
        pid = lb.get("product_index")
        out.append({
            "episode_id": row.get("episode_id"),
            "step_id": row.get("step_id"),
            "time_step": row.get("time_step"),
            "env_id": row.get("env_id"),
            "human_key": hk,
            "human_index": lb.get("human_index"),
            "product_index": pid,
            "task": lb.get("task"),
            "subtask_index": lb.get("subtask_index"),
            "subtask_name": lb.get("subtask_name"),
            "subtask_done": lb.get("subtask_done"),
            "area_id": lb.get("current_area_id"),
            "camera_paths": camera_paths,
            "agent_signal": agent_signals.get(str(pid), {}),
            "text_context": row.get("text_context", ""),
            "_episode_dir": episode_dir,
        })
    return out


def _load_samples(dataset_dir: Path) -> list[dict]:
    samples: list[dict] = []
    for meta in sorted(dataset_dir.glob("env_*_episode_*/meta.jsonl")):
        ep = str(meta.parent)
        for line in meta.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("human_key"):
                samples.append(row | {"_episode_dir": ep})
            else:
                samples.extend(_legacy_row_to_sample(row, ep))
    if not samples:
        raise FileNotFoundError(f"No samples under {dataset_dir}")
    return samples


class PerceptionDataset(Dataset):
    def __init__(self, dataset_dir: str | Path, image_size: int = 224):
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
            imgs.append(_resize_rgb_tensor(np.array(Image.open(ep / rel).convert("RGB")), self.image_size) if rel
                         else torch.zeros(3, self.image_size, self.image_size))
        while len(imgs) < self.num_cameras:
            imgs.append(imgs[0].clone())
        imgs = torch.stack(imgs[: self.num_cameras])

        use_sig = CfgPerceptionTraining.get("use_agent_signals", True)
        sig = s.get("agent_signal") if use_sig else None
        return {
            "images": imgs,
            "signals": torch.tensor(_signals_vec(sig), dtype=torch.float32),
            "subtask_done": torch.tensor(int(bool(s.get("subtask_done"))), dtype=torch.float32),
        }


class SubtaskDoneModel(nn.Module):
    def __init__(self, num_cameras: int, signal_dim: int = SIGNAL_DIM):
        super().__init__()
        self.num_cameras = num_cameras
        from torchvision import models
        self.backbone = nn.Sequential(*list(models.resnet18(weights=models.ResNet18_Weights.DEFAULT).children())[:-1])
        self.cam_enc = nn.Linear(512, 256)
        self.sig_enc = nn.Linear(signal_dim, 64)
        dim = 256 * num_cameras + 64
        self.head_done = nn.Sequential(nn.Linear(dim, 64), nn.ReLU(True), nn.Linear(64, 1))

    def forward(self, images: torch.Tensor, signals: torch.Tensor) -> torch.Tensor:
        feats = [self.cam_enc(self.backbone(images[:, i]).flatten(1)) for i in range(images.shape[1])]
        fused = torch.cat([torch.cat(feats, dim=-1), self.sig_enc(signals)], dim=-1)
        return self.head_done(fused).squeeze(-1)


def done_confusion_counts(gt: torch.Tensor, pred: torch.Tensor) -> dict[str, int]:
    """GT/pred: 0=not_done, 1=done. Returns four confusion cells."""
    gt_b = gt.long().view(-1)
    pred_b = pred.long().view(-1)
    return {
        "done_correct": int(((gt_b == 1) & (pred_b == 1)).sum().item()),
        "done_as_not_done": int(((gt_b == 1) & (pred_b == 0)).sum().item()),
        "not_done_correct": int(((gt_b == 0) & (pred_b == 0)).sum().item()),
        "not_done_as_done": int(((gt_b == 0) & (pred_b == 1)).sum().item()),
    }


def _confusion_notes(counts: dict[str, int], gt_done: int, gt_not_done: int) -> list[str]:
    notes: list[str] = []
    if gt_done == 0:
        notes.append("GT 中无 subtask_done=true 样本，done 相关指标（done_correct / done_as_not_done）无法评估")
    if gt_not_done == 0:
        notes.append("GT 中无 subtask_done=false 样本，not_done 相关指标无法评估")
    if gt_done > 0 and gt_done < 5:
        notes.append(f"GT done 样本仅 {gt_done} 条，done 类统计不可靠")
    if gt_not_done > 0 and gt_not_done < 5:
        notes.append(f"GT not_done 样本仅 {gt_not_done} 条，not_done 类统计不可靠")
    return notes


def summarize_done_confusion(counts: dict[str, int], total: int, gt_done: int, gt_not_done: int) -> dict:
    notes = _confusion_notes(counts, gt_done, gt_not_done)
    out = {
        **counts,
        "total": total,
        "gt_done": gt_done,
        "gt_not_done": gt_not_done,
        "done_acc": (counts["done_correct"] + counts["not_done_correct"]) / max(1, total),
        "done_recall": counts["done_correct"] / gt_done if gt_done else None,
        "not_done_recall": counts["not_done_correct"] / gt_not_done if gt_not_done else None,
        "data_notes": notes,
    }
    return out


class PerceptionTrainer:
    def __init__(self, cfg: dict | None = None):
        self.cfg = copy.deepcopy(cfg or CfgPerceptionTraining)
        self.device = torch.device(self.cfg["device"] if torch.cuda.is_available() else "cpu")

    def train(self) -> Path:
        ds = PerceptionDataset(self.cfg["dataset_dir"], self.cfg["image_size"])
        n_val = max(1, int(len(ds) * self.cfg["val_ratio"]))
        train_set, val_set = random_split(ds, [len(ds) - n_val, n_val], generator=torch.Generator().manual_seed(42))
        train_ld = DataLoader(train_set, self.cfg["batch_size"], shuffle=True,
                              num_workers=self.cfg["num_workers"], pin_memory=True)
        val_ld = DataLoader(val_set, self.cfg["batch_size"], num_workers=self.cfg["num_workers"], pin_memory=True)
        model = SubtaskDoneModel(ds.num_cameras, self.cfg.get("signal_dim", SIGNAL_DIM)).to(self.device)
        opt = torch.optim.AdamW(model.parameters(), lr=self.cfg["learning_rate"], weight_decay=self.cfg["weight_decay"])
        run_dir = Path(self.cfg["output_dir"]) / self.cfg["run_name"]
        run_dir.mkdir(parents=True, exist_ok=True)
        best, history = 0.0, []

        for epoch in range(self.cfg["num_epochs"]):
            model.train()
            loss_sum = 0.0
            for b in train_ld:
                img, sig = b["images"].to(self.device), b["signals"].to(self.device)
                done = model(img, sig)
                loss = F.binary_cross_entropy_with_logits(done, b["subtask_done"].to(self.device))
                opt.zero_grad(); loss.backward(); opt.step()
                loss_sum += loss.item()
            m = self.evaluate(model, val_ld) | {"epoch": epoch, "train_loss": loss_sum / max(1, len(train_ld))}
            history.append(m)
            c = m
            print(
                f"[epoch {epoch:03d}] loss={m['train_loss']:.4f} acc={m['done_acc']:.3f} | "
                f"done✓{c['done_correct']} done→¬{c['done_as_not_done']} "
                f"¬✓{c['not_done_correct']} ¬→done{c['not_done_as_done']}"
            )
            for note in c.get("data_notes", []):
                print(f"  [WARN] {note}")
            ckpt = {"model_state_dict": model.state_dict(), "num_cameras": ds.num_cameras,
                    "signal_dim": self.cfg.get("signal_dim", SIGNAL_DIM), "train_cfg": self.cfg}
            torch.save(ckpt, run_dir / "last.pt")
            if m["done_acc"] >= best:
                best = m["done_acc"]
                torch.save(ckpt, run_dir / "best.pt")
        (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(f"[INFO] Checkpoints -> {run_dir}")
        return run_dir

    @torch.no_grad()
    def evaluate(self, model: nn.Module, loader: DataLoader) -> dict:
        model.eval()
        totals = {"done_correct": 0, "done_as_not_done": 0, "not_done_correct": 0, "not_done_as_done": 0}
        gt_done = gt_not_done = total = 0
        for b in loader:
            img, sig = b["images"].to(self.device), b["signals"].to(self.device)
            gt = b["subtask_done"].to(self.device)
            pred = (torch.sigmoid(model(img, sig)) > 0.5).long()
            cell = done_confusion_counts(gt, pred)
            for k in totals:
                totals[k] += cell[k]
            gt_done += int((gt > 0.5).sum().item())
            gt_not_done += int((gt <= 0.5).sum().item())
            total += img.shape[0]
        return summarize_done_confusion(totals, total, gt_done, gt_not_done)


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
        model = SubtaskDoneModel(
            payload["num_cameras"], payload.get("signal_dim", SIGNAL_DIM),
        ).to(trainer.device)
        model.load_state_dict(payload["model_state_dict"])
        metrics = trainer.evaluate(model, DataLoader(
            PerceptionDataset(args.dataset_dir), args.batch_size))
        print(json.dumps(metrics, indent=2, ensure_ascii=False))
        print("\n--- 四格统计 ---")
        print(f"  done 判断正确 (TP):           {metrics['done_correct']}")
        print(f"  done 误判为 not_done (FN):    {metrics['done_as_not_done']}")
        print(f"  not_done 判断正确 (TN):       {metrics['not_done_correct']}")
        print(f"  not_done 误判为 done (FP):    {metrics['not_done_as_done']}")
        if metrics.get("data_notes"):
            print("\n--- 数据说明 ---")
            for note in metrics["data_notes"]:
                print(f"  • {note}")


if __name__ == "__main__":
    main()
