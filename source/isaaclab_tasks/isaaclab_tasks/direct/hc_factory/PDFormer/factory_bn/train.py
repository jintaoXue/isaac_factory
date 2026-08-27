"""Train BNPDFormer on exported FactoryBN episode arrays.

Example::

    cd source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/PDFormer
    python -m factory_bn.export_dataset \\
        --run_dir ../output/bottleneck_dataset/old_machine2.0 \\
        --out_dir raw_data/old2.0
    python -m factory_bn.train \\
        --config factory_bn/configs/FactoryBN.json \\
        --data_dir raw_data/old2.0 \\
        --save_dir libcity/cache/model_cache/old2.0 \\
        --max_epoch 5
    python -m factory_bn.train --config factory_bn/configs/FactoryBN.json --wandb_activate
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

# Ensure PDFormer root is on sys.path when run as module or script
_PDFORMER_ROOT = Path(__file__).resolve().parent.parent
if str(_PDFORMER_ROOT) not in sys.path:
    sys.path.insert(0, str(_PDFORMER_ROOT))

from factory_bn.dataset import build_dataloaders, make_pattern_keys
from factory_bn.model import BNPDFormer


def _load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def _jsonable_cfg(cfg: dict[str, Any], data_feature: dict[str, Any]) -> dict[str, Any]:
    """Hyper-params + data sizes for wandb.config (no tensors)."""
    out: dict[str, Any] = {}
    for k, v in cfg.items():
        if k == "device":
            out[k] = str(v)
            continue
        if isinstance(v, (str, int, float, bool)) or v is None:
            out[k] = v
        elif isinstance(v, (list, dict)):
            try:
                json.dumps(v)
            except TypeError:
                continue
            out[k] = v
    out["num_nodes"] = int(data_feature["num_nodes"])
    out["feature_dim"] = int(data_feature["feature_dim"])
    out["n_train"] = int(data_feature["n_train"])
    out["n_val"] = int(data_feature["n_val"])
    out["n_test"] = int(data_feature["n_test"])
    out["split_by"] = str(data_feature.get("split_by") or "episode")
    out["n_train_episodes"] = int(data_feature.get("n_train_episodes") or 0)
    out["n_val_episodes"] = int(data_feature.get("n_val_episodes") or 0)
    out["n_test_episodes"] = int(data_feature.get("n_test_episodes") or 0)
    out["train_episodes_by_run"] = data_feature.get("train_episodes_by_run") or {}
    out["val_episodes_by_run"] = data_feature.get("val_episodes_by_run") or {}
    out["test_episodes_by_run"] = data_feature.get("test_episodes_by_run") or {}
    out["n_event_positive_train"] = int(data_feature.get("n_event_positive_train") or 0)
    out["n_event_surv_train"] = int(data_feature.get("n_event_surv_train") or 0)
    out["n_cause_labeled_train"] = int(data_feature.get("n_cause_labeled_train") or 0)
    out["n_cause_classes"] = int(data_feature.get("n_cause_classes") or 0)
    out["cause_majority"] = int(data_feature.get("cause_majority", -1))
    return out


def _wandb_enabled(cfg: dict[str, Any]) -> bool:
    return bool(cfg.get("wandb_activate"))


def _init_wandb(cfg: dict[str, Any], data_feature: dict[str, Any]) -> Any:
    """Same entry style as repo-root ``train.py`` (project / name / resume=allow)."""
    try:
        import wandb
    except ImportError as exc:
        raise ImportError(
            "wandb_activate=true 需要先安装 wandb：pip install wandb && wandb login"
        ) from exc

    run_name = str(cfg.get("wandb_name") or "").strip()
    if not run_name:
        run_name = f"BNPDFormer_{time.strftime('%Y-%m-%d_%H-%M-%S')}"
    init_kw: dict[str, Any] = {
        "project": str(cfg.get("wandb_project") or "FactoryBN_PDFormer"),
        "name": run_name,
        "config": _jsonable_cfg(cfg, data_feature),
        "resume": "allow",
        "sync_tensorboard": False,
    }
    entity = str(cfg.get("wandb_entity") or "").strip()
    if entity:
        init_kw["entity"] = entity
    run = wandb.init(**init_kw)
    wandb.define_metric("epoch")
    wandb.define_metric("Train/*", step_metric="epoch")
    wandb.define_metric("Val/*", step_metric="epoch")
    wandb.define_metric("Test/*", step_metric="epoch")
    print(f"[wandb] project={init_kw['project']} name={run_name} url={getattr(run, 'url', '')}")
    return run


def _wandb_log(parts: dict[str, dict[str, Any]], epoch: int) -> None:
    """One wandb.log per step so Train/Val/Test share the same epoch axis."""
    import wandb

    payload: dict[str, Any] = {"epoch": epoch}
    for prefix, metrics in parts.items():
        for k, v in metrics.items():
            if isinstance(v, (int, float)) and k != "epoch":
                payload[f"{prefix}/{k}"] = float(v)
    wandb.log(payload, step=epoch)


def _masked_score_mae(
    pred: torch.Tensor, target: torch.Tensor, remain_mask: torch.Tensor | None
) -> float:
    if remain_mask is None:
        return float(torch.mean(torch.abs(pred - target)).item())
    m = remain_mask.unsqueeze(-1).unsqueeze(-1)
    denom = m.sum().clamp_min(1.0)
    return float(((pred - target).abs() * m).sum().item() / denom.item())


def _near_remain_mask(remain_mask: torch.Tensor | None, near_k: int) -> torch.Tensor | None:
    if remain_mask is None:
        return None
    steps = torch.arange(remain_mask.shape[-1], device=remain_mask.device)
    return remain_mask.float() * (steps < max(int(near_k), 1)).float()


def _occupancy_eval_mask(
    remain_mask: torch.Tensor | None,
    occ_node_mask: torch.Tensor | None,
) -> torch.Tensor | None:
    """(B, K, N) bool: near remaining windows × labeled occupancy columns."""
    if remain_mask is None:
        return None
    step = remain_mask > 0.5
    if occ_node_mask is None:
        return step.unsqueeze(-1)
    node = occ_node_mask
    if node.dim() == 2:
        return step.unsqueeze(-1) & (node > 0.5).unsqueeze(1)
    if node.dim() == 3:
        return step.unsqueeze(-1) & (node > 0.5)
    return step.unsqueeze(-1)


def _average_precision(y_true: torch.Tensor, y_score: torch.Tensor) -> float:
    """Non-interpolated AP (sklearn-style) for a 1-d binary label / score."""
    y = y_true.reshape(-1).float()
    s = y_score.reshape(-1).float()
    n_pos = float(y.sum().item())
    if n_pos <= 0 or y.numel() == 0:
        return 0.0
    try:
        order = torch.argsort(s, descending=True, stable=True)
    except TypeError:
        order = torch.argsort(s, descending=True)
    y = y[order]
    tp = torch.cumsum(y, dim=0)
    prec = tp / torch.arange(1, y.numel() + 1, device=y.device, dtype=torch.float32)
    return float((prec * y).sum().item() / n_pos)


def _will_metrics(y_true: torch.Tensor, y_prob: torch.Tensor, thresh: float = 0.5) -> dict[str, float]:
    y = y_true.reshape(-1).float()
    p = y_prob.reshape(-1).float()
    hat = (p >= thresh).float()
    tp = float(((hat == 1) & (y == 1)).sum().item())
    fp = float(((hat == 1) & (y == 0)).sum().item())
    fn = float(((hat == 0) & (y == 1)).sum().item())
    tn = float(((hat == 0) & (y == 0)).sum().item())
    prec = tp / max(tp + fp, 1.0)
    rec = tp / max(tp + fn, 1.0)
    f1 = (2.0 * prec * rec / max(prec + rec, 1e-8)) if (prec + rec) > 0 else 0.0
    acc = (tp + tn) / max(tp + fp + fn + tn, 1.0)
    return {
        "will_acc": acc,
        "will_precision": prec,
        "will_recall": rec,
        "will_f1": f1,
        "will_ap": _average_precision(y, p),
        "will_pos_rate": float(y.mean().item()) if y.numel() else 0.0,
        "will_pred_pos_rate": float(hat.mean().item()) if hat.numel() else 0.0,
    }


OCC_EVAL_TYPES = ("machine", "gantry", "agv")


def _harmonic_mean(values: list[float]) -> float:
    xs = [max(float(v), 1e-8) for v in values]
    if not xs:
        return 0.0
    return len(xs) / sum(1.0 / x for x in xs)


def _model_type_masks(model: BNPDFormer) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for name in OCC_EVAL_TYPES:
        buf = getattr(model, f"occ_type_{name}", None)
        if buf is not None and buf.numel() > 0 and float(buf.sum()) > 0:
            out[name] = buf
    return out


def _epoch_loop(
    model: BNPDFormer,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    train: bool,
    cause_majority: int = -1,
    hot_eval_threshold: float = 0.5,
) -> dict[str, float]:
    if train:
        model.train()
    else:
        model.eval()

    totals: dict[str, float] = {}
    n = 0
    score_mae = 0.0
    remain_len_mae = 0.0
    remain_n = 0
    hot_true: list[torch.Tensor] = []
    hot_prob: list[torch.Tensor] = []
    hot_true_t: dict[str, list[torch.Tensor]] = {t: [] for t in OCC_EVAL_TYPES}
    hot_prob_t: dict[str, list[torch.Tensor]] = {t: [] for t in OCC_EVAL_TYPES}
    type_masks = _model_type_masks(model)
    correct_cause = 0
    total_cause = 0
    majority_hit = 0
    will_true: list[torch.Tensor] = []
    will_prob: list[torch.Tensor] = []

    for batch in loader:
        batch = _move_batch(batch, device)
        if train:
            assert optimizer is not None
            optimizer.zero_grad(set_to_none=True)
            loss, stats = model.calculate_loss(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
        else:
            with torch.enable_grad():
                loss, stats = model.calculate_loss(batch)
            with torch.no_grad():
                pred = model.predict(batch)
                will_true.append(batch["will"].detach().cpu())
                will_prob.append(pred["will_prob"].detach().cpu())
                near_k = int(getattr(model, "near_remain_windows", 60) or 60)
                near_m = _near_remain_mask(batch.get("remain_mask"), near_k)
                score_mae += float(
                    _masked_score_mae(pred["score_pred"], batch["y_score"], near_m)
                ) * batch["X"].shape[0]
                if "remain_len" in batch and "remain_len_pred" in pred:
                    remain_len_mae += float(
                        torch.mean(torch.abs(pred["remain_len_pred"] - batch["remain_len"])).item()
                    ) * batch["X"].shape[0]
                    remain_n += batch["X"].shape[0]
                if "y_hot" in batch and "hot_prob" in pred and near_m is not None:
                    cell = _occupancy_eval_mask(near_m, batch.get("occ_node_mask"))
                    if cell is not None and cell.any():
                        hot_true.append(batch["y_hot"][cell].detach().cpu().reshape(-1))
                        hot_prob.append(pred["hot_prob"][cell].detach().cpu().reshape(-1))
                        for name, nmask in type_masks.items():
                            tcell = cell & nmask.to(device=cell.device).bool().view(1, 1, -1)
                            if tcell.any():
                                hot_true_t[name].append(
                                    batch["y_hot"][tcell].detach().cpu().reshape(-1)
                                )
                                hot_prob_t[name].append(
                                    pred["hot_prob"][tcell].detach().cpu().reshape(-1)
                                )
                if "cause" in batch and "cause_pred" in pred:
                    valid = batch["cause"] >= 0
                    if "window_hot" in batch:
                        valid = valid & (batch["window_hot"].reshape(-1) > 0.5)
                    if valid.any():
                        y = batch["cause"][valid]
                        hat = pred["cause_pred"][valid]
                        correct_cause += int((hat == y).sum().item())
                        total_cause += int(valid.sum().item())
                        if cause_majority >= 0:
                            majority_hit += int((y == cause_majority).sum().item())

        for k, v in stats.items():
            totals[k] = totals.get(k, 0.0) + v
        n += 1

    out = {k: v / max(n, 1) for k, v in totals.items()}
    if not train:
        out["score_mae"] = score_mae / max(len(loader.dataset), 1)
        if will_true:
            out.update(_will_metrics(torch.cat(will_true), torch.cat(will_prob)))
        if remain_n > 0:
            out["remain_len_mae"] = remain_len_mae / remain_n
        if hot_true:
            hm = _will_metrics(torch.cat(hot_true), torch.cat(hot_prob), thresh=hot_eval_threshold)
            out["hot_f1"] = hm["will_f1"]
            out["hot_precision"] = hm["will_precision"]
            out["hot_recall"] = hm["will_recall"]
            out["hot_ap"] = hm["will_ap"]
            out["hot_pos_rate"] = hm["will_pos_rate"]
            out["hot_pred_pos_rate"] = hm["will_pred_pos_rate"]
            type_f1s: list[float] = []
            for name in OCC_EVAL_TYPES:
                if not hot_true_t[name]:
                    continue
                tm = _will_metrics(
                    torch.cat(hot_true_t[name]),
                    torch.cat(hot_prob_t[name]),
                    thresh=hot_eval_threshold,
                )
                out[f"hot_f1_{name}"] = tm["will_f1"]
                out[f"hot_precision_{name}"] = tm["will_precision"]
                out[f"hot_recall_{name}"] = tm["will_recall"]
                out[f"hot_pos_rate_{name}"] = tm["will_pos_rate"]
                out[f"hot_pred_pos_rate_{name}"] = tm["will_pred_pos_rate"]
                if tm["will_pos_rate"] > 0:
                    type_f1s.append(tm["will_f1"])
            out["hot_type_hmean"] = (
                _harmonic_mean(type_f1s) if type_f1s else float(out["hot_f1"])
            )
        if total_cause > 0:
            out["cause_acc"] = correct_cause / total_cause
            out["cause_n"] = float(total_cause)
            if cause_majority >= 0:
                out["cause_majority_acc"] = majority_hit / total_cause
    return out


def train(cfg: dict[str, Any]) -> Path:
    device = torch.device(
        cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    )
    data_dir = Path(cfg["data_dir"])
    if not data_dir.is_absolute():
        data_dir = (_PDFORMER_ROOT / data_dir).resolve()
    npz_path = data_dir / "episodes.npz"
    if not npz_path.is_file():
        raise SystemExit(
            f"No episodes.npz under {data_dir}. Pass a dataset folder, e.g.\n"
            "  --data_dir raw_data/new1.0 \\\n"
            "  --save_dir libcity/cache/model_cache/new1.0"
        )

    train_loader, val_loader, test_loader, data_feature = build_dataloaders(
        data_dir=data_dir,
        input_window=int(cfg.get("input_window", 30)),
        output_window=int(cfg.get("output_window", 1)),
        horizon_s=float(cfg.get("horizon_s", 180)),
        batch_size=int(cfg.get("batch_size", 16)),
        train_ratio=float(cfg.get("train_rate", 0.7)),
        val_ratio=float(cfg.get("eval_rate", 0.15)),
        seed=int(cfg.get("seed", 42)),
        max_hist_events=int(cfg.get("max_hist_events", 8)),
        remain_to_jobs_done=bool(cfg.get("remain_to_jobs_done", True)),
        max_remain_windows=int(cfg.get("max_remain_windows", 15)),
        hot_score_threshold=float(cfg.get("hot_score_threshold", 0.55)),
        occupancy_horizon_windows=int(
            cfg.get("occupancy_horizon_windows", cfg.get("max_remain_windows", 15))
        ),
        hot_min_windows=int(cfg.get("hot_min_windows", 2)),
        hot_gap_windows=int(cfg.get("hot_gap_windows", 1)),
        train_only_contains=list(cfg.get("train_only_contains") or []),
    )

    pattern_keys = make_pattern_keys(
        data_feature["train_feature_windows"],
        s_attn_size=int(cfg.get("s_attn_size", 3)),
        n_cluster=int(cfg.get("n_cluster", 16)),
        output_channel=int(cfg.get("pattern_channel", 4)),
    )
    data_feature["pattern_keys"] = pattern_keys
    train_windows = data_feature.pop("train_feature_windows")

    cfg = dict(cfg)
    cfg["device"] = device
    model = BNPDFormer(cfg, data_feature).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.get("learning_rate", 1e-3)),
        weight_decay=float(cfg.get("weight_decay", 0.05)),
    )
    max_epoch = int(cfg.get("max_epoch", 50))
    patience = int(cfg.get("patience", 15))
    save_dir = Path(cfg.get("save_dir", "libcity/cache/model_cache"))
    if not save_dir.is_absolute():
        save_dir = (_PDFORMER_ROOT / save_dir).resolve()
    if save_dir.name == "model_cache":
        raise SystemExit(
            f"save_dir is the cache root ({save_dir}). Pass a run folder, e.g.\n"
            "  --save_dir libcity/cache/model_cache/new1.0"
        )
    save_dir.mkdir(parents=True, exist_ok=True)

    best_mae = float("inf")
    best_hot_f1 = -1.0
    best_ckpt = -1.0
    best_val_loss = float("inf")
    best_path = save_dir / "BNPDFormer_best.pt"
    stale = 0
    last_epoch = 0
    epoch_log: list[dict[str, Any]] = []

    def _ckpt_payload(epoch: int, val_metrics: dict[str, float]) -> dict[str, Any]:
        return {
            "model": model.state_dict(),
            "config": {k: (str(v) if k == "device" else v) for k, v in cfg.items()},
            "data_meta": {
                "resource_ids": data_feature["resource_ids"],
                "resource_types": data_feature["resource_types"],
                "num_nodes": data_feature["num_nodes"],
                "feature_dim": data_feature["feature_dim"],
                "cause_classes": data_feature.get("cause_classes"),
                "n_cause_classes": data_feature.get("n_cause_classes"),
                "cause_majority": cause_majority,
                "pattern_keys": pattern_keys,
                "adj_mx": data_feature["adj_mx"],
                "sh_mx": data_feature["sh_mx"],
                "sem_mx": data_feature["sem_mx"],
                "feature_scaler_mean": data_feature["feature_scaler"].mean,
                "feature_scaler_std": data_feature["feature_scaler"].std,
                "score_scaler_mean": data_feature["score_scaler"].mean,
                "score_scaler_std": data_feature["score_scaler"].std,
                "remain_to_jobs_done": bool(cfg.get("remain_to_jobs_done", True)),
                "max_remain_windows": int(cfg.get("max_remain_windows", 512)),
            },
            "best_score_mae": best_mae,
            "best_hot_f1": best_hot_f1,
            "best_val_loss": best_val_loss,
            "ckpt_metric": str(cfg.get("ckpt_metric") or "hot_f1"),
            "epoch": epoch,
            "val_metrics": {k: float(v) for k, v in val_metrics.items() if isinstance(v, (int, float))},
        }

    print(
        f"[train] device={device} N={data_feature['num_nodes']} F={data_feature['feature_dim']} "
        f"train/val/test={data_feature['n_train']}/{data_feature['n_val']}/{data_feature['n_test']} "
        f"episodes={data_feature.get('n_train_episodes', '?')}/"
        f"{data_feature.get('n_val_episodes', '?')}/"
        f"{data_feature.get('n_test_episodes', '?')} "
        f"event+train={data_feature.get('n_event_positive_train', 0)} "
        f"event_surv_train={data_feature.get('n_event_surv_train', 0)} "
        f"cause+train={data_feature.get('n_cause_labeled_train', 0)} "
        f"remain_jobs={data_feature.get('remain_to_jobs_done', True)} "
        f"Kmax={data_feature.get('max_remain_windows', 512)} "
        f"ckpt={cfg.get('ckpt_metric', 'hot_f1')} pattern_keys={pattern_keys.shape}"
    )
    if data_feature.get("split_by") == "episode":
        print(
            f"[train] split=episode "
            f"train={data_feature.get('train_episodes_by_run')} "
            f"val={data_feature.get('val_episodes_by_run')} "
            f"test={data_feature.get('test_episodes_by_run')}"
        )
    cause_majority = int(data_feature.get("cause_majority", -1))
    hot_eval_threshold = float(cfg.get("hot_eval_threshold", 0.55))
    counts = data_feature.get("cause_train_counts")
    classes = data_feature.get("cause_classes") or []
    if counts is not None and classes:
        brief = ", ".join(
            f"{name}={int(counts[i])}"
            for i, name in enumerate(classes)
            if i < len(counts) and int(counts[i]) > 0
        )
        if brief:
            print(f"[train] cause train counts: {brief} majority={cause_majority}")

    wandb_run = None
    if _wandb_enabled(cfg):
        wandb_run = _init_wandb(cfg, data_feature)

    try:
        for epoch in range(1, max_epoch + 1):
            t0 = time.time()
            tr = _epoch_loop(model, train_loader, optimizer, device, train=True)
            va = _epoch_loop(
                model,
                val_loader,
                None,
                device,
                train=False,
                cause_majority=cause_majority,
                hot_eval_threshold=hot_eval_threshold,
            )
            dt = time.time() - t0
            last_epoch = epoch
            epoch_log.append(
                {
                    "epoch": epoch,
                    "train_loss": float(tr.get("loss", 0.0)),
                    "train_hot": float(tr.get("loss_hot", 0.0)),
                    "train_dice": float(tr.get("loss_dice", 0.0)),
                    "train_iou": float(tr.get("loss_iou", 0.0)),
                    "train_cause": float(tr.get("loss_cause", 0.0)),
                    "train_remain": float(tr.get("loss_remain_len", 0.0)),
                    "train_contrast": float(tr.get("loss_contrast", 0.0)),
                    "val_loss": float(va.get("loss", 0.0)),
                    "hot_f1": float(va.get("hot_f1", 0.0)),
                    "hot_precision": float(va.get("hot_precision", 0.0)),
                    "hot_recall": float(va.get("hot_recall", 0.0)),
                    "hot_type_hmean": float(va.get("hot_type_hmean", 0.0)),
                    "hot_precision_machine": float(va.get("hot_precision_machine", 0.0)),
                    "hot_precision_gantry": float(va.get("hot_precision_gantry", 0.0)),
                    "hot_precision_agv": float(va.get("hot_precision_agv", 0.0)),
                    "remain_len_mae": float(va.get("remain_len_mae", 0.0)),
                    "score_mae": float(va.get("score_mae", 0.0)),
                    "cause_acc": float(va.get("cause_acc", 0.0)),
                }
            )
            print(
                f"epoch {epoch:03d}/{max_epoch}  "
                f"train_loss={tr['loss']:.4f} (hot={tr.get('loss_hot', 0):.4f} "
                f"dice={tr.get('loss_dice', 0):.4f} iou={tr.get('loss_iou', 0):.4f} "
                f"cause={tr.get('loss_cause', 0):.4f} remain={tr.get('loss_remain_len', 0):.4f} "
                f"contrast={tr.get('loss_contrast', 0):.4f})  "
                f"val_loss={va['loss']:.4f} score_mae={va.get('score_mae', 0):.4f} "
                f"will_f1={va.get('will_f1', 0):.3f} will_p={va.get('will_precision', 0):.3f} "
                f"will_r={va.get('will_recall', 0):.3f} will_ap={va.get('will_ap', 0):.3f} "
                f"hot_f1={va.get('hot_f1', 0):.3f} hot_p={va.get('hot_precision', 0):.3f} "
                f"hot_r={va.get('hot_recall', 0):.3f} "
                f"m_p={va.get('hot_precision_machine', 0):.3f} "
                f"g_p={va.get('hot_precision_gantry', 0):.3f} "
                f"a_p={va.get('hot_precision_agv', 0):.3f} "
                f"type_h={va.get('hot_type_hmean', 0):.3f} "
                f"remain_mae={va.get('remain_len_mae', 0):.1f} "
                f"nll={va.get('nll', 0):.3f}  ({dt:.1f}s)"
            )

            ckpt_metric = str(cfg.get("ckpt_metric") or "hot_f1")
            min_hot_p = float(cfg.get("ckpt_min_hot_precision", 0.0))
            mae = float(va.get("score_mae", float("inf")))
            hot_f1 = float(va.get("hot_f1", 0.0))
            hot_p = float(va.get("hot_precision", 0.0))
            type_h = float(va.get("hot_type_hmean", 0.0))
            if ckpt_metric == "hot_type_hmean":
                improved = (hot_p + 1e-12 >= min_hot_p) and (type_h > best_ckpt + 1e-6)
                ckpt_show = type_h
            elif ckpt_metric == "hot_f1":
                improved = (hot_p + 1e-12 >= min_hot_p) and (hot_f1 > best_hot_f1 + 1e-6)
                ckpt_show = hot_f1
            else:
                improved = mae < best_mae - 1e-6
                ckpt_show = mae
            if improved:
                best_mae = min(best_mae, mae)
                best_hot_f1 = max(best_hot_f1, hot_f1)
                best_ckpt = max(best_ckpt, type_h if ckpt_metric == "hot_type_hmean" else hot_f1)
                stale = 0
                torch.save(_ckpt_payload(epoch, va), best_path)
                print(
                    f"  saved best {ckpt_metric}={ckpt_show:.4f}"
                    f" (hot_p={hot_p:.3f} type_h={type_h:.3f})"
                    f" -> {best_path}"
                )
                if wandb_run is not None:
                    import wandb

                    wandb.summary["best_score_mae"] = best_mae
                    wandb.summary["best_hot_f1"] = best_hot_f1
                    wandb.summary["best_epoch"] = epoch
            else:
                stale += 1

            loss_v = float(va.get("loss", float("inf")))
            if loss_v < best_val_loss - 1e-5:
                best_val_loss = loss_v

            if wandb_run is not None:
                _wandb_log(
                    {
                        "Train": {**tr, "epoch_sec": dt},
                        "Val": {
                            **va,
                            "best_score_mae": best_mae,
                            "best_hot_f1": best_hot_f1,
                            "best_loss": best_val_loss,
                        },
                    },
                    epoch,
                )

            if stale >= patience:
                print(f"early stop at epoch {epoch} ({ckpt_metric} patience={patience})")
                break

        if not best_path.is_file():
            fail_metrics = save_dir / "last_metrics.json"
            fail_metrics.write_text(
                json.dumps(
                    {
                        "ckpt_written": False,
                        "reason": "hot_precision_gate",
                        "ckpt_metric": str(cfg.get("ckpt_metric") or "hot_f1"),
                        "ckpt_min_hot_precision": float(cfg.get("ckpt_min_hot_precision", 0.0)),
                        "val_best_hot_f1": best_hot_f1,
                        "epochs": epoch_log,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            raise SystemExit(
                "No checkpoint written: val hot_precision never reached "
                f"ckpt_min_hot_precision={float(cfg.get('ckpt_min_hot_precision', 0.0)):.2f} "
                f"(best hot_f1 seen={best_hot_f1:.3f}). "
                "Do not keep a junk ckpt; check occupancy labels / collect."
            )
        try:
            ckpt = torch.load(best_path, map_location=device, weights_only=False)
        except TypeError:
            ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        te = _epoch_loop(
            model,
            test_loader,
            None,
            device,
            train=False,
            cause_majority=cause_majority,
            hot_eval_threshold=hot_eval_threshold,
        )
        print(
            f"[test] loss={te['loss']:.4f} score_mae={te.get('score_mae', 0):.4f} "
            f"will_f1={te.get('will_f1', 0):.3f} will_p={te.get('will_precision', 0):.3f} "
            f"will_r={te.get('will_recall', 0):.3f} will_ap={te.get('will_ap', 0):.3f} "
            f"hot_f1={te.get('hot_f1', 0):.3f} hot_p={te.get('hot_precision', 0):.3f} "
            f"hot_r={te.get('hot_recall', 0):.3f} "
            f"m_p={te.get('hot_precision_machine', 0):.3f} "
            f"g_p={te.get('hot_precision_gantry', 0):.3f} "
            f"a_p={te.get('hot_precision_agv', 0):.3f} "
            f"type_h={te.get('hot_type_hmean', 0):.3f} "
            f"remain_mae={te.get('remain_len_mae', 0):.1f} "
            f"nll={te.get('nll', 0):.3f} cause_acc={te.get('cause_acc', 0):.3f}"
        )
        metrics_path = save_dir / "last_metrics.json"
        metrics_path.write_text(
            json.dumps(
                {
                    "ckpt_metric": str(cfg.get("ckpt_metric") or "hot_f1"),
                    "val_best_score_mae": best_mae,
                    "val_best_hot_f1": best_hot_f1,
                    "val_best_hot_type_hmean": best_ckpt,
                    "val_best_loss": best_val_loss,
                    "best_epoch": int(ckpt.get("epoch") or 0),
                    "test": te,
                    "n_train_windows_used": int(train_windows.shape[0]),
                    "split_by": str(data_feature.get("split_by") or "episode"),
                    "n_train_episodes": int(data_feature.get("n_train_episodes") or 0),
                    "n_val_episodes": int(data_feature.get("n_val_episodes") or 0),
                    "n_test_episodes": int(data_feature.get("n_test_episodes") or 0),
                    "train_episodes_by_run": data_feature.get("train_episodes_by_run") or {},
                    "val_episodes_by_run": data_feature.get("val_episodes_by_run") or {},
                    "test_episodes_by_run": data_feature.get("test_episodes_by_run") or {},
                    "n_event_positive_train": int(data_feature.get("n_event_positive_train") or 0),
                    "n_event_surv_train": int(data_feature.get("n_event_surv_train") or 0),
                    "ckpt_min_hot_precision": float(cfg.get("ckpt_min_hot_precision", 0.0)),
                    "epochs": epoch_log,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        if wandb_run is not None:
            import wandb

            log_epoch = last_epoch if last_epoch > 0 else int(ckpt.get("epoch") or 0)
            _wandb_log({"Test": te}, log_epoch)
            wandb.summary["test_loss"] = te.get("loss")
            wandb.summary["test_will_acc"] = te.get("will_acc")
            wandb.summary["test_will_f1"] = te.get("will_f1")
            wandb.summary["test_will_ap"] = te.get("will_ap")
            wandb.summary["test_cause_acc"] = te.get("cause_acc")
            wandb.summary["test_score_mae"] = te.get("score_mae")
            wandb.summary["test_hot_f1"] = te.get("hot_f1")
            wandb.summary["test_remain_len_mae"] = te.get("remain_len_mae")
            wandb.summary["test_nll"] = te.get("nll")
            wandb.save(str(best_path))
            wandb.save(str(metrics_path))
    finally:
        if wandb_run is not None:
            import wandb

            wandb.finish()

    return best_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).parent / "configs" / "FactoryBN.json"),
    )
    parser.add_argument("--max_epoch", type=int, default=None)
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Dataset folder under raw_data/ (must contain episodes.npz).",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Run folder under libcity/cache/model_cache/ for weights.",
    )
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--ckpt_min_hot_precision",
        type=float,
        default=None,
        help="Override FactoryBN.json ckpt_min_hot_precision for this run.",
    )
    parser.add_argument(
        "--wandb_activate",
        action="store_true",
        default=None,
        help="Enable Weights & Biases logging (default off).",
    )
    parser.add_argument("--wandb_project", type=str, default=None, help="wandb project name.")
    parser.add_argument("--wandb_name", type=str, default=None, help="Optional wandb run name.")
    parser.add_argument("--wandb_entity", type=str, default=None, help="Optional wandb team/user.")
    parser.add_argument(
        "--train_only_contains",
        action="append",
        default=None,
        help="Episode names containing this substring stay in train (repeatable). "
        "Use n10_mix_ so mix OOD regularizers never enter val/test.",
    )
    args = parser.parse_args()

    cfg = _load_config(Path(args.config))
    if args.max_epoch is not None:
        cfg["max_epoch"] = args.max_epoch
    if args.data_dir is not None:
        cfg["data_dir"] = args.data_dir
    if args.save_dir is not None:
        cfg["save_dir"] = args.save_dir
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    if args.device is not None:
        cfg["device"] = args.device
    if args.ckpt_min_hot_precision is not None:
        cfg["ckpt_min_hot_precision"] = float(args.ckpt_min_hot_precision)
    if args.wandb_activate:
        cfg["wandb_activate"] = True
    if args.wandb_project:
        cfg["wandb_project"] = args.wandb_project
    if args.wandb_name:
        cfg["wandb_name"] = args.wandb_name
    if args.wandb_entity:
        cfg["wandb_entity"] = args.wandb_entity
    if args.train_only_contains:
        cfg["train_only_contains"] = list(args.train_only_contains)

    train(cfg)


if __name__ == "__main__":
    main()
