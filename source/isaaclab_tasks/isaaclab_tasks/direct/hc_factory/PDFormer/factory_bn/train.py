"""Train BNPDFormer on exported FactoryBN episode arrays.

Example::

    cd source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/PDFormer
    python -m factory_bn.export_dataset \\
        --run_dir ../output/bottleneck_dataset/18_materials
    python -m factory_bn.train --config factory_bn/configs/FactoryBN.json --max_epoch 5
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


def _epoch_loop(
    model: BNPDFormer,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    train: bool,
) -> dict[str, float]:
    if train:
        model.train()
    else:
        model.eval()

    totals: dict[str, float] = {}
    n = 0
    correct_will = 0
    total_will = 0
    score_mae = 0.0

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
                will_hat = (pred["will_prob"] >= 0.5).float()
                correct_will += int((will_hat == batch["will"]).sum().item())
                total_will += batch["will"].numel()
                score_mae += float(
                    torch.mean(torch.abs(pred["score_pred"] - batch["y_score"])).item()
                ) * batch["X"].shape[0]

        for k, v in stats.items():
            totals[k] = totals.get(k, 0.0) + v
        n += 1

    out = {k: v / max(n, 1) for k, v in totals.items()}
    if not train and total_will > 0:
        out["will_acc"] = correct_will / total_will
        out["score_mae"] = score_mae / max(len(loader.dataset), 1)
    return out


def train(cfg: dict[str, Any]) -> Path:
    device = torch.device(
        cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    )
    data_dir = Path(cfg["data_dir"])
    if not data_dir.is_absolute():
        data_dir = (_PDFORMER_ROOT / data_dir).resolve()

    train_loader, val_loader, test_loader, data_feature = build_dataloaders(
        data_dir=data_dir,
        input_window=int(cfg.get("input_window", 12)),
        output_window=int(cfg.get("output_window", 1)),
        horizon_s=float(cfg.get("horizon_s", 180)),
        batch_size=int(cfg.get("batch_size", 16)),
        train_ratio=float(cfg.get("train_rate", 0.7)),
        val_ratio=float(cfg.get("eval_rate", 0.15)),
        seed=int(cfg.get("seed", 42)),
        max_hist_events=int(cfg.get("max_hist_events", 8)),
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
    save_dir = Path(cfg.get("save_dir", "libcity/cache/model_cache/FactoryBN"))
    if not save_dir.is_absolute():
        save_dir = (_PDFORMER_ROOT / save_dir).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    best_path = save_dir / "BNPDFormer_best.pt"
    stale = 0

    print(
        f"[train] device={device} N={data_feature['num_nodes']} F={data_feature['feature_dim']} "
        f"train/val/test={data_feature['n_train']}/{data_feature['n_val']}/{data_feature['n_test']} "
        f"event+train={data_feature.get('n_event_positive_train', 0)} "
        f"pattern_keys={pattern_keys.shape}"
    )

    for epoch in range(1, max_epoch + 1):
        t0 = time.time()
        tr = _epoch_loop(model, train_loader, optimizer, device, train=True)
        va = _epoch_loop(model, val_loader, None, device, train=False)
        dt = time.time() - t0
        print(
            f"epoch {epoch:03d}/{max_epoch}  "
            f"train_loss={tr['loss']:.4f} (score={tr['loss_score']:.4f} "
            f"will={tr['loss_will']:.4f} event={tr.get('loss_event', 0):.4f})  "
            f"val_loss={va['loss']:.4f} will_acc={va.get('will_acc', 0):.3f} "
            f"score_mae={va.get('score_mae', 0):.4f}  ({dt:.1f}s)"
        )
        if va["loss"] < best_val - 1e-5:
            best_val = va["loss"]
            stale = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": {k: (str(v) if k == "device" else v) for k, v in cfg.items()},
                    "data_meta": {
                        "resource_ids": data_feature["resource_ids"],
                        "resource_types": data_feature["resource_types"],
                        "num_nodes": data_feature["num_nodes"],
                        "feature_dim": data_feature["feature_dim"],
                        "pattern_keys": pattern_keys,
                        "adj_mx": data_feature["adj_mx"],
                        "sh_mx": data_feature["sh_mx"],
                        "sem_mx": data_feature["sem_mx"],
                        "feature_scaler_mean": data_feature["feature_scaler"].mean,
                        "feature_scaler_std": data_feature["feature_scaler"].std,
                        "score_scaler_mean": data_feature["score_scaler"].mean,
                        "score_scaler_std": data_feature["score_scaler"].std,
                    },
                    "best_val_loss": best_val,
                    "epoch": epoch,
                },
                best_path,
            )
            print(f"  saved best -> {best_path}")
        else:
            stale += 1
            if stale >= patience:
                print(f"early stop at epoch {epoch}")
                break

    try:
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    te = _epoch_loop(model, test_loader, None, device, train=False)
    print(
        f"[test] loss={te['loss']:.4f} will_acc={te.get('will_acc', 0):.3f} "
        f"score_mae={te.get('score_mae', 0):.4f}"
    )
    (save_dir / "last_metrics.json").write_text(
        json.dumps(
            {
                "val_best": best_val,
                "test": te,
                "n_train_windows_used": int(train_windows.shape[0]),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return best_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).parent / "configs" / "FactoryBN.json"),
    )
    parser.add_argument("--max_epoch", type=int, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    cfg = _load_config(Path(args.config))
    if args.max_epoch is not None:
        cfg["max_epoch"] = args.max_epoch
    if args.data_dir is not None:
        cfg["data_dir"] = args.data_dir
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    if args.device is not None:
        cfg["device"] = args.device

    train(cfg)


if __name__ == "__main__":
    main()
