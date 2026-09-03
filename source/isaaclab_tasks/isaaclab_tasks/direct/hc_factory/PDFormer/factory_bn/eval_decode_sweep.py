"""Sweep P-safe decode on a saved ckpt using the official episode test split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

_PDFORMER_ROOT = Path(__file__).resolve().parent.parent
if str(_PDFORMER_ROOT) not in sys.path:
    sys.path.insert(0, str(_PDFORMER_ROOT))

from factory_bn.dataset import build_dataloaders, make_pattern_keys  # noqa: E402
from factory_bn.model import BNPDFormer  # noqa: E402
from factory_bn.train import _epoch_loop, _load_init_ckpt, _resolve_run_path  # noqa: E402


def _load_cfg(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    cfg = _load_cfg(Path(args.config))
    cfg["device"] = args.device
    device = torch.device(args.device)
    data_dir = _resolve_run_path(args.data_dir)
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
        occupancy_horizon_windows=int(cfg.get("occupancy_horizon_windows", 15)),
        train_mode=str(cfg.get("train_mode") or "unsupervised"),
    )
    del train_loader, val_loader
    data_feature["pattern_keys"] = make_pattern_keys(
        data_feature.pop("train_feature_windows"),
        s_attn_size=int(cfg.get("s_attn_size", 3)),
        n_cluster=int(cfg.get("n_cluster", 16)),
        output_channel=int(cfg.get("pattern_channel", 4)),
    )
    model = BNPDFormer(cfg, data_feature).to(device)
    _load_init_ckpt(model, _resolve_run_path(args.ckpt), device)
    ev_kw = dict(
        event_iou_min=0.5,
        event_min_windows=8,
        event_report_threshold=float(cfg.get("event_report_threshold", 0.65)),
        start_tol_windows=3,
        ongoing_will_floor=0.62,
        force_ongoing_will=False,
    )

    sweeps = [
        ("baseline", dict()),
        ("force_ongoing", dict(force_ongoing_will=True, ongoing_will_floor=0.62)),
        (
            "force+gantry78",
            dict(
                force_ongoing_will=True,
                ongoing_will_floor=0.62,
                event_report_threshold_by_type={"gantry": 0.78},
            ),
        ),
        (
            "force+precursor58",
            dict(
                force_ongoing_will=True,
                ongoing_will_floor=0.62,
                recall_lift_threshold=0.58,
                recall_lift_cluster_ids=[1, 2],
                recall_lift_types=["machine", "workbench"],
            ),
        ),
        (
            "force+floor055",
            dict(force_ongoing_will=True, ongoing_will_floor=0.55),
        ),
    ]
    print(f"[sweep] test windows={len(test_loader.dataset)} ckpt={args.ckpt}")
    for name, attrs in sweeps:
        model.force_ongoing_will = bool(attrs.get("force_ongoing_will", False))
        model.ongoing_will_floor = float(attrs.get("ongoing_will_floor", 0.62))
        model.recall_lift_threshold = float(attrs.get("recall_lift_threshold", 0.0))
        model.recall_lift_cluster_ids = list(attrs.get("recall_lift_cluster_ids") or [])
        model.recall_lift_types = list(attrs.get("recall_lift_types") or [])
        model.event_report_threshold_by_type = dict(
            attrs.get("event_report_threshold_by_type") or {}
        )
        ev_kw["force_ongoing_will"] = model.force_ongoing_will
        ev_kw["ongoing_will_floor"] = model.ongoing_will_floor
        te = _epoch_loop(
            model,
            test_loader,
            None,
            device,
            train=False,
            cause_majority=int(data_feature.get("cause_majority", -1)),
            hot_eval_threshold=0.55,
            **ev_kw,
        )
        print(
            f"[{name}] P={te.get('report_precision', 0):.3f} "
            f"R={te.get('report_recall', 0):.3f} "
            f"F1={te.get('report_f1', 0):.3f} "
            f"onR={te.get('report_recall_ongoing', 0):.3f} "
            f"upR={te.get('report_recall_upcoming', 0):.3f} "
            f"n_pred={te.get('n_pred_who', 0):.0f}"
        )


if __name__ == "__main__":
    main()
