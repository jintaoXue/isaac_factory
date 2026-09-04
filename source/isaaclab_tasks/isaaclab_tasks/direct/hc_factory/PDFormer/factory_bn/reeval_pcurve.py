"""Re-evaluate dense_i1 ckpt after force_ongoing fix."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from factory_bn.dataset import build_dataloaders, make_pattern_keys
from factory_bn.model import BNPDFormer
from factory_bn.train import _epoch_loop, _load_init_ckpt


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    cfg = json.loads(
        (root / "factory_bn/configs/FactoryBN_dense_f1_p80.json").read_text(encoding="utf-8")
    )
    cfg["device"] = "cuda"
    device = torch.device("cuda")
    _, val_loader, test_loader, df = build_dataloaders(
        data_dir=(root / "raw_data/dense_i1").resolve(),
        input_window=30,
        output_window=1,
        horizon_s=180,
        batch_size=24,
        train_ratio=0.7,
        val_ratio=0.15,
        seed=42,
        max_hist_events=8,
        remain_to_jobs_done=True,
        max_remain_windows=15,
        occupancy_horizon_windows=15,
        train_mode="unsupervised",
    )
    df["pattern_keys"] = make_pattern_keys(
        df.pop("train_feature_windows"),
        s_attn_size=3,
        n_cluster=16,
        output_channel=4,
    )
    model = BNPDFormer(cfg, df).to(device)
    _load_init_ckpt(
        model,
        (root / "libcity/cache/model_cache/dense_i1_f1_p80v2/BNPDFormer_best.pt").resolve(),
        device,
    )
    model._report_threshold_sweep = [0.5, 0.55, 0.58, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
    model._report_ckpt_min_precision = 0.80

    # Also print raw curves without auto-pick
    for thr in [0.55, 0.65, 0.75, 0.8, 0.85, 0.9]:
        model._report_threshold_sweep = []
        ev = _epoch_loop(
            model,
            val_loader,
            None,
            device,
            train=False,
            cause_majority=int(df.get("cause_majority", -1)),
            hot_eval_threshold=0.52,
            event_iou_min=0.5,
            event_min_windows=6,
            event_report_threshold=float(thr),
            start_tol_windows=3,
            ongoing_will_floor=0.58,
            force_ongoing_will=True,
        )
        print(
            f"[val@thr={thr:.2f}] P={ev.get('report_precision', 0):.3f} "
            f"R={ev.get('report_recall', 0):.3f} F1={ev.get('report_f1', 0):.3f} "
            f"who_p={ev.get('who_precision', 0):.3f}",
            flush=True,
        )

    model._report_threshold_sweep = [0.5, 0.55, 0.58, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
    for name, loader in (("val", val_loader), ("test", test_loader)):
        ev = _epoch_loop(
            model,
            loader,
            None,
            device,
            train=False,
            cause_majority=int(df.get("cause_majority", -1)),
            hot_eval_threshold=0.52,
            event_iou_min=0.5,
            event_min_windows=6,
            event_report_threshold=0.58,
            start_tol_windows=3,
            ongoing_will_floor=0.58,
            force_ongoing_will=True,
        )
        print(
            f"[{name}-sweep] thr_used={ev.get('report_threshold_used')} "
            f"P={ev.get('report_precision', 0):.3f} "
            f"R={ev.get('report_recall', 0):.3f} F1={ev.get('report_f1', 0):.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
