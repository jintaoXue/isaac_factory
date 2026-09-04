"""Dump event-will scores once, then sweep report thresholds in numpy."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from factory_bn.dataset import build_dataloaders, make_pattern_keys
from factory_bn.model import BNPDFormer
from factory_bn.remain import station_report_metrics
from factory_bn.train import _load_init_ckpt, _move_batch, _near_remain_mask


def _collect(model, loader, device, near_k: int):
    y_grids, will_grids, start_grids, dur_grids, remain_grids, occ_grids, hist_grids = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = _move_batch(batch, device)
            pred = model.predict(batch)
            near_m = _near_remain_mask(batch.get("remain_mask"), near_k)
            if near_m is None or "y_hot" not in batch or "event_will_prob" not in pred:
                continue
            y_grids.append(batch["y_hot"].detach().cpu())
            will_grids.append(pred["event_will_prob"].detach().cpu())
            start_grids.append(pred["event_start_idx"].detach().cpu())
            dur_grids.append(pred["event_dur"].detach().cpu())
            remain_grids.append(near_m.detach().cpu())
            occ = batch.get("occ_node_mask")
            if occ is not None:
                occ_grids.append(occ.detach().cpu())
            if "hist_last_hot" in batch:
                hist_grids.append(batch["hist_last_hot"].detach().cpu())
    y = torch.cat(y_grids, dim=0).numpy()
    will = torch.cat(will_grids, dim=0).numpy()
    start = torch.cat(start_grids, dim=0).numpy()
    dur = torch.cat(dur_grids, dim=0).numpy()
    remain = torch.cat(remain_grids, dim=0).numpy()
    if occ_grids:
        o0 = occ_grids[0]
        occ = o0.numpy() if o0.dim() == 1 else torch.cat(occ_grids, dim=0).numpy()
    else:
        occ = np.ones((y.shape[0], y.shape[2]), dtype=np.float32)
    last = None
    if hist_grids:
        h0 = hist_grids[0]
        last = h0.numpy() if h0.dim() == 1 else torch.cat(hist_grids, dim=0).numpy()
    return y, will, start, dur, remain, occ, last


def _print_curve(name: str, y, will, start, dur, remain, occ, last) -> None:
    print(f"==== {name} ====", flush=True)
    best80 = None
    for force in (False, True):
        for thr in (
            0.50,
            0.55,
            0.60,
            0.65,
            0.70,
            0.75,
            0.80,
            0.85,
            0.90,
            0.92,
            0.95,
            0.97,
            0.99,
        ):
            ev = station_report_metrics(
                y,
                will,
                start,
                dur,
                remain,
                occ,
                threshold=float(thr),
                min_windows=8,
                start_tol_windows=3,
                hist_last_hot=last,
                will_floor=0.62,
                force_ongoing_will=force,
            )
            p = float(ev.get("report_precision", 0.0))
            r = float(ev.get("report_recall", 0.0))
            f1 = float(ev.get("report_f1", 0.0))
            print(
                f"  force={int(force)} thr={thr:.2f} P={p:.3f} R={r:.3f} F1={f1:.3f} "
                f"up={float(ev.get('report_recall_upcoming', 0)):.3f} "
                f"on={float(ev.get('report_recall_ongoing', 0)):.3f}",
                flush=True,
            )
            if p + 1e-12 >= 0.80 and (best80 is None or f1 > best80[0]):
                best80 = (f1, p, r, thr, force)
    if best80:
        print(
            f"  BEST P>=0.80: F1={best80[0]:.3f} P={best80[1]:.3f} R={best80[2]:.3f} "
            f"thr={best80[3]:.2f} force={best80[4]}",
            flush=True,
        )
    else:
        print("  no operating point with P>=0.80", flush=True)


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    cfg = json.loads(
        (root / "factory_bn/configs/FactoryBN_dense_f1_p80.json").read_text(encoding="utf-8")
    )
    cfg["device"] = "cuda"
    device = torch.device("cuda")
    _, val_loader, test_loader, df = build_dataloaders(
        data_dir=(root / "raw_data/n10_plus20").resolve(),
        input_window=30,
        output_window=1,
        horizon_s=180,
        batch_size=16,
        train_ratio=0.7,
        val_ratio=0.15,
        seed=42,
        max_hist_events=8,
        remain_to_jobs_done=True,
        max_remain_windows=15,
        occupancy_horizon_windows=15,
        train_mode="unsupervised",
        train_only_contains=list(cfg.get("train_only_contains") or []),
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
        (root / "libcity/cache/model_cache/n10_plus20_f1_lift/BNPDFormer_best.pt").resolve(),
        device,
    )
    # Raw scores: disable decode hacks so the sweep sees true probabilities.
    model.force_ongoing_will = False
    model.recall_lift_threshold = 0.0
    model.event_report_threshold_by_type = {}
    model.event_onset_threshold = 0.0
    for name, loader in (("val", val_loader), ("test", test_loader)):
        packed = _collect(model, loader, device, 15)
        _print_curve(name, *packed)


if __name__ == "__main__":
    main()
