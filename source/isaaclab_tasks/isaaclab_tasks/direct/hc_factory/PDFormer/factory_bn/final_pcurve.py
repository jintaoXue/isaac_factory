"""Val/test threshold curve for the precision-nudge checkpoint."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from factory_bn.dataset import build_dataloaders, make_pattern_keys
from factory_bn.model import BNPDFormer
from factory_bn.remain import station_report_metrics
from factory_bn.train import _load_init_ckpt, _move_batch, _near_remain_mask


def collect(model, loader, device):
    ys, ws, ss, ds, rs, os_, hs = [], [], [], [], [], [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = _move_batch(batch, device)
            pred = model.predict(batch)
            near = _near_remain_mask(batch.get("remain_mask"), 15)
            ys.append(batch["y_hot"].cpu())
            ws.append(pred["event_will_prob"].cpu())
            ss.append(pred["event_start_idx"].cpu())
            ds.append(pred["event_dur"].cpu())
            rs.append(near.cpu())
            occ = batch.get("occ_node_mask")
            if occ is not None:
                os_.append(occ.cpu())
            if "hist_last_hot" in batch:
                hs.append(batch["hist_last_hot"].cpu())
    y = torch.cat(ys).numpy()
    w = torch.cat(ws).numpy()
    s = torch.cat(ss).numpy()
    d = torch.cat(ds).numpy()
    r = torch.cat(rs).numpy()
    o = os_[0].numpy() if os_[0].dim() == 1 else torch.cat(os_).numpy()
    h = hs[0].numpy() if hs[0].dim() == 1 else torch.cat(hs).numpy()
    return y, w, s, d, r, o, h


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    cfg = json.loads((root / "factory_bn/configs/FactoryBN_dense_f1_p80.json").read_text(encoding="utf-8"))
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
        df.pop("train_feature_windows"), s_attn_size=3, n_cluster=16, output_channel=4
    )
    model = BNPDFormer(cfg, df).to(device)
    _load_init_ckpt(
        model,
        (root / "libcity/cache/model_cache/n10_plus20_prec_nudge/BNPDFormer_best.pt").resolve(),
        device,
    )
    model.force_ongoing_will = False
    for split, loader in (("val", val_loader), ("test", test_loader)):
        y, w, s, d, r, o, h = collect(model, loader, device)
        print(f"==== {split} ====", flush=True)
        best80 = None
        bestf = None
        for thr in (
            0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.78, 0.80,
            0.82, 0.85, 0.88, 0.90, 0.92, 0.95, 0.97, 0.99,
        ):
            ev = station_report_metrics(
                y, w, s, d, r, o,
                threshold=float(thr),
                min_windows=8,
                start_tol_windows=3,
                hist_last_hot=h,
                will_floor=0.65,
                force_ongoing_will=False,
            )
            p = float(ev["report_precision"])
            rec = float(ev["report_recall"])
            f1 = float(ev["report_f1"])
            up = float(ev.get("report_recall_upcoming", 0.0))
            on = float(ev.get("report_recall_ongoing", 0.0))
            print(
                f"  thr={thr:.2f} P={p:.3f} R={rec:.3f} F1={f1:.3f} up={up:.3f} on={on:.3f}",
                flush=True,
            )
            if bestf is None or f1 > bestf[0]:
                bestf = (f1, p, rec, thr)
            if p + 1e-12 >= 0.80 and (best80 is None or f1 > best80[0]):
                best80 = (f1, p, rec, thr)
        print(
            f"  maxF1: F1={bestf[0]:.3f} P={bestf[1]:.3f} R={bestf[2]:.3f} thr={bestf[3]:.2f}",
            flush=True,
        )
        if best80:
            print(
                f"  P>=0.80: F1={best80[0]:.3f} P={best80[1]:.3f} R={best80[2]:.3f} thr={best80[3]:.2f}",
                flush=True,
            )
        else:
            print("  no P>=0.80", flush=True)


if __name__ == "__main__":
    main()
