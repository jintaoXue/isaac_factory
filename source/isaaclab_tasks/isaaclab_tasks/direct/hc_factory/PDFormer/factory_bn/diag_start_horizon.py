"""How much of the recall gap is late-start events vs zero-will near events."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from factory_bn.dataset import build_dataloaders, make_pattern_keys
from factory_bn.model import BNPDFormer
from factory_bn.remain import _prf, node_event_targets
from factory_bn.train import _load_init_ckpt, _move_batch, _near_remain_mask


def _collect(model, loader, device):
    ys, ws, ss, rs, os_, hs = [], [], [], [], [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = _move_batch(batch, device)
            pred = model.predict(batch)
            near = _near_remain_mask(batch.get("remain_mask"), 15)
            ys.append(batch["y_hot"].cpu().numpy())
            ws.append(pred["event_will_prob"].cpu().numpy())
            ss.append(pred["event_start_idx"].cpu().numpy())
            rs.append(near.cpu().numpy())
            occ = batch.get("occ_node_mask")
            os_.append(occ.cpu().numpy() if occ is not None else np.ones(ys[-1].shape[-1]))
            hs.append(batch["hist_last_hot"].cpu().numpy())
    y = np.concatenate(ys, 0)
    w = np.concatenate(ws, 0)
    s = np.concatenate(ss, 0)
    r = np.concatenate(rs, 0)
    o0 = os_[0]
    o = o0 if np.asarray(o0).ndim == 1 else np.concatenate(os_, 0)
    h = hs[0] if np.asarray(hs[0]).ndim == 1 else np.concatenate(hs, 0)
    return y, w, s, r, o, h


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    cfg = json.loads((root / "factory_bn/configs/FactoryBN_dense_f1_p80.json").read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        root / "libcity/cache/model_cache/n10_plus20_p80_continue/BNPDFormer_best.pt",
        device,
    )
    model.force_ongoing_will = False
    model.event_union_occupancy = False
    model.event_report_threshold_by_type = {}
    model.recall_lift_threshold = 0.0

    for split, loader in (("val", val_loader), ("test", test_loader)):
        y, w, s, r, o, h = _collect(model, loader, device)
        y_will, y_start, y_dur = node_event_targets(y, min_windows=8, remain_mask=r, occ_node_mask=o)
        last = np.asarray(h, np.float32)
        if last.ndim == 1:
            last = np.broadcast_to(last.reshape(1, -1), y_will.shape)
        last = last[:, : y_will.shape[-1]]
        w = w[:, : y_will.shape[-1]]
        pred_s = np.where(last > 0.5, 0, np.asarray(s)[:, : y_will.shape[-1]])
        if o.ndim == 1:
            node_ok = np.broadcast_to(o.reshape(1, -1) > 0.5, y_will.shape)
        else:
            node_ok = o[:, : y_will.shape[-1]] > 0.5
        true_pos = (y_will > 0.5) & node_ok
        hot = last > 0.5
        upcoming = true_pos & ~hot
        print(f"\n== {split}", flush=True)
        print("  upcoming by y_start: n / will_mean / will>=0.70 / will>=0.05", flush=True)
        for st in range(0, 8):
            m = upcoming & (y_start == st)
            n = float(m.sum())
            if n <= 0:
                continue
            ww = w[m]
            print(
                f"    start={st} n={n:.0f} mean={ww.mean():.3f} "
                f"ge70={float((ww >= 0.70).sum()):.0f} ge05={float((ww >= 0.05).sum()):.0f}",
                flush=True,
            )

        for max_st in (2, 3, 4, 5, 6, 7):
            gt = true_pos & (y_start <= max_st)
            n_true = float(gt.sum())
            best = None
            for on_t in (0.50, 0.70, 0.80, 0.90):
                for up_t in (0.20, 0.50, 0.70, 0.85, 0.90):
                    pred = node_ok & (
                        (hot & (w >= on_t))
                        | ((~hot) & (w >= up_t) & (pred_s <= max_st))
                    )
                    hit = pred & gt & (np.abs(pred_s - y_start) <= 3)
                    n_pred = float(pred.sum())
                    tp = float(hit.sum())
                    p, rec, f1 = _prf(tp, n_pred - tp, n_true - tp)
                    if p + 1e-12 >= 0.80 and (best is None or f1 > best[0]):
                        best = (f1, p, rec, on_t, up_t, n_true, n_pred, tp)
            if best:
                print(
                    f"  max_start={max_st} P80-best F1={best[0]:.3f} P={best[1]:.3f} "
                    f"R={best[2]:.3f} on={best[3]:.2f} up={best[4]:.2f} n_true={best[5]:.0f}",
                    flush=True,
                )
            else:
                print(f"  max_start={max_st} no P>=80 n_true={n_true:.0f}", flush=True)


if __name__ == "__main__":
    main()
