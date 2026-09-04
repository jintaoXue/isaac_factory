"""Val/test decode ceiling: last_hot rules + will bars vs P80/F180."""

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
    ys, ws, ss, ds, rs, os_, hs, hots = [], [], [], [], [], [], [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = _move_batch(batch, device)
            pred = model.predict(batch)
            near = _near_remain_mask(batch.get("remain_mask"), 15)
            ys.append(batch["y_hot"].cpu().numpy())
            ws.append(pred["event_will_prob"].cpu().numpy())
            ss.append(pred["event_start_idx"].cpu().numpy())
            ds.append(pred["event_dur"].cpu().numpy())
            rs.append(near.cpu().numpy())
            occ = batch.get("occ_node_mask")
            os_.append(occ.cpu().numpy() if occ is not None else np.ones(ys[-1].shape[-1]))
            hs.append(batch["hist_last_hot"].cpu().numpy())
            if "hot_prob" in pred:
                hots.append(pred["hot_prob"].cpu().numpy())
    y = np.concatenate(ys, 0)
    w = np.concatenate(ws, 0)
    s = np.concatenate(ss, 0)
    d = np.concatenate(ds, 0)
    r = np.concatenate(rs, 0)
    o0 = os_[0]
    o = o0 if np.asarray(o0).ndim == 1 else np.concatenate(os_, 0)
    h = hs[0] if np.asarray(hs[0]).ndim == 1 else np.concatenate(hs, 0)
    hot = np.concatenate(hots, 0) if hots else None
    return y, w, s, d, r, o, h, hot


def _score(true_pos, report_ok, pred, n_up, n_on, upcoming, ongoing):
    pred = pred.astype(bool)
    hit = pred & true_pos & report_ok
    n_pred = float(pred.sum())
    tp = float(hit.sum())
    p, rec, f1 = _prf(tp, n_pred - tp, float(true_pos.sum()) - tp)
    up = float((hit & upcoming).sum()) / n_up if n_up else 0.0
    on = float((hit & ongoing).sum()) / n_on if n_on else 0.0
    return p, rec, f1, up, on, n_pred, tp


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
    ckpt = root / "libcity/cache/model_cache/n10_plus20_p80_continue/BNPDFormer_best.pt"
    _load_init_ckpt(model, ckpt, device)
    model.force_ongoing_will = False
    model.event_union_occupancy = False
    model.event_report_threshold_by_type = {}
    model.recall_lift_threshold = 0.0

    for split, loader in (("val", val_loader), ("test", test_loader)):
        y, w, s, d, r, o, h, hot = _collect(model, loader, device)
        y_will, y_start, _ = node_event_targets(y, min_windows=8, remain_mask=r, occ_node_mask=o)
        last = np.asarray(h, dtype=np.float32)
        if last.ndim == 1:
            last = np.broadcast_to(last.reshape(1, -1), y_will.shape)
        last = last[:, : y_will.shape[-1]]
        w = w[:, : y_will.shape[-1]]
        s = np.where(last > 0.5, 0, np.asarray(s)[:, : y_will.shape[-1]])
        d = np.asarray(d)[:, : y_will.shape[-1]]
        if o.ndim == 1:
            node_ok = np.broadcast_to(o.reshape(1, -1) > 0.5, y_will.shape)
        else:
            node_ok = o[:, : y_will.shape[-1]] > 0.5
        true_pos = (y_will > 0.5) & node_ok
        report_ok = np.abs(s.astype(np.int64) - y_start.astype(np.int64)) <= 3
        upcoming = true_pos & (last <= 0.5)
        ongoing = true_pos & (last > 0.5)
        n_true = float(true_pos.sum())
        n_up = float(upcoming.sum())
        n_on = float(ongoing.sum())
        hot_st = last > 0.5
        print(
            f"\n== {split} n_true={n_true:.0f} up={n_up:.0f} on={n_on:.0f} "
            f"last_hot={float(hot_st.sum()):.0f} last_hot&true={float((hot_st & true_pos).sum()):.0f}",
            flush=True,
        )
        # last_hot precision vs true event
        lh_pred = hot_st & node_ok
        p, rec, f1, up, on, n_pred, tp = _score(true_pos, report_ok, lh_pred, n_up, n_on, upcoming, ongoing)
        print(f"  last_hot only          P={p:.3f} R={rec:.3f} F1={f1:.3f} up={up:.3f} on={on:.3f} n_pred={n_pred:.0f}", flush=True)

        prefix = None
        if hot is not None:
            run = hot[:, :, : y_will.shape[-1]] >= 0.55
            prefix = np.cumprod(run, axis=1).sum(axis=1)
            occ_on = hot_st & (prefix >= 8) & node_ok
            p, rec, f1, up, on, n_pred, tp = _score(true_pos, report_ok, occ_on, n_up, n_on, upcoming, ongoing)
            print(f"  last_hot & occ_prefix>=8 P={p:.3f} R={rec:.3f} F1={f1:.3f} up={up:.3f} on={on:.3f} n_pred={n_pred:.0f}", flush=True)

        dur_on = hot_st & (d >= 8) & node_ok
        p, rec, f1, up, on, n_pred, tp = _score(true_pos, report_ok, dur_on, n_up, n_on, upcoming, ongoing)
        print(f"  last_hot & dur>=8      P={p:.3f} R={rec:.3f} F1={f1:.3f} up={up:.3f} on={on:.3f} n_pred={n_pred:.0f}", flush=True)

        best80 = None
        bestf = None
        for on_rule, on_mask in (
            ("none", np.zeros_like(hot_st, dtype=bool)),
            ("lh", hot_st & node_ok),
            ("lh_dur", hot_st & (d >= 8) & node_ok),
            ("lh_occ", hot_st & (prefix >= 8) & node_ok if prefix is not None else hot_st & (d >= 8) & node_ok),
        ):
            cold = ~hot_st
            for thr in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.88, 0.90, 0.92, 0.94, 0.96, 0.98, 0.99]:
                up_pred = cold & node_ok & (w >= thr)
                pred = on_mask | up_pred
                p, rec, f1, up, on, n_pred, tp = _score(true_pos, report_ok, pred, n_up, n_on, upcoming, ongoing)
                recs = (f1, p, rec, up, on, thr, on_rule, n_pred)
                if bestf is None or f1 > bestf[0]:
                    bestf = recs
                if p + 1e-12 >= 0.80 and (best80 is None or f1 > best80[0]):
                    best80 = recs
                if p + 1e-12 >= 0.80 and f1 + 1e-12 >= 0.80:
                    print(
                        f"  HIT P80 F180 rule={on_rule} thr={thr:.2f} "
                        f"P={p:.3f} R={rec:.3f} F1={f1:.3f} up={up:.3f} on={on:.3f}",
                        flush=True,
                    )
        print(
            f"  best F1: F1={bestf[0]:.3f} P={bestf[1]:.3f} R={bestf[2]:.3f} "
            f"up={bestf[3]:.3f} on={bestf[4]:.3f} thr={bestf[5]:.2f} rule={bestf[6]}",
            flush=True,
        )
        if best80:
            print(
                f"  best P>=80: F1={best80[0]:.3f} P={best80[1]:.3f} R={best80[2]:.3f} "
                f"up={best80[3]:.3f} on={best80[4]:.3f} thr={best80[5]:.2f} rule={best80[6]}",
                flush=True,
            )
        else:
            print("  no P>=0.80 in this grid", flush=True)

        # upcoming score gap
        if n_up and n_on:
            print(
                f"  will mean  up_true={w[upcoming].mean():.3f} on_true={w[ongoing].mean():.3f} "
                f"cold_neg={w[cold & node_ok & ~true_pos].mean():.3f} "
                f"hot_neg={w[hot_st & node_ok & ~true_pos].mean():.3f}",
                flush=True,
            )
            for q in (10, 25, 50, 75, 90):
                print(
                    f"  will p{q:02d} up_true={np.percentile(w[upcoming], q):.3f} "
                    f"cold_neg={np.percentile(w[cold & node_ok & ~true_pos], q):.3f}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
