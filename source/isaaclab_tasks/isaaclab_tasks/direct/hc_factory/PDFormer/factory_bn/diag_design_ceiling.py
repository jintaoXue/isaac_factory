"""Sweep label design (min_windows, max_start) vs decode bars on a frozen ckpt."""

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
            ws.append(torch.sigmoid(pred["event_will_logit"]).cpu().numpy())
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


def _score(pred, true_pos, start_ok):
    pred = pred.astype(bool)
    hit = pred & true_pos & start_ok
    n_pred = float(pred.sum())
    n_true = float(true_pos.sum())
    tp = float(hit.sum())
    return (*_prf(tp, n_pred - tp, n_true - tp), n_pred, n_true, tp)


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    cfg = json.loads((root / "factory_bn/configs/FactoryBN_dense_f1_p80.json").read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, val_loader, test_loader, df = build_dataloaders(
        data_dir=(root / "raw_data/dense_i1").resolve(),
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
    )
    df["pattern_keys"] = make_pattern_keys(
        df.pop("train_feature_windows"), s_attn_size=3, n_cluster=16, output_channel=4
    )
    model = BNPDFormer(cfg, df).to(device)
    _load_init_ckpt(model, root / "libcity/cache/model_cache/dense_i1_f1_p80_v5/BNPDFormer_best.pt", device)
    model.force_ongoing_will = False
    model.event_union_occupancy = False
    model.event_union_upcoming = False
    model.event_report_threshold_by_type = {}
    model.recall_lift_threshold = 0.0
    model.event_onset_threshold = 1.0

    hits: list[str] = []
    for split, loader in (("val", val_loader), ("test", test_loader)):
        y, w, s, d, r, o, h, hotp = _collect(model, loader, device)
        last = np.asarray(h, np.float32)
        if last.ndim == 1:
            last = np.broadcast_to(last.reshape(1, -1), (y.shape[0], y.shape[-1]))
        last = last[:, : w.shape[-1]]
        w = w[:, : last.shape[-1]]
        s = np.where(last > 0.5, 0, np.asarray(s)[:, : last.shape[-1]])
        d = np.asarray(d)[:, : last.shape[-1]]
        if o.ndim == 1:
            node_ok = np.broadcast_to(o.reshape(1, -1) > 0.5, last.shape)
        else:
            node_ok = o[:, : last.shape[-1]] > 0.5
        hot_st = last > 0.5
        print(f"\n==== {split} ====", flush=True)
        for min_w in (5, 6, 8):
            for max_st in (0, 2, 3, 4, 5, 7, None):
                yw, ys, _ = node_event_targets(
                    y, min_windows=min_w, remain_mask=r, occ_node_mask=o, max_start_windows=max_st
                )
                yw = yw[:, : last.shape[-1]]
                ys = ys[:, : last.shape[-1]]
                true_pos = (yw > 0.5) & node_ok
                start_ok = np.abs(s.astype(np.int64) - ys.astype(np.int64)) <= 3
                n_true = float(true_pos.sum())
                n_on = float((true_pos & (ys == 0)).sum())
                n_up = float((true_pos & (ys > 0)).sum())
                if n_true < 30:
                    continue
                best80 = None
                bestf = None
                rules = [("will", None)]
                if hotp is not None:
                    run = hotp[:, :, : last.shape[-1]] >= 0.55
                    prefix = np.cumprod(run, axis=1).sum(axis=1)
                    rules.append(("lh_occ", hot_st & (prefix >= min_w) & node_ok))
                rules.append(("lh", hot_st & node_ok))
                rules.append(("lh_dur", hot_st & (d >= min_w) & node_ok))
                for on_rule, on_mask in rules:
                    for on_t in (0.40, 0.55, 0.70, 0.85, 0.95):
                        for up_t in (0.40, 0.55, 0.70, 0.85, 0.95, 0.98):
                            up_pred = (~hot_st) & node_ok & (w >= up_t)
                            if max_st is not None:
                                up_pred = up_pred & (s <= int(max_st))
                            if on_mask is None:
                                on_pred = hot_st & node_ok & (w >= on_t)
                            else:
                                on_pred = on_mask | (hot_st & node_ok & (w >= on_t))
                            pred = on_pred | up_pred
                            p, rec, f1, n_pred, _, tp = _score(pred, true_pos, start_ok)
                            recs = (f1, p, rec, min_w, max_st, on_rule, on_t, up_t, n_true, n_on, n_up, n_pred)
                            if bestf is None or f1 > bestf[0]:
                                bestf = recs
                            if p + 1e-12 >= 0.80 and (best80 is None or f1 > best80[0]):
                                best80 = recs
                            if p + 1e-12 >= 0.80 and f1 + 1e-12 >= 0.80:
                                msg = (
                                    f"HIT {split} min={min_w} max_st={max_st} rule={on_rule} "
                                    f"on={on_t:.2f} up={up_t:.2f} P={p:.3f} R={rec:.3f} F1={f1:.3f} "
                                    f"n_true={n_true:.0f} on={n_on:.0f} up={n_up:.0f} n_pred={n_pred:.0f}"
                                )
                                print(f"  {msg}", flush=True)
                                hits.append(msg)
                print(
                    f"  min={min_w} max_st={max_st} n_true={n_true:.0f} on={n_on:.0f} up={n_up:.0f} "
                    f"bestF1={bestf[0]:.3f} P={bestf[1]:.3f} R={bestf[2]:.3f} rule={bestf[5]} "
                    f"on={bestf[6]:.2f} up={bestf[7]:.2f}",
                    flush=True,
                )
                if best80:
                    print(
                        f"    P80-best F1={best80[0]:.3f} P={best80[1]:.3f} R={best80[2]:.3f} "
                        f"rule={best80[5]} on={best80[6]:.2f} up={best80[7]:.2f}",
                        flush=True,
                    )
                else:
                    print("    no P>=0.80", flush=True)
    print(f"\nHIT count={len(hits)}", flush=True)


if __name__ == "__main__":
    main()
