"""Decode / last_hot ceiling on the current unsupervised near5 ckpt."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from factory_bn.dataset import build_dataloaders, make_pattern_keys
from factory_bn.model import BNPDFormer
from factory_bn.remain import node_event_targets, station_report_metrics
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
            hots.append(pred["hot_prob"].cpu().numpy())
    y = np.concatenate(ys, 0)
    w = np.concatenate(ws, 0)
    s = np.concatenate(ss, 0)
    d = np.concatenate(ds, 0)
    r = np.concatenate(rs, 0)
    o0 = os_[0]
    o = o0 if np.asarray(o0).ndim == 1 else np.concatenate(os_, 0)
    h = hs[0] if np.asarray(hs[0]).ndim == 1 else np.concatenate(hs, 0)
    hot = np.concatenate(hots, 0)
    return y, w, s, d, r, o, h, hot


def _brief(name: str, m: dict[str, float]) -> None:
    print(
        f"  {name:<28} P={m.get('report_precision', 0):.3f} "
        f"R={m.get('report_recall', 0):.3f} F1={m.get('report_f1', 0):.3f} "
        f"up={m.get('report_recall_upcoming', 0):.3f} "
        f"on={m.get('report_recall_ongoing', 0):.3f} "
        f"n_pred={int(m.get('n_pred_who', 0))} n_true={int(m.get('n_true_who', 0))}",
        flush=True,
    )


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
        hot_min_windows=int(cfg.get("hot_min_windows", 8)),
        train_mode="unsupervised",
    )
    df["pattern_keys"] = make_pattern_keys(
        df.pop("train_feature_windows"), s_attn_size=3, n_cluster=16, output_channel=4
    )
    model = BNPDFormer(cfg, df).to(device)
    ckpt = root / "libcity/cache/model_cache/dense_i1_a1_near5/BNPDFormer_best.pt"
    _load_init_ckpt(model, ckpt, device)
    model.force_ongoing_will = False
    model.event_union_occupancy = False
    model.event_union_upcoming = False
    model.event_report_threshold_by_type = {}
    model.recall_lift_threshold = 0.0

    ev_kw = dict(
        min_windows=5,
        start_tol_windows=3,
        max_start_windows=2,
        report_ongoing_only=False,
    )
    for split, loader in (("val", val_loader), ("test", test_loader)):
        y, w, s, d, r, o, h, hot = _collect(model, loader, device)
        y_will, y_start, _ = node_event_targets(
            y, min_windows=5, remain_mask=r, occ_node_mask=o, max_start_windows=2
        )
        last = h
        if last.ndim == 1:
            last = np.broadcast_to(last.reshape(1, -1), y_will.shape)
        last = last[:, : y_will.shape[-1]]
        if o.ndim == 1:
            node_ok = np.broadcast_to(o.reshape(1, -1) > 0.5, y_will.shape)
        else:
            node_ok = o[:, : y_will.shape[-1]] > 0.5
        true_pos = (y_will > 0.5) & node_ok
        ongoing = true_pos & (y_start == 0)
        upcoming = true_pos & (y_start > 0)
        lh = (last > 0.5) & node_ok
        print(
            f"\n== {split} n_true={int(true_pos.sum())} "
            f"on={int(ongoing.sum())} up={int(upcoming.sum())} "
            f"last_hot={int(lh.sum())} "
            f"lh&on={int((lh & ongoing).sum())} "
            f"lh&~true={int((lh & ~true_pos).sum())} "
            f"on&~lh={int((ongoing & ~lh).sum())}",
            flush=True,
        )
        prefix = np.cumprod(hot[:, :, : y_will.shape[-1]] >= 0.55, axis=1).sum(axis=1)
        for name, pred_w, force, floor, fto, req_dur in (
            ("last_hot only", np.where(lh, 1.0, 0.0), True, 0.0, 1.0, False),
            ("lh & dur>=5", np.where(lh & (d[:, : y_will.shape[-1]] >= 5), 1.0, 0.0), True, 0.0, 1.0, False),
            ("lh & prefix>=5", np.where(lh & (prefix >= 5), 1.0, 0.0), True, 0.0, 1.0, False),
            ("raw@0.90", w, False, 0.70, 0.80, False),
            ("force floor0.70 lift1", w, True, 0.70, 1.0, False),
            ("force floor0.50 lift1", w, True, 0.50, 1.0, False),
            ("force floor0.00 lift1", w, True, 0.00, 1.0, False),
        ):
            m = station_report_metrics(
                y, pred_w, s, d, r, o, threshold=0.80, hist_last_hot=last,
                will_floor=floor, force_ongoing_will=force, force_to=fto,
                force_require_dur=req_dur, **ev_kw,
            )
            _brief(name, m)

        print("  -- sweep force floor=0.50 lift=τ --", flush=True)
        best = None
        for thr in (0.55, 0.65, 0.70, 0.75, 0.80, 0.82, 0.85, 0.88, 0.90, 0.94, 0.98):
            wp = w.copy()
            wp = np.where(lh & (w >= 0.50), np.maximum(wp, thr), wp)
            occ_up = (~lh) & (prefix >= 5) & node_ok
            wp = np.where(occ_up, np.maximum(wp, 0.80), wp)
            m = station_report_metrics(
                y, wp, s, d, r, o, threshold=thr, hist_last_hot=last,
                will_floor=0.50, force_ongoing_will=True, force_to=thr,
                force_require_dur=False, **ev_kw,
            )
            rec = (
                float(m["report_f1"]),
                float(m["report_precision"]),
                float(m["report_recall"]),
                thr,
                float(m.get("report_recall_upcoming", 0)),
                float(m.get("report_recall_ongoing", 0)),
            )
            if best is None or (
                (rec[1] >= 0.80 and (best[1] < 0.80 or rec[0] > best[0]))
                or (rec[1] < 0.80 and best[1] < 0.80 and rec[0] > best[0])
            ):
                best = rec
            print(
                f"    τ={thr:.2f} P={m['report_precision']:.3f} R={m['report_recall']:.3f} "
                f"F1={m['report_f1']:.3f} up={m.get('report_recall_upcoming', 0):.3f} "
                f"on={m.get('report_recall_ongoing', 0):.3f}",
                flush=True,
            )
        print(
            f"  best P>=0.80-prefer F1={best[0]:.3f} P={best[1]:.3f} R={best[2]:.3f} τ={best[3]:.2f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
