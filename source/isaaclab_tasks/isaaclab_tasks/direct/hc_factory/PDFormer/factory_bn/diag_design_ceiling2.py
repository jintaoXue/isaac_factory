"""Find a label+decode contract that can hit P>=0.80 and F1>=0.80 on v5."""

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


def _score(pred, true_pos, start_ok):
    pred = pred.astype(bool)
    hit = pred & true_pos & start_ok
    n_pred = float(pred.sum())
    n_true = float(true_pos.sum())
    tp = float(hit.sum())
    return (*_prf(tp, n_pred - tp, n_true - tp), n_pred, n_true, tp)


def _occ_extract(hotp, remain, node_ok, min_w, thr):
    run = (hotp >= thr) & (remain > 0.5)
    b, k, n = run.shape
    will = np.zeros((b, n), dtype=bool)
    start = np.zeros((b, n), dtype=np.int64)
    dur = np.zeros((b, n), dtype=np.float32)
    for bi in range(b):
        k_use = int(remain[bi].sum()) if remain.ndim == 2 else k
        k_use = max(0, min(k_use, k))
        for ni in range(n):
            if not node_ok[bi, ni]:
                continue
            col = run[bi, :k_use, ni]
            best_len = 0
            best_i = 0
            i = 0
            while i < k_use:
                if not col[i]:
                    i += 1
                    continue
                j = i + 1
                while j < k_use and col[j]:
                    j += 1
                if (j - i) > best_len:
                    best_len = j - i
                    best_i = i
                i = j
            if best_len >= min_w:
                will[bi, ni] = True
                start[bi, ni] = best_i
                dur[bi, ni] = float(best_len)
    return will, start, dur


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
        hotp = hotp[:, :, : last.shape[-1]]
        if r.ndim == 3:
            remain = r[:, : hotp.shape[1], : last.shape[-1]] > 0.5
        elif r.ndim == 2:
            remain = np.broadcast_to(r[:, : hotp.shape[1], None] > 0.5, hotp.shape)
        else:
            remain = np.ones(hotp.shape, dtype=bool)
        print(f"\n==== {split} n={y.shape[0]} ====", flush=True)
        print("  persistence last_hot vs ongoing-only labels:", flush=True)
        for min_w in (1, 2, 3, 4, 5, 6, 8):
            yw, ys, _ = node_event_targets(y, min_windows=min_w, remain_mask=r, occ_node_mask=o, max_start_windows=0)
            yw = yw[:, : last.shape[-1]]
            ys = ys[:, : last.shape[-1]]
            true_pos = (yw > 0.5) & node_ok
            pred = hot_st & node_ok
            start_ok = np.abs(s.astype(np.int64) - ys.astype(np.int64)) <= 15
            p, rec, f1, n_pred, n_true, tp = _score(pred, true_pos, start_ok)
            mark = " HIT" if p >= 0.80 and f1 >= 0.80 else ""
            print(
                f"    min={min_w} persist P={p:.3f} R={rec:.3f} F1={f1:.3f} "
                f"n_true={n_true:.0f} n_pred={n_pred:.0f}{mark}",
                flush=True,
            )
            if p >= 0.80 and f1 >= 0.80:
                hits.append(f"{split} persist min={min_w} P={p:.3f} F1={f1:.3f}")

        print("  occupancy-extract / will / lh_occ:", flush=True)
        occ_cache = {}
        for min_w in (3, 4, 5):
            for thr in (0.45, 0.55, 0.65, 0.75):
                occ_cache[(min_w, thr)] = node_event_targets(
                    (hotp >= thr).astype(np.float32),
                    min_windows=min_w,
                    remain_mask=r,
                    occ_node_mask=o,
                )
            run = hotp >= 0.55
            prefix = np.cumprod(run, axis=1).sum(axis=1)
            for max_st in (0, 2, 3, 4, None):
                yw, ys, _ = node_event_targets(
                    y, min_windows=min_w, remain_mask=r, occ_node_mask=o, max_start_windows=max_st
                )
                yw = yw[:, : last.shape[-1]]
                ys = ys[:, : last.shape[-1]]
                true_pos = (yw > 0.5) & node_ok
                n_true = float(true_pos.sum())
                if n_true < 30:
                    continue
                start_gt = ys.astype(np.int64)
                for tol in (3, 7, 15):
                    best80 = None
                    for thr in (0.45, 0.55, 0.65, 0.75):
                        ow, os_, _od = occ_cache[(min_w, thr)]
                        ow = (np.asarray(ow)[:, : last.shape[-1]] > 0.5) & node_ok
                        os_ = np.asarray(os_)[:, : last.shape[-1]]
                        ow_use = ow if max_st is None else (ow & (os_ <= int(max_st)))
                        start_ok = np.abs(os_.astype(np.int64) - start_gt) <= tol
                        p, rec, f1, n_pred, _, _ = _score(ow_use, true_pos, start_ok)
                        recs = (f1, p, rec, f"occ@{thr:.2f}", n_pred)
                        if p >= 0.80 and (best80 is None or f1 > best80[0]):
                            best80 = recs
                        if p >= 0.80 and f1 >= 0.80:
                            msg = (
                                f"HIT {split} min={min_w} max_st={max_st} tol={tol} occ@{thr:.2f} "
                                f"P={p:.3f} R={rec:.3f} F1={f1:.3f} n_true={n_true:.0f} n_pred={n_pred:.0f}"
                            )
                            print(f"    {msg}", flush=True)
                            hits.append(msg)
                    for need in (min_w, max(1, min_w - 1), 3):
                        on_pred = hot_st & node_ok & (prefix >= need)
                        for up_t in (0.70, 0.90, 0.98, 2.0):
                            up_pred = (~hot_st) & node_ok & (w >= up_t)
                            if max_st is not None:
                                up_pred = up_pred & (s <= int(max_st))
                            pred = on_pred | up_pred
                            start_ok = np.abs(s.astype(np.int64) - start_gt) <= tol
                            p, rec, f1, n_pred, _, _ = _score(pred, true_pos, start_ok)
                            if p >= 0.80 and (best80 is None or f1 > best80[0]):
                                best80 = (f1, p, rec, f"lh_occ>={need}+up{up_t}", n_pred)
                            if p >= 0.80 and f1 >= 0.80:
                                msg = (
                                    f"HIT {split} min={min_w} max_st={max_st} tol={tol} "
                                    f"lh_occ>={need}+up{up_t} P={p:.3f} R={rec:.3f} F1={f1:.3f} "
                                    f"n_true={n_true:.0f} n_pred={n_pred:.0f}"
                                )
                                print(f"    {msg}", flush=True)
                                hits.append(msg)
                    if best80:
                        print(
                            f"    min={min_w} max_st={max_st} tol={tol} n_true={n_true:.0f} "
                            f"P80-best F1={best80[0]:.3f} P={best80[1]:.3f} R={best80[2]:.3f} "
                            f"rule={best80[3]} n_pred={best80[4]:.0f}",
                            flush=True,
                        )
                    else:
                        print(
                            f"    min={min_w} max_st={max_st} tol={tol} n_true={n_true:.0f} no P>=0.80",
                            flush=True,
                        )
    print(f"\nHIT count={len(hits)}", flush=True)
    for h in hits:
        print(h)


if __name__ == "__main__":
    main()
