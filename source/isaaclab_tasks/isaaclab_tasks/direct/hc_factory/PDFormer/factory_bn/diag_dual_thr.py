"""Search separate ongoing/upcoming bars, including very low upcoming thresholds."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from factory_bn.dataset import build_dataloaders, make_pattern_keys
from factory_bn.model import BNPDFormer, OCC_TYPE_NAMES
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


def _pack(y, w, s, r, o, h):
    y_will, y_start, _ = node_event_targets(y, min_windows=8, remain_mask=r, occ_node_mask=o)
    last = np.asarray(h, dtype=np.float32)
    if last.ndim == 1:
        last = np.broadcast_to(last.reshape(1, -1), y_will.shape)
    last = last[:, : y_will.shape[-1]]
    w = np.asarray(w)[:, : y_will.shape[-1]]
    s = np.where(last > 0.5, 0, np.asarray(s)[:, : y_will.shape[-1]])
    if o.ndim == 1:
        node_ok = np.broadcast_to(o.reshape(1, -1) > 0.5, y_will.shape)
    else:
        node_ok = o[:, : y_will.shape[-1]] > 0.5
    true_pos = (y_will > 0.5) & node_ok
    report_ok = np.abs(s.astype(np.int64) - y_start.astype(np.int64)) <= 3
    hot = last > 0.5
    return {
        "w": w,
        "node_ok": node_ok,
        "true_pos": true_pos,
        "report_ok": report_ok,
        "hot": hot,
        "upcoming": true_pos & ~hot,
        "ongoing": true_pos & hot,
        "n_true": float(true_pos.sum()),
        "n_up": float((true_pos & ~hot).sum()),
        "n_on": float((true_pos & hot).sum()),
    }


def _eval(pack, pred):
    pred = pred & pack["node_ok"]
    hit = pred & pack["true_pos"] & pack["report_ok"]
    n_pred = float(pred.sum())
    tp = float(hit.sum())
    p, rec, f1 = _prf(tp, n_pred - tp, pack["n_true"] - tp)
    up = float((hit & pack["upcoming"]).sum()) / pack["n_up"] if pack["n_up"] else 0.0
    on = float((hit & pack["ongoing"]).sum()) / pack["n_on"] if pack["n_on"] else 0.0
    return p, rec, f1, up, on, n_pred, tp


def _type_id(model, n_nodes: int) -> np.ndarray:
    tid = np.full((n_nodes,), 4, dtype=np.int16)
    for i, name in enumerate(OCC_TYPE_NAMES):
        buf = model._occ_type_masks().get(name)
        if buf is None:
            continue
        node = buf.detach().cpu().numpy().reshape(-1)[:n_nodes] > 0.5
        tid[node] = i
    return tid


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
    _load_init_ckpt(model, root / "libcity/cache/model_cache/n10_plus20_p80_continue/BNPDFormer_best.pt", device)
    model.force_ongoing_will = False
    model.event_union_occupancy = False
    model.event_report_threshold_by_type = {}
    model.recall_lift_threshold = 0.0

    caches = {}
    for name, loader in (("val", val_loader), ("test", test_loader)):
        y, w, s, r, o, h = _collect(model, loader, device)
        caches[name] = _pack(y, w, s, r, o, h)
        print(f"collected {name}", flush=True)

    tid = _type_id(model, caches["val"]["w"].shape[-1])
    names = list(OCC_TYPE_NAMES) + ["other"]

    pack = caches["val"]
    cold_neg = pack["node_ok"] & ~pack["hot"] & ~pack["true_pos"]
    print("\n== val cold_neg will mass", flush=True)
    for thr in (1e-4, 1e-3, 0.01, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.90):
        n = float((cold_neg & (pack["w"] >= thr)).sum())
        tp_u = float((pack["upcoming"] & (pack["w"] >= thr)).sum())
        print(f"  thr={thr:g} cold_fp={n:.0f} up_tp={tp_u:.0f}/{pack['n_up']:.0f}", flush=True)

    print("\n== val per-type upcoming will>=0.05", flush=True)
    for i, name in enumerate(names):
        col = tid[: pack["w"].shape[-1]] == i
        up = pack["upcoming"] & col
        fp = cold_neg & col & (pack["w"] >= 0.05)
        tp = up & (pack["w"] >= 0.05)
        print(
            f"  {name:10s} up={float(up.sum()):.0f} tp@0.05={float(tp.sum()):.0f} "
            f"fp@0.05={float(fp.sum()):.0f}",
            flush=True,
        )

    on_grid = [0.20, 0.30, 0.40, 0.50, 0.55, 0.60, 0.70, 0.80, 0.90]
    up_grid = [0.001, 0.01, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.70, 0.90]
    type_up = {
        "machine": [0.001, 0.05, 0.20, 0.50, 0.70],
        "gantry": [0.20, 0.50, 0.70, 0.85, 0.90],
        "agv": [0.20, 0.50, 0.70, 0.85, 0.90],
        "workbench": [0.20, 0.50, 0.70, 0.80, 0.90],
    }

    print("\n== val dual-threshold (global)", flush=True)
    best80 = None
    hit = []
    for ot in on_grid:
        on_pred = pack["hot"] & pack["node_ok"] & (pack["w"] >= ot)
        for ut in up_grid:
            up_pred = (~pack["hot"]) & pack["node_ok"] & (pack["w"] >= ut)
            p, rec, f1, up, on, n_pred, tp = _eval(pack, on_pred | up_pred)
            recs = (f1, p, rec, up, on, ot, ut, n_pred)
            if p + 1e-12 >= 0.80 and f1 + 1e-12 >= 0.80:
                hit.append(recs)
            if p + 1e-12 >= 0.80 and (best80 is None or f1 > best80[0]):
                best80 = recs
    if best80:
        print(
            f"  best P>=80: F1={best80[0]:.3f} P={best80[1]:.3f} R={best80[2]:.3f} "
            f"up={best80[3]:.3f} on={best80[4]:.3f} on_thr={best80[5]:.2f} up_thr={best80[6]:.3f}",
            flush=True,
        )
    else:
        print("  no global dual-thr P>=80", flush=True)
    print(f"  P80 and F180 hits={len(hit)}", flush=True)
    for recs in sorted(hit, reverse=True)[:8]:
        print(
            f"    F1={recs[0]:.3f} P={recs[1]:.3f} R={recs[2]:.3f} "
            f"up={recs[3]:.3f} on={recs[4]:.3f} on={recs[5]:.2f} up={recs[6]:.3f}",
            flush=True,
        )

    print("\n== val per-type upcoming bars (ongoing will>=0.40)", flush=True)
    keys = list(type_up)
    import itertools

    best_type = None
    type_hits = 0
    on_pred = pack["hot"] & pack["node_ok"] & (pack["w"] >= 0.40)
    n = 0
    for combo in itertools.product(*[type_up[k] for k in keys]):
        n += 1
        bars = dict(zip(keys, combo))
        up_pred = np.zeros_like(pack["hot"], dtype=bool)
        for i, name in enumerate(keys):
            col = tid[: pack["w"].shape[-1]] == i
            up_pred |= (~pack["hot"]) & pack["node_ok"] & col & (pack["w"] >= bars[name])
        p, rec, f1, up, on, n_pred, tp = _eval(pack, on_pred | up_pred)
        recs = (f1, p, rec, up, on, bars, n_pred)
        if p + 1e-12 >= 0.80 and f1 + 1e-12 >= 0.80:
            type_hits += 1
            if best_type is None or f1 > best_type[0]:
                best_type = recs
        elif p + 1e-12 >= 0.80 and (best_type is None or f1 > best_type[0]):
            best_type = recs
    print(f"  combos={n} P80+F180 hits={type_hits}", flush=True)
    if best_type:
        print(
            f"  best: F1={best_type[0]:.3f} P={best_type[1]:.3f} R={best_type[2]:.3f} "
            f"up={best_type[3]:.3f} on={best_type[4]:.3f} bars={best_type[5]}",
            flush=True,
        )
        bars = best_type[5]
        te = caches["test"]
        on_te = te["hot"] & te["node_ok"] & (te["w"] >= 0.40)
        up_te = np.zeros_like(te["hot"], dtype=bool)
        for i, name in enumerate(keys):
            col = tid[: te["w"].shape[-1]] == i
            up_te |= (~te["hot"]) & te["node_ok"] & col & (te["w"] >= bars[name])
        p, rec, f1, up, on, n_pred, tp = _eval(te, on_te | up_te)
        print(
            f"  test @val-bars: P={p:.3f} R={rec:.3f} F1={f1:.3f} up={up:.3f} on={on:.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
