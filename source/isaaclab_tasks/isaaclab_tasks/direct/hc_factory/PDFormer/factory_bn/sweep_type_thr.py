"""Grid-search per-type report bars for max F1 at P>=80."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from factory_bn.dataset import build_dataloaders, make_pattern_keys
from factory_bn.model import BNPDFormer
from factory_bn.remain import _prf, node_event_targets
from factory_bn.train import _load_init_ckpt, _move_batch, _near_remain_mask


def _collect(model, loader, device):
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
    return (
        torch.cat(ys).numpy(),
        torch.cat(ws).numpy(),
        torch.cat(ss).numpy(),
        torch.cat(ds).numpy(),
        torch.cat(rs).numpy(),
        os_[0].numpy() if os_[0].dim() == 1 else torch.cat(os_).numpy(),
        hs[0].numpy() if hs[0].dim() == 1 else torch.cat(hs).numpy(),
    )


def _type_id(model, n_nodes: int) -> np.ndarray:
    tid = np.full((n_nodes,), 4, dtype=np.int16)
    for i, name in enumerate(("machine", "gantry", "agv", "workbench")):
        buf = model._occ_type_masks().get(name)
        if buf is None:
            continue
        node = buf.detach().cpu().numpy().reshape(-1)[:n_nodes] > 0.5
        tid[node] = i
    return tid


def _pack(y, w, s, r, o, h, min_windows: int = 8, start_tol: int = 3, max_start_windows=None):
    y_will, y_start, _ = node_event_targets(
        y,
        min_windows=min_windows,
        remain_mask=r,
        occ_node_mask=o,
        max_start_windows=max_start_windows,
    )
    pred_start = np.asarray(s, dtype=np.int64)
    last = np.asarray(h, dtype=np.float32)
    if last.ndim == 1:
        last = np.broadcast_to(last.reshape(1, -1), y_will.shape)
    last = last[:, : y_will.shape[-1]]
    pred_start = np.where(last > 0.5, 0, pred_start[:, : y_will.shape[-1]])
    if o.ndim == 1:
        node_ok = np.broadcast_to(o.reshape(1, -1) > 0.5, y_will.shape)
    else:
        node_ok = o[:, : y_will.shape[-1]] > 0.5
    true_pos = (y_will > 0.5) & node_ok
    report_ok = np.abs(pred_start.astype(np.int64) - y_start.astype(np.int64)) <= int(start_tol)
    upcoming = true_pos & (last <= 0.5)
    ongoing = true_pos & (last > 0.5)
    return {
        "will": np.asarray(w, dtype=np.float32)[:, : y_will.shape[-1]],
        "hot": last > 0.5,
        "node_ok": node_ok,
        "true_pos": true_pos,
        "report_ok": report_ok,
        "upcoming": upcoming,
        "ongoing": ongoing,
        "n_true": float(true_pos.sum()),
        "n_up": float(upcoming.sum()),
        "n_on": float(ongoing.sum()),
    }


def _score(pack, pred: np.ndarray) -> dict[str, float]:
    pred = pred & pack["node_ok"]
    hit = pred & pack["true_pos"] & pack["report_ok"]
    n_pred = float(pred.sum())
    tp = float(hit.sum())
    p, rec, f1 = _prf(tp, n_pred - tp, pack["n_true"] - tp)
    up = float((hit & pack["upcoming"]).sum()) / pack["n_up"] if pack["n_up"] else 0.0
    on = float((hit & pack["ongoing"]).sum()) / pack["n_on"] if pack["n_on"] else 0.0
    return {"P": p, "R": rec, "F1": f1, "up": up, "on": on}


def _pred(pack, type_id: np.ndarray, gthr: float, bars: tuple[float, float, float, float]):
    node_thr = np.array([bars[0], bars[1], bars[2], bars[3], gthr], dtype=np.float32)
    thr = node_thr[type_id[: pack["will"].shape[-1]]]
    cold_ok = pack["will"] >= thr.reshape(1, -1)
    hot_ok = pack["will"] >= float(gthr)
    return np.where(pack["hot"], hot_ok, cold_ok)


def _search(pack, type_id, grid, globals_):
    keys = list(grid)
    best80 = None
    bestf = None
    n = 0
    for gthr, combo in itertools.product(globals_, itertools.product(*[grid[k] for k in keys])):
        pred = _pred(pack, type_id, float(gthr), combo)
        ev = _score(pack, pred)
        n += 1
        rec = (ev["F1"], ev["P"], ev["R"], ev["up"], ev["on"], float(gthr), dict(zip(keys, combo)))
        if bestf is None or ev["F1"] > bestf[0] + 1e-9:
            bestf = rec
        if ev["P"] + 1e-12 >= 0.80 and (best80 is None or ev["F1"] > best80[0] + 1e-9):
            best80 = rec
    return n, bestf, best80


def _fmt(tag: str, rec) -> str:
    return (
        f"{tag}: F1={rec[0]:.3f} P={rec[1]:.3f} R={rec[2]:.3f} "
        f"up={rec[3]:.3f} on={rec[4]:.3f} thr={rec[5]:.2f} bars={rec[6]}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="factory_bn/configs/FactoryBN_dense_f1_p80.json")
    ap.add_argument("--ckpt", required=True)
    args = ap.parse_args()
    root = Path(__file__).resolve().parent.parent
    cfg = json.loads((root / args.config).read_text(encoding="utf-8"))
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
    _load_init_ckpt(model, (root / args.ckpt).resolve(), device)
    model.force_ongoing_will = False
    model.event_union_occupancy = False
    model.event_report_threshold_by_type = {}
    min_w = int(cfg.get("event_min_windows", cfg.get("hot_min_windows", 8)))
    start_tol = int(cfg.get("start_tol_windows", 3))
    raw_max_start = cfg.get("event_max_start_windows")
    max_start = None if raw_max_start in (None, "", False) else int(raw_max_start)

    cache = {}
    for split, loader in (("val", val_loader), ("test", test_loader)):
        cache[split] = _collect(model, loader, device)
        print(f"collected {split}", flush=True)

    type_id = _type_id(model, cache["val"][1].shape[-1])
    packs = {
        name: _pack(
            y, w, s, r, o, h,
            min_windows=min_w,
            start_tol=start_tol,
            max_start_windows=max_start,
        )
        for name, (y, w, s, d, r, o, h) in cache.items()
    }
    grid = {
        "machine": [0.70, 0.75, 0.80, 0.85, 0.88, 0.90, 0.92],
        "gantry": [0.82, 0.85, 0.88, 0.90, 0.92, 0.94, 0.96],
        "agv": [0.85, 0.88, 0.90, 0.92, 0.94, 0.96, 0.98],
        "workbench": [0.70, 0.75, 0.78, 0.80, 0.85, 0.88, 0.90],
    }
    globals_ = [0.70, 0.75, 0.80, 0.85, 0.88, 0.90, 0.92, 0.94]
    n, bestf, best80 = _search(packs["val"], type_id, grid, globals_)
    print(f"val combinations={n}", flush=True)
    print(_fmt("val maxF1", bestf), flush=True)
    if best80 is None:
        print("val no P>=0.80", flush=True)
        return
    print(_fmt("val P>=0.80", best80), flush=True)
    gthr, bars = best80[5], best80[6]
    pred = _pred(
        packs["test"],
        type_id,
        gthr,
        (bars["machine"], bars["gantry"], bars["agv"], bars["workbench"]),
    )
    te = _score(packs["test"], pred)
    print(
        f"test @val-P80-bars: P={te['P']:.3f} R={te['R']:.3f} F1={te['F1']:.3f} "
        f"up={te['up']:.3f} on={te['on']:.3f}",
        flush=True,
    )
    _, tf, t80 = _search(packs["test"], type_id, grid, globals_)
    print(_fmt("test maxF1", tf), flush=True)
    if t80:
        print(_fmt("test P>=0.80", t80), flush=True)
    else:
        print("test no P>=0.80", flush=True)


if __name__ == "__main__":
    main()
