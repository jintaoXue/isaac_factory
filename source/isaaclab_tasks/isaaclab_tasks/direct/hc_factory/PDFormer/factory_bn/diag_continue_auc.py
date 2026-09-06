"""Can continue scores rank last_hot flicker vs true ongoing events?"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from factory_bn.dataset import build_dataloaders, make_pattern_keys
from factory_bn.model import BNPDFormer
from factory_bn.remain import _prf, node_event_targets
from factory_bn.train import _load_init_ckpt, _move_batch, _near_remain_mask


def _auroc(scores: np.ndarray, y: np.ndarray) -> float:
    y = y.astype(bool)
    if y.sum() == 0 or (~y).sum() == 0:
        return float("nan")
    order = np.argsort(-scores)
    y = y[order]
    tp = np.cumsum(y)
    fp = np.cumsum(~y)
    tpr = tp / tp[-1]
    fpr = fp / fp[-1]
    return float(np.trapz(tpr, fpr))


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
        hot_min_windows=8,
        train_mode="unsupervised",
    )
    df["pattern_keys"] = make_pattern_keys(
        df.pop("train_feature_windows"), s_attn_size=3, n_cluster=16, output_channel=4
    )
    model = BNPDFormer(cfg, df).to(device)
    _load_init_ckpt(model, root / "libcity/cache/model_cache/dense_i1_a1_f180/BNPDFormer_best.pt", device)
    model.force_ongoing_will = False
    model.event_union_occupancy = False

    for split, loader in (("val", val_loader), ("test", test_loader)):
        conts, onsets, durs, hots, lasts, ys, rs, os_ = [], [], [], [], [], [], [], []
        model.eval()
        with torch.no_grad():
            for batch in loader:
                batch = _move_batch(batch, device)
                out = model.forward(batch)
                pred = model.predict(batch)
                near = _near_remain_mask(batch.get("remain_mask"), 15)
                ys.append(batch["y_hot"].cpu().numpy())
                rs.append(near.cpu().numpy())
                occ = batch.get("occ_node_mask")
                os_.append(occ.cpu().numpy() if occ is not None else np.ones(ys[-1].shape[-1]))
                lasts.append(batch["hist_last_hot"].cpu().numpy())
                conts.append(torch.sigmoid(out["event_will_continue_logit"]).cpu().numpy())
                onsets.append(torch.sigmoid(out["event_will_onset_logit"]).cpu().numpy())
                durs.append(out["event_dur"].cpu().numpy())
                hots.append(pred["hot_prob"].cpu().numpy())
        y = np.concatenate(ys, 0)
        r = np.concatenate(rs, 0)
        o0 = os_[0]
        o = o0 if np.asarray(o0).ndim == 1 else np.concatenate(os_, 0)
        last = lasts[0] if np.asarray(lasts[0]).ndim == 1 else np.concatenate(lasts, 0)
        cont = np.concatenate(conts, 0)
        onset = np.concatenate(onsets, 0)
        dur = np.concatenate(durs, 0)
        hot = np.concatenate(hots, 0)
        y_will, y_start, y_dur = node_event_targets(
            y, min_windows=8, remain_mask=r, occ_node_mask=o, max_start_windows=2
        )
        if o.ndim == 1:
            node_ok = np.broadcast_to(o.reshape(1, -1) > 0.5, y_will.shape)
        else:
            node_ok = o[:, : y_will.shape[-1]] > 0.5
        if last.ndim == 1:
            last = np.broadcast_to(last.reshape(1, -1), y_will.shape)
        last = last[:, : y_will.shape[-1]]
        true = (y_will > 0.5) & node_ok
        ongoing = true & (y_start == 0)
        upcoming = true & (y_start > 0)
        lh = (last > 0.5) & node_ok
        prefix = np.cumprod(hot[:, :, : y_will.shape[-1]] >= 0.55, axis=1).sum(axis=1)
        print(
            f"\n== {split} true={int(true.sum())} on={int(ongoing.sum())} up={int(upcoming.sum())} "
            f"lh={int(lh.sum())} lh&on={int((lh & ongoing).sum())} lh&~true={int((lh & ~true).sum())}",
            flush=True,
        )
        lh_y = (lh & ongoing).reshape(-1)[lh.reshape(-1)]
        print(
            f"  AUROC last_hot→ongoing  cont={_auroc(cont[lh], lh_y):.3f} "
            f"onset={_auroc(onset[lh], lh_y):.3f} "
            f"dur={_auroc(dur[lh], lh_y):.3f} "
            f"prefix={_auroc(prefix[lh], lh_y):.3f}",
            flush=True,
        )
        best = None
        for cthr in np.linspace(0.05, 0.95, 19):
            on_pred = lh & (cont >= cthr)
            for uthr in (0.70, 0.80, 0.85, 0.90, 0.94, 0.98):
                up_pred = (~lh) & node_ok & (onset >= uthr)
                pred = (on_pred | up_pred) & node_ok
                start_ok = np.ones_like(true)
                start_ok[upcoming] = True
                hit = pred & true
                p, rec, f1 = _prf(float(hit.sum()), float(pred.sum() - hit.sum()), float(true.sum() - hit.sum()))
                on_r = float((hit & ongoing).sum()) / max(float(ongoing.sum()), 1.0)
                up_r = float((hit & upcoming).sum()) / max(float(upcoming.sum()), 1.0)
                row = (f1, p, rec, on_r, up_r, cthr, uthr, float(pred.sum()))
                if best is None or (
                    (p + 1e-12 >= 0.80 and (best[1] < 0.80 or f1 > best[0]))
                    or (p < 0.80 and best[1] < 0.80 and f1 > best[0])
                ):
                    best = row
        print(
            f"  best P>=0.80-prefer F1={best[0]:.3f} P={best[1]:.3f} R={best[2]:.3f} "
            f"on={best[3]:.3f} up={best[4]:.3f} cthr={best[5]:.2f} uthr={best[6]:.2f} n_pred={best[7]:.0f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
