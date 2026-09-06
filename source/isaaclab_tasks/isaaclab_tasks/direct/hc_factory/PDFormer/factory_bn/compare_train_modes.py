"""Compare unsupervised ops occupancy vs supervised bottleneck_score occupancy.

Phase ``labels`` (CPU): count cell / event overlap on the official episode split.
Phase ``eval`` (GPU): score one ckpt on both label sets; sweep τ on val, freeze on test.

Example::

    python -m factory_bn.compare_train_modes --phase labels --data_dir raw_data/dense_i1
    python -m factory_bn.compare_train_modes --phase eval --ckpt <pt> --data_dir raw_data/dense_i1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_PDFORMER_ROOT = Path(__file__).resolve().parent.parent
if str(_PDFORMER_ROOT) not in sys.path:
    sys.path.insert(0, str(_PDFORMER_ROOT))

from factory_bn.dataset import (  # noqa: E402
    FactoryBNWindowDataset,
    Scaler,
    _build_samples,
    load_factory_bn_bundle,
    split_episodes_by_name,
)
from factory_bn.remain import (  # noqa: E402
    first_done_index,
    node_event_targets,
    node_hot_mask,
    occupancy_node_mask,
    ops_hot_mask,
    pack_remain_target,
    parse_max_start_windows,
)


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = (_PDFORMER_ROOT / p).resolve()
    return p


def _split_sets(bundle: dict[str, Any], cfg: dict[str, Any]) -> dict[str, set[str]]:
    names = [str(ep.get("name") or ep["episode_id"]) for ep in bundle["episodes"]]
    train, val, test = split_episodes_by_name(
        names,
        train_ratio=float(cfg.get("train_rate", 0.7)),
        val_ratio=float(cfg.get("eval_rate", 0.15)),
        seed=int(cfg.get("seed", 42)),
    )
    return {"train": train, "val": val, "test": test}


def _event_counts(
    will: np.ndarray,
    start: np.ndarray,
    occ: np.ndarray,
) -> dict[str, int]:
    pos = (will.reshape(-1) > 0.5) & (occ.reshape(-1) > 0.5)
    on = pos & (start.reshape(-1) <= 0)
    up = pos & (start.reshape(-1) > 0)
    return {
        "n_true": int(pos.sum()),
        "n_ongoing": int(on.sum()),
        "n_upcoming": int(up.sum()),
    }


def collect_label_stats(cfg: dict[str, Any], data_dir: Path) -> dict[str, Any]:
    bundle = load_factory_bn_bundle(data_dir)
    splits = _split_sets(bundle, cfg)
    window_size = float(bundle["window_size_s"])
    input_window = int(cfg.get("input_window", 30))
    max_remain = int(cfg.get("max_remain_windows", 15))
    k_occ = int(cfg.get("occupancy_horizon_windows", 15))
    hot_min = int(cfg.get("hot_min_windows", 8))
    hot_gap = int(cfg.get("hot_gap_windows", 1))
    score_thr = float(cfg.get("hot_score_threshold", 0.55))
    ev_min = int(cfg.get("event_min_windows", 5))
    max_start = parse_max_start_windows(cfg.get("event_max_start_windows"))

    out: dict[str, Any] = {
        "data_dir": str(data_dir),
        "event_min_windows": ev_min,
        "event_max_start_windows": max_start,
        "hot_min_windows": hot_min,
        "hot_score_threshold": score_thr,
        "splits": {},
    }
    for split_name, ep_set in splits.items():
        cell_inter = 0
        cell_union = 0
        cell_ops = 0
        cell_score = 0
        cell_valid = 0
        ev_ops = {"n_true": 0, "n_ongoing": 0, "n_upcoming": 0}
        ev_score = {"n_true": 0, "n_ongoing": 0, "n_upcoming": 0}
        ev_both = 0
        ev_ops_only = 0
        ev_score_only = 0
        n_windows = 0
        for ep in bundle["episodes"]:
            name = str(ep.get("name") or ep["episode_id"])
            if name not in ep_set:
                continue
            feats = ep["features"]
            scores = ep["scores"]
            jobs_rem = np.asarray(
                ep.get("jobs_remaining", np.linspace(feats.shape[0], 1, feats.shape[0], dtype=np.float32)),
                dtype=np.float32,
            )
            done_ti = first_done_index(jobs_rem)
            ops = ops_hot_mask(
                feats,
                window_size_s=window_size,
                min_hot_windows=hot_min,
                gap_windows=hot_gap,
            )
            score_hot = node_hot_mask(
                feats,
                scores,
                score_threshold=score_thr,
                window_size_s=window_size,
                min_hot_windows=hot_min,
                gap_windows=hot_gap,
            )
            occ = occupancy_node_mask(feats)
            t_len = int(feats.shape[0])
            t_hi = min(t_len, int(done_ti))
            for t in range(input_window, t_hi):
                label_idx = t - 1
                if float(jobs_rem[label_idx]) <= 0:
                    continue
                _, y_ops, remain_mask, _ = pack_remain_target(
                    scores,
                    ops,
                    t=t,
                    done_ti=done_ti,
                    max_remain_windows=max_remain,
                    occupancy_horizon_windows=k_occ,
                )
                _, y_score, _, _ = pack_remain_target(
                    scores,
                    score_hot,
                    t=t,
                    done_ti=done_ti,
                    max_remain_windows=max_remain,
                    occupancy_horizon_windows=k_occ,
                )
                k_use = int(remain_mask.sum())
                if k_use <= 0:
                    continue
                node_ok = occ > 0.5
                a = y_ops[:k_use][:, node_ok] >= 0.5
                b = y_score[:k_use][:, node_ok] >= 0.5
                cell_ops += int(a.sum())
                cell_score += int(b.sum())
                cell_inter += int((a & b).sum())
                cell_union += int((a | b).sum())
                cell_valid += int(a.size)
                will_o, start_o, _ = node_event_targets(
                    y_ops,
                    min_windows=ev_min,
                    remain_mask=remain_mask,
                    occ_node_mask=occ,
                    max_start_windows=max_start,
                )
                will_s, start_s, _ = node_event_targets(
                    y_score,
                    min_windows=ev_min,
                    remain_mask=remain_mask,
                    occ_node_mask=occ,
                    max_start_windows=max_start,
                )
                c_o = _event_counts(will_o, start_o, occ)
                c_s = _event_counts(will_s, start_s, occ)
                for k in ev_ops:
                    ev_ops[k] += c_o[k]
                    ev_score[k] += c_s[k]
                both = (will_o > 0.5) & (will_s > 0.5) & (occ > 0.5)
                ev_both += int(both.sum())
                ev_ops_only += int(((will_o > 0.5) & (will_s <= 0.5) & (occ > 0.5)).sum())
                ev_score_only += int(((will_s > 0.5) & (will_o <= 0.5) & (occ > 0.5)).sum())
                n_windows += 1
        jaccard = (cell_inter / cell_union) if cell_union else 0.0
        out["splits"][split_name] = {
            "n_episodes": len(ep_set),
            "n_windows": n_windows,
            "occupancy_cells": {
                "valid": cell_valid,
                "ops_pos": cell_ops,
                "score_pos": cell_score,
                "intersection": cell_inter,
                "union": cell_union,
                "jaccard": jaccard,
                "ops_only": cell_ops - cell_inter,
                "score_only": cell_score - cell_inter,
            },
            "events_ops": ev_ops,
            "events_score": ev_score,
            "events_both": ev_both,
            "events_ops_only": ev_ops_only,
            "events_score_only": ev_score_only,
        }
    return out


def _brief(metrics: dict[str, float]) -> str:
    return (
        f"rep_p={metrics.get('report_precision', 0):.3f} "
        f"rep_r={metrics.get('report_recall', 0):.3f} "
        f"rep_f1={metrics.get('report_f1', 0):.3f} "
        f"who_p={metrics.get('who_precision', 0):.3f} "
        f"who_r={metrics.get('who_recall', 0):.3f} "
        f"up_r={metrics.get('report_recall_upcoming', 0):.3f} "
        f"on_r={metrics.get('report_recall_ongoing', 0):.3f} "
        f"st_mae={metrics.get('start_mae', 0):.2f} "
        f"dur_mae={metrics.get('dur_mae', 0):.2f} "
        f"n_true={int(metrics.get('n_true_who', 0))} "
        f"n_pred={int(metrics.get('n_pred_who', 0))} "
        f"thr={metrics.get('report_threshold_used', 0):.2f} "
        f"hot_p={metrics.get('hot_precision', 0):.3f} "
        f"hot_r={metrics.get('hot_recall', 0):.3f} "
        f"score_mae={metrics.get('score_mae', 0):.4f}"
    )


def _event_kw(cfg: dict[str, Any], *, threshold: float | None = None) -> dict[str, Any]:
    return dict(
        event_iou_min=float(cfg.get("event_iou_min", 0.5)),
        event_min_windows=int(cfg.get("event_min_windows", cfg.get("hot_min_windows", 8))),
        event_report_threshold=float(
            cfg.get("event_report_threshold", 0.70) if threshold is None else threshold
        ),
        start_tol_windows=int(cfg.get("start_tol_windows", 3)),
        ongoing_will_floor=float(cfg.get("ongoing_will_floor", 0.62)),
        force_ongoing_will=bool(cfg.get("force_ongoing_will", False)),
        force_to=float(
            cfg.get(
                "event_lift_to",
                cfg.get("ckpt_min_report_precision", cfg.get("event_report_threshold", 0.70)),
            )
        ),
        force_require_dur=bool(cfg.get("event_force_require_dur", True)),
        max_start_windows=parse_max_start_windows(cfg.get("event_max_start_windows")),
        report_ongoing_only=bool(cfg.get("event_report_ongoing_only", False)),
    )


def eval_ckpt_both_labels(
    ckpt_path: Path,
    data_dir: Path,
    *,
    device_name: str | None,
    batch_size: int,
) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader

    from factory_bn.infer import _data_feature_from_ckpt, _load_ckpt
    from factory_bn.model import BNPDFormer
    from factory_bn.train import _epoch_loop

    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = _load_ckpt(ckpt_path, torch.device("cpu"))
    cfg = dict(ckpt["config"])
    cfg["device"] = device
    meta = ckpt["data_meta"]
    data_feature = _data_feature_from_ckpt(meta)
    model = BNPDFormer(cfg, data_feature).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    sweep = cfg.get("report_threshold_sweep") or []
    model._report_threshold_sweep = [float(x) for x in sweep]
    model._report_ckpt_min_precision = float(cfg.get("ckpt_min_report_precision", 0.0) or 0.0)

    feature_scaler = Scaler(
        mean=np.asarray(meta["feature_scaler_mean"], dtype=np.float32),
        std=np.asarray(meta["feature_scaler_std"], dtype=np.float32),
    )
    score_scaler = Scaler(
        mean=np.asarray(meta["score_scaler_mean"], dtype=np.float32),
        std=np.asarray(meta["score_scaler_std"], dtype=np.float32),
    )
    bundle = load_factory_bn_bundle(data_dir)
    splits = _split_sets(bundle, cfg)
    window_size = float(bundle["window_size_s"])
    report: dict[str, Any] = {
        "ckpt": str(ckpt_path),
        "data_dir": str(data_dir),
        "ckpt_train_mode": str(cfg.get("train_mode") or ""),
        "ckpt_epoch": int(ckpt.get("epoch") or 0),
        "labels": {},
    }
    for label_mode in ("unsupervised", "supervised"):
        samples = _build_samples(
            bundle["episodes"],
            input_window=int(cfg.get("input_window", 30)),
            output_window=int(cfg.get("output_window", 1)),
            horizon_windows=max(1, int(round(float(cfg.get("horizon_s", 180)) / window_size))),
            max_hist_events=int(cfg.get("max_hist_events", 8)),
            window_size_s=window_size,
            horizon_s=float(cfg.get("horizon_s", 180)),
            remain_to_jobs_done=bool(cfg.get("remain_to_jobs_done", True)),
            max_remain_windows=int(cfg.get("max_remain_windows", 15)),
            hot_score_threshold=float(cfg.get("hot_score_threshold", 0.55)),
            occupancy_horizon_windows=int(cfg.get("occupancy_horizon_windows", 15)),
            hot_min_windows=int(cfg.get("hot_min_windows", 8)),
            hot_gap_windows=int(cfg.get("hot_gap_windows", 1)),
            train_mode=label_mode,
        )
        split_metrics: dict[str, Any] = {}
        frozen_thr: float | None = None
        for split_name in ("val", "test"):
            subset = [s for s in samples if str(s.get("episode_name")) in splits[split_name]]
            ds = FactoryBNWindowDataset(subset, feature_scaler, score_scaler)
            loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
            if split_name == "val":
                model._report_threshold_sweep = [float(x) for x in sweep]
                ev_kw = _event_kw(cfg)
            else:
                model._report_threshold_sweep = []
                ev_kw = _event_kw(cfg, threshold=frozen_thr)
            metrics = _epoch_loop(
                model,
                loader,
                None,
                device,
                train=False,
                cause_majority=int(meta.get("cause_majority", -1)),
                hot_eval_threshold=float(cfg.get("hot_eval_threshold", 0.55)),
                **ev_kw,
            )
            if split_name == "val":
                frozen_thr = float(metrics.get("report_threshold_used", cfg.get("event_report_threshold", 0.70)))
            keep = {
                k: float(v)
                for k, v in metrics.items()
                if isinstance(v, (int, float, np.floating))
            }
            split_metrics[split_name] = keep
            print(f"[{label_mode}/{split_name}] n={len(subset)} {_brief(keep)}")
        report["labels"][label_mode] = split_metrics
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["labels", "eval"], required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--config", default="factory_bn/configs/FactoryBN_dense_f1_p80.json")
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--out_json", default=None)
    args = parser.parse_args()

    data_dir = _resolve(args.data_dir)
    cfg = json.loads(_resolve(args.config).read_text(encoding="utf-8"))
    if args.phase == "labels":
        report = collect_label_stats(cfg, data_dir)
        out = _resolve(args.out_json or "libcity/cache/model_cache/compare_unsup_sup_labels.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        print(f"[labels] wrote {out}")
        return

    if not args.ckpt:
        raise SystemExit("--ckpt is required for --phase eval")
    report = eval_ckpt_both_labels(
        _resolve(args.ckpt),
        data_dir,
        device_name=args.device,
        batch_size=int(args.batch_size),
    )
    out = _resolve(args.out_json or "libcity/cache/model_cache/compare_unsup_sup_eval.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[eval] wrote {out}")


if __name__ == "__main__":
    main()
