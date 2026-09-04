"""Evaluate a saved BNPDFormer ckpt on a pack using the ckpt scalers (no refit, no train).

All episodes are test. Prints overall occupancy metrics and slices by mix group.

Example::

    python -m factory_bn.eval_ckpt \\
      --ckpt libcity/cache/model_cache/n10_i1_20ep_s9/BNPDFormer_best.pt \\
      --data_dir raw_data/n10_mix_ood
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

_PDFORMER_ROOT = Path(__file__).resolve().parent.parent
if str(_PDFORMER_ROOT) not in sys.path:
    sys.path.insert(0, str(_PDFORMER_ROOT))

from factory_bn.dataset import (  # noqa: E402
    FactoryBNWindowDataset,
    Scaler,
    _build_samples,
    load_factory_bn_bundle,
)
from factory_bn.infer import _data_feature_from_ckpt, _load_ckpt  # noqa: E402
from factory_bn.model import BNPDFormer  # noqa: E402
from factory_bn.train import _epoch_loop  # noqa: E402

NO_HUMAN = {
    "n10_mix_mm1.0",
    "n10_mix_ml1.0",
    "n10_mix_lm1.0",
    "n10_mix_mlm1.0",
}
PAIRS = {
    "n10_mix_mm1.0",
    "n10_mix_ml1.0",
    "n10_mix_lm1.0",
    "n10_mix_hl1.0",
    "n10_mix_mh1.0",
    "n10_mix_hm1.0",
}


def _run_prefix(episode_name: str) -> str:
    return str(episode_name).split("__", 1)[0]


def _align_scaler(scaler: Scaler, feat_dim: int) -> Scaler:
    mean = np.asarray(scaler.mean, dtype=np.float32).reshape(-1)
    std = np.asarray(scaler.std, dtype=np.float32).reshape(-1)
    if mean.size == feat_dim:
        return Scaler(mean=mean, std=np.where(std < 1e-6, 1.0, std))
    if mean.size < feat_dim:
        extra = feat_dim - mean.size
        mean = np.concatenate([mean, np.zeros(extra, dtype=np.float32)])
        std = np.concatenate([std, np.ones(extra, dtype=np.float32)])
        print(f"[eval] pad feature scaler {mean.size - extra} → {feat_dim}")
        return Scaler(mean=mean, std=std)
    print(f"[eval] trim feature scaler {mean.size} → {feat_dim}")
    return Scaler(mean=mean[:feat_dim], std=np.where(std[:feat_dim] < 1e-6, 1.0, std[:feat_dim]))


def _brief(metrics: dict[str, float]) -> str:
    return (
        f"P={metrics.get('hot_precision', 0):.3f} "
        f"R={metrics.get('hot_recall', 0):.3f} "
        f"F1={metrics.get('hot_f1', 0):.3f} "
        f"m_p={metrics.get('hot_precision_machine', 0):.3f} "
        f"g_p={metrics.get('hot_precision_gantry', 0):.3f} "
        f"a_p={metrics.get('hot_precision_agv', 0):.3f} "
        f"w_p={metrics.get('hot_precision_workbench', 0):.3f} "
        f"ev_p={metrics.get('event_precision', 0):.3f} "
        f"ev_r={metrics.get('event_recall', 0):.3f} "
        f"ev_f1={metrics.get('event_f1', 0):.3f} "
        f"who_p={metrics.get('who_precision', 0):.3f} "
        f"who_r={metrics.get('who_recall', 0):.3f} "
        f"rep_p={metrics.get('report_precision', 0):.3f} "
        f"rep_r={metrics.get('report_recall', 0):.3f} "
        f"rep_f1={metrics.get('report_f1', 0):.3f} "
        f"pred={metrics.get('hot_pred_pos_rate', 0):.3f} "
        f"true={metrics.get('hot_pos_rate', 0):.3f} "
        f"remain_mae={metrics.get('remain_len_mae', 0):.1f}"
    )


def eval_loader(
    model: BNPDFormer,
    samples: list[dict[str, Any]],
    feature_scaler: Scaler,
    score_scaler: Scaler,
    *,
    batch_size: int,
    device: torch.device,
    cause_majority: int,
    hot_eval_threshold: float,
    event_iou_min: float = 0.5,
    event_min_windows: int = 8,
    event_report_threshold: float = 0.70,
    start_tol_windows: int = 3,
    ongoing_will_floor: float = 0.62,
    force_ongoing_will: bool = False,
    max_start_windows: int | None = None,
) -> dict[str, float]:
    if not samples:
        return {}
    ds = FactoryBNWindowDataset(samples, feature_scaler, score_scaler)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return _epoch_loop(
        model,
        loader,
        None,
        device,
        train=False,
        cause_majority=cause_majority,
        hot_eval_threshold=hot_eval_threshold,
        event_iou_min=event_iou_min,
        event_min_windows=event_min_windows,
        event_report_threshold=event_report_threshold,
        start_tol_windows=start_tol_windows,
        ongoing_will_floor=ongoing_will_floor,
        force_ongoing_will=force_ongoing_will,
        max_start_windows=max_start_windows,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument(
        "--out_json",
        type=str,
        default=None,
        help="Optional metrics json path.",
    )
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.is_absolute():
        ckpt_path = (_PDFORMER_ROOT / ckpt_path).resolve()
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = (_PDFORMER_ROOT / data_dir).resolve()

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    ckpt = _load_ckpt(ckpt_path, torch.device("cpu"))
    cfg = dict(ckpt["config"])
    cfg["device"] = device
    meta = ckpt["data_meta"]
    data_feature = _data_feature_from_ckpt(meta)
    model = BNPDFormer(cfg, data_feature).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    feature_scaler = Scaler(
        mean=np.asarray(meta["feature_scaler_mean"], dtype=np.float32),
        std=np.asarray(meta["feature_scaler_std"], dtype=np.float32),
    )
    score_scaler = Scaler(
        mean=np.asarray(meta["score_scaler_mean"], dtype=np.float32),
        std=np.asarray(meta["score_scaler_std"], dtype=np.float32),
    )

    bundle = load_factory_bn_bundle(data_dir)
    window_size = float(bundle["window_size_s"])
    input_window = int(cfg.get("input_window", 30))
    occupancy_horizon_windows = int(cfg.get("occupancy_horizon_windows", 15))
    max_remain_windows = int(cfg.get("max_remain_windows", 15))
    samples = _build_samples(
        bundle["episodes"],
        input_window=input_window,
        output_window=int(cfg.get("output_window", 1)),
        horizon_windows=max(1, int(round(float(cfg.get("horizon_s", 180)) / window_size))),
        max_hist_events=int(cfg.get("max_hist_events", 8)),
        window_size_s=window_size,
        horizon_s=float(cfg.get("horizon_s", 180)),
        remain_to_jobs_done=bool(cfg.get("remain_to_jobs_done", True)),
        max_remain_windows=max_remain_windows,
        hot_score_threshold=float(cfg.get("hot_score_threshold", 0.55)),
        occupancy_horizon_windows=occupancy_horizon_windows,
        hot_min_windows=int(cfg.get("hot_min_windows", 8)),
        hot_gap_windows=int(cfg.get("hot_gap_windows", 1)),
    )
    if not samples:
        raise SystemExit(f"No samples under {data_dir}")
    feat_dim = int(samples[0]["x"].shape[-1])
    feature_scaler = _align_scaler(feature_scaler, feat_dim)
    ckpt_f = int(meta.get("feature_dim") or feat_dim)
    if feat_dim != ckpt_f:
        print(f"[eval] warning pack F={feat_dim} ckpt F={ckpt_f}")

    cause_majority = int(meta.get("cause_majority", -1))
    hot_thr = float(cfg.get("hot_eval_threshold", 0.55))
    batch_size = int(args.batch_size)
    event_kw = dict(
        event_iou_min=float(cfg.get("event_iou_min", 0.5)),
        event_min_windows=int(cfg.get("event_min_windows", cfg.get("hot_min_windows", 8))),
        event_report_threshold=float(cfg.get("event_report_threshold", 0.70)),
        start_tol_windows=int(cfg.get("start_tol_windows", 3)),
        ongoing_will_floor=float(cfg.get("ongoing_will_floor", 0.62)),
        force_ongoing_will=bool(cfg.get("force_ongoing_will", False)),
        max_start_windows=(
            None
            if cfg.get("event_max_start_windows") in (None, "", False)
            else int(cfg.get("event_max_start_windows"))
        ),
    )

    by_run: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for s in samples:
        by_run[_run_prefix(str(s.get("episode_name") or ""))].append(s)

    slices: dict[str, list[dict[str, Any]]] = {
        "all": samples,
        "no_human": [s for s in samples if _run_prefix(str(s.get("episode_name"))) in NO_HUMAN],
        "with_human": [s for s in samples if _run_prefix(str(s.get("episode_name"))) not in NO_HUMAN],
        "pairs": [s for s in samples if _run_prefix(str(s.get("episode_name"))) in PAIRS],
        "triple_quad": [s for s in samples if _run_prefix(str(s.get("episode_name"))) not in PAIRS],
    }
    for run, run_samples in sorted(by_run.items()):
        slices[run] = run_samples

    report: dict[str, Any] = {
        "ckpt": str(ckpt_path),
        "data_dir": str(data_dir),
        "n_episodes": len({str(s.get("episode_name")) for s in samples}),
        "n_windows": len(samples),
        "ckpt_epoch": int(ckpt.get("epoch") or 0),
        "slices": {},
    }
    print(
        f"[eval] ckpt={ckpt_path.name} epoch={report['ckpt_epoch']} "
        f"device={device} episodes={report['n_episodes']} windows={report['n_windows']}"
    )
    for name, subset in slices.items():
        metrics = eval_loader(
            model,
            subset,
            feature_scaler,
            score_scaler,
            batch_size=batch_size,
            device=device,
            cause_majority=cause_majority,
            hot_eval_threshold=hot_thr,
            **event_kw,
        )
        n_ep = len({str(s.get("episode_name")) for s in subset})
        report["slices"][name] = {
            "n_episodes": n_ep,
            "n_windows": len(subset),
            "metrics": {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))},
        }
        print(f"  {name:<22} ep={n_ep:2d} n={len(subset):5d}  {_brief(metrics)}")

    out_json = Path(args.out_json) if args.out_json else (data_dir / "s9_ood_metrics.json")
    if not out_json.is_absolute():
        out_json = (_PDFORMER_ROOT / out_json).resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[eval] wrote {out_json}")


if __name__ == "__main__":
    main()
