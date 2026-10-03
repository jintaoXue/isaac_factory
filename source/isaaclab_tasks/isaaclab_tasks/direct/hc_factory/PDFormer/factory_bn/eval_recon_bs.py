"""Blocked / starved duration forecast eval from the unsupervised recon head.

The recon head predicts future window features ``(B, K, N, F)``; channels
``blocked_time_s`` (6) and ``starved_time_s`` (7) are seconds per 60 s window.
Granularity and horizon follow the training setup (window_size_s, K =
occupancy_horizon_windows), no re-aggregation.

Example::

    cd source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/PDFormer
    python -m factory_bn.eval_recon_bs eval \\
        --run_dir libcity/cache/model_cache/dense_i1_12_3_start5_min8_cause4_opt_main_seed42
    python -m factory_bn.eval_recon_bs summarize
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

_PDFORMER_ROOT = Path(__file__).resolve().parent.parent
if str(_PDFORMER_ROOT) not in sys.path:
    sys.path.insert(0, str(_PDFORMER_ROOT))

from factory_bn.dataset import build_dataloaders
from factory_bn.model import BNPDFormer
from factory_bn.remain import occupancy_node_mask

CHANNELS = {"blocked": 6, "starved": 7}
DEFAULT_OUT = _PDFORMER_ROOT / "libcity/cache/recon_eval"
MAIN_RUN = "dense_i1_12_3_start5_min8_cause4_opt_main_seed42"


def _load_ckpt(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _mae_rmse(err: np.ndarray) -> tuple[float, float]:
    if err.size == 0:
        return float("nan"), float("nan")
    return float(np.abs(err).mean()), float(np.sqrt((err**2).mean()))


def _r2(y: np.ndarray, p: np.ndarray) -> float:
    if y.size < 2:
        return float("nan")
    ss_res = float(((y - p) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 1e-9 else float("nan")


@torch.no_grad()
def evaluate_run(run_dir: Path, out_root: Path, device: torch.device) -> Path:
    ckpt_path = run_dir / "BNPDFormer_best.pt"
    ckpt = _load_ckpt(ckpt_path)
    cfg = dict(ckpt["config"])
    meta = ckpt["data_meta"]
    if str(cfg.get("train_mode")) != "unsupervised":
        raise SystemExit(f"{run_dir.name}: train_mode={cfg.get('train_mode')} has no recon head")

    data_dir = Path(cfg["data_dir"])
    if not data_dir.is_absolute():
        data_dir = (_PDFORMER_ROOT / data_dir).resolve()
    seed = int(cfg.get("seed", 42))
    _, _, test_loader, data_feature = build_dataloaders(
        data_dir=data_dir,
        input_window=int(cfg.get("input_window", 30)),
        output_window=int(cfg.get("output_window", 1)),
        horizon_s=float(cfg.get("horizon_s", 180)),
        batch_size=int(cfg.get("batch_size", 16)),
        train_ratio=float(cfg.get("train_rate", 0.7)),
        val_ratio=float(cfg.get("eval_rate", 0.15)),
        seed=int(cfg.get("split_seed", seed)),
        max_hist_events=int(cfg.get("max_hist_events", 8)),
        remain_to_jobs_done=bool(cfg.get("remain_to_jobs_done", True)),
        max_remain_windows=int(cfg.get("max_remain_windows", 15)),
        hot_score_threshold=float(cfg.get("hot_score_threshold", 0.55)),
        occupancy_horizon_windows=int(
            cfg.get("occupancy_horizon_windows", cfg.get("max_remain_windows", 15))
        ),
        hot_min_windows=int(cfg.get("hot_min_windows", 8)),
        hot_gap_windows=int(cfg.get("hot_gap_windows", 1)),
        hot_smoothing_order=str(cfg.get("hot_smoothing_order", "legacy")),
        min_episode_jobs_total=float(cfg.get("min_episode_jobs_total", 0.0)),
        train_only_contains=list(cfg.get("train_only_contains") or []),
        train_mode="unsupervised",
        observed_resource_types=list(cfg.get("observed_resource_types") or []),
        masked_feature_indices=list(cfg.get("masked_feature_indices") or []),
    )
    data_feature.pop("train_feature_windows", None)
    for key in ("adj_mx", "sh_mx", "sem_mx", "pattern_keys"):
        if meta.get(key) is not None:
            data_feature[key] = meta[key]

    mean = np.asarray(meta["feature_scaler_mean"], dtype=np.float32)
    std = np.asarray(meta["feature_scaler_std"], dtype=np.float32)
    scaler = data_feature["feature_scaler"]
    if not (np.allclose(scaler.mean, mean, atol=1e-4) and np.allclose(scaler.std, std, atol=1e-4)):
        raise SystemExit(f"{run_dir.name}: feature scaler differs from checkpoint data_meta")

    cfg["device"] = device
    model = BNPDFormer(cfg, data_feature).to(device)
    # Older ckpts predate adj_row (derived buffer) and precursor_mlp (zero-init, event heads only).
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    bad = [k for k in missing if k != "adj_row" and not k.startswith("precursor_mlp.")]
    if bad or unexpected:
        raise SystemExit(f"{run_dir.name}: state_dict mismatch missing={bad} unexpected={list(unexpected)}")
    model.eval()

    samples = test_loader.dataset.samples
    window_s = float(data_feature.get("window_size_s") or 60.0)
    k_max = int(cfg.get("max_remain_windows", 15))
    k_occ = min(k_max, int(cfg.get("occupancy_horizon_windows", k_max)))
    ch_idx = list(CHANNELS.values())
    dev_mask = np.zeros(len(data_feature["resource_ids"]), dtype=bool)
    for s in samples:
        occ = s.get("occ_node_mask")
        occ = np.asarray(occ) if occ is not None else occupancy_node_mask(s["x"])
        dev_mask |= occ > 0.5
    dev_idx = np.flatnonzero(dev_mask)
    resource_ids = [str(x) for x in data_feature["resource_ids"]]
    resource_types = [str(x) for x in data_feature["resource_types"]]

    preds = []
    for batch in test_loader:
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        out = model(batch)
        recon = out["recon"][:, :k_occ].float().cpu().numpy()
        preds.append(recon[..., dev_idx, :][..., ch_idx])
    pred = np.concatenate(preds, axis=0) * std[ch_idx] + mean[ch_idx]
    pred = np.clip(pred, 0.0, window_s)

    n = len(samples)
    true = np.zeros_like(pred)
    valid = np.zeros(pred.shape[:2], dtype=bool)
    persist = np.zeros((n, len(dev_idx), len(ch_idx)), dtype=np.float32)
    episodes, t_index = [], np.zeros(n, dtype=np.int64)
    for i, s in enumerate(samples):
        y_x = np.asarray(s["y_x"], dtype=np.float32)[:k_occ]
        true[i] = y_x[:, dev_idx][..., ch_idx]
        length = int(min(float(s["remain_len"]), k_occ))
        valid[i, :length] = True
        persist[i] = np.asarray(s["x"], dtype=np.float32)[-1][dev_idx][:, ch_idx]
        episodes.append(str(s["episode_name"]))
        t_index[i] = int(s["t_index"])

    metrics: dict[str, Any] = {
        "run": run_dir.name,
        "ckpt_epoch": int(ckpt.get("epoch") or 0),
        "window_size_s": window_s,
        "horizon_windows": k_occ,
        "n_test_windows": n,
        "n_test_episodes": len(set(episodes)),
        "devices": [resource_ids[i] for i in dev_idx],
        "observed_resource_types": list(cfg.get("observed_resource_types") or []),
        "masked_feature_indices": list(cfg.get("masked_feature_indices") or []),
        "graph_mode": cfg.get("graph_mode"),
    }
    groups = {
        "all": np.ones(len(dev_idx), dtype=bool),
        "machine": np.array([resource_types[i] == "machine" for i in dev_idx]),
    }
    for c, name in enumerate(CHANNELS):
        p, y = pred[..., c], true[..., c]
        for gname, gm in groups.items():
            one_err = (p[:, 0] - y[:, 0])[valid[:, 0]][:, gm]
            hor_err = (p - y)[valid][:, gm]
            per_err = (persist[:, :, c] - y[:, 0])[valid[:, 0]][:, gm]
            metrics[f"{name}_{gname}_mae_1step"], metrics[f"{name}_{gname}_rmse_1step"] = _mae_rmse(one_err)
            metrics[f"{name}_{gname}_mae_horizon"], metrics[f"{name}_{gname}_rmse_horizon"] = _mae_rmse(hor_err)
            metrics[f"{name}_{gname}_mae_persist"], metrics[f"{name}_{gname}_rmse_persist"] = _mae_rmse(per_err)
            y1 = y[:, 0][valid[:, 0]][:, gm]
            p1 = p[:, 0][valid[:, 0]][:, gm]
            act = y1 > 0
            metrics[f"{name}_{gname}_mae_1step_active"], metrics[f"{name}_{gname}_rmse_1step_active"] = _mae_rmse(
                (p1 - y1)[act]
            )
            metrics[f"{name}_{gname}_active_frac"] = float(act.mean()) if act.size else float("nan")
            metrics[f"{name}_{gname}_r2_1step"] = _r2(y1.ravel(), p1.ravel())
        metrics[f"{name}_all_mae_by_step"] = [
            _mae_rmse((p[:, k] - y[:, k])[valid[:, k]])[0] for k in range(k_occ)
        ]
        metrics[f"{name}_all_rmse_by_step"] = [
            _mae_rmse((p[:, k] - y[:, k])[valid[:, k]])[1] for k in range(k_occ)
        ]

    per_device = []
    v0 = valid[:, 0]
    for j, ni in enumerate(dev_idx):
        row: dict[str, Any] = {"device": resource_ids[ni], "type": resource_types[ni]}
        for c, name in enumerate(CHANNELS):
            y1, p1 = true[v0, 0, j, c], pred[v0, 0, j, c]
            row[f"{name}_mae_1step"], row[f"{name}_rmse_1step"] = _mae_rmse(p1 - y1)
            row[f"{name}_r2_1step"] = _r2(y1, p1)
            row[f"{name}_true_mean"] = float(y1.mean())
            row[f"{name}_true_std"] = float(y1.std())
            hv = valid
            row[f"{name}_mae_horizon"], row[f"{name}_rmse_horizon"] = _mae_rmse(
                (pred[..., j, c] - true[..., j, c])[hv]
            )
        per_device.append(row)
    metrics["per_device"] = per_device

    out_dir = out_root / run_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with (out_dir / "per_device.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(per_device[0].keys()))
        writer.writeheader()
        writer.writerows(per_device)
    np.savez_compressed(
        out_dir / "series.npz",
        pred=pred.astype(np.float32),
        true=true.astype(np.float32),
        valid=valid,
        persist=persist,
        episode=np.asarray(episodes),
        t_index=t_index,
        devices=np.asarray([resource_ids[i] for i in dev_idx]),
        types=np.asarray([resource_types[i] for i in dev_idx]),
        channels=np.asarray(list(CHANNELS)),
    )
    print(
        f"[recon-bs] {run_dir.name} epoch={metrics['ckpt_epoch']} n={n} "
        f"blocked MAE/RMSE 1step={metrics['blocked_all_mae_1step']:.2f}/{metrics['blocked_all_rmse_1step']:.2f} "
        f"starved MAE/RMSE 1step={metrics['starved_all_mae_1step']:.2f}/{metrics['starved_all_rmse_1step']:.2f}"
    )
    return out_dir


def _pick_key_device(main: dict[str, Any]) -> str:
    """Machine with above-median truth variability and the best mean 1-step R2."""
    rows = [r for r in main["per_device"] if r["type"] == "machine"]
    act = np.array([r["blocked_true_std"] + r["starved_true_std"] for r in rows])
    keep = [r for r, a in zip(rows, act) if a >= np.median(act)]
    def score(r: dict[str, Any]) -> float:
        vals = [r["blocked_r2_1step"], r["starved_r2_1step"]]
        vals = [v for v in vals if np.isfinite(v)]
        return float(np.mean(vals)) if vals else -1e9
    return max(keep, key=score)["device"]


def _pick_episode(series: dict[str, np.ndarray], dev_j: int) -> str:
    """Test episode where the key device has the most blocked+starved activity."""
    true = series["true"][:, 0, dev_j, :].sum(-1)
    eps = series["episode"]
    best, best_v = None, -1.0
    for e in np.unique(eps):
        v = float(true[eps == e].std())
        if v > best_v:
            best, best_v = str(e), v
    return best


def _label(run: str) -> str:
    name = run.replace("dense_i1_", "").replace("_min8", "").replace("_seed42", "")
    name = name.replace("_cause4_opt_main", "").replace("_cold_ep50", "")
    return name


def summarize(out_root: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = sorted(p for p in out_root.iterdir() if (p / "metrics.json").is_file())
    if not runs:
        raise SystemExit(f"No metrics under {out_root}")
    all_m = {p.name: json.loads((p / "metrics.json").read_text(encoding="utf-8")) for p in runs}
    main_name = MAIN_RUN if MAIN_RUN in all_m else runs[0].name
    key_dev = _pick_key_device(all_m[main_name])
    main_series = dict(np.load(out_root / main_name / "series.npz", allow_pickle=False))
    devices = [str(x) for x in main_series["devices"]]
    key_j = devices.index(key_dev)
    key_ep = _pick_episode(main_series, key_j)

    rows = []
    for name, m in all_m.items():
        row: dict[str, Any] = {"run": name, "label": _label(name), "epoch": m["ckpt_epoch"]}
        for ch in CHANNELS:
            for g in ("all", "machine"):
                for kind in ("1step", "horizon", "persist", "1step_active"):
                    row[f"{ch}_{g}_mae_{kind}"] = m[f"{ch}_{g}_mae_{kind}"]
                    row[f"{ch}_{g}_rmse_{kind}"] = m[f"{ch}_{g}_rmse_{kind}"]
                row[f"{ch}_{g}_r2_1step"] = m[f"{ch}_{g}_r2_1step"]
            kd = next(r for r in m["per_device"] if r["device"] == key_dev)
            row[f"{ch}_key_mae_1step"] = kd[f"{ch}_mae_1step"]
            row[f"{ch}_key_rmse_1step"] = kd[f"{ch}_rmse_1step"]
            row[f"{ch}_key_r2_1step"] = kd[f"{ch}_r2_1step"]
        rows.append(row)
    with (out_root / "summary.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Recon blocked/starved eval summary",
        "",
        f"- window = {all_m[main_name]['window_size_s']:.0f} s, horizon K = {all_m[main_name]['horizon_windows']} windows",
        f"- all devices = {len(devices)} supervised nodes: {', '.join(devices)}",
        f"- key device = `{key_dev}` (picked on `{main_name}`)",
        f"- key episode for plots = `{key_ep}`",
        "",
        "Units: seconds per window. 1step = first future window; horizon = mean over all valid K steps; "
        "persist = last observed window copied forward (sanity floor, 1 step).",
        "",
        "| run | Blocked MAE 1step | Blocked RMSE 1step | Starved MAE 1step | Starved RMSE 1step "
        "| Blocked MAE hor | Blocked RMSE hor | Starved MAE hor | Starved RMSE hor "
        "| Key Blocked MAE/RMSE | Key Starved MAE/RMSE |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['label']} | {r['blocked_all_mae_1step']:.2f} | {r['blocked_all_rmse_1step']:.2f} "
            f"| {r['starved_all_mae_1step']:.2f} | {r['starved_all_rmse_1step']:.2f} "
            f"| {r['blocked_all_mae_horizon']:.2f} | {r['blocked_all_rmse_horizon']:.2f} "
            f"| {r['starved_all_mae_horizon']:.2f} | {r['starved_all_rmse_horizon']:.2f} "
            f"| {r['blocked_key_mae_1step']:.2f}/{r['blocked_key_rmse_1step']:.2f} "
            f"| {r['starved_key_mae_1step']:.2f}/{r['starved_key_rmse_1step']:.2f} |"
        )
    lines += [
        "",
        "## Active windows only (true > 0) and R2, all devices, 1step",
        "",
        "| run | Blocked MAE act | Blocked RMSE act | Blocked R2 | Starved MAE act | Starved RMSE act | Starved R2 "
        "| Machine Starved MAE/RMSE | Machine Starved R2 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['label']} | {r['blocked_all_mae_1step_active']:.2f} | {r['blocked_all_rmse_1step_active']:.2f} "
            f"| {r['blocked_all_r2_1step']:.3f} | {r['starved_all_mae_1step_active']:.2f} "
            f"| {r['starved_all_rmse_1step_active']:.2f} | {r['starved_all_r2_1step']:.3f} "
            f"| {r['starved_machine_mae_1step']:.2f}/{r['starved_machine_rmse_1step']:.2f} "
            f"| {r['starved_machine_r2_1step']:.3f} |"
        )
    m0 = all_m[main_name]
    lines += [
        "",
        f"Persistence floor (all devices, 1step): blocked {m0['blocked_all_mae_persist']:.2f}/"
        f"{m0['blocked_all_rmse_persist']:.2f}, starved {m0['starved_all_mae_persist']:.2f}/"
        f"{m0['starved_all_rmse_persist']:.2f} (MAE/RMSE)",
        "",
        f"## Per-device ({_label(main_name)}, 1step)",
        "",
        "| device | type | Blocked MAE | Blocked RMSE | Blocked R2 | Starved MAE | Starved RMSE | Starved R2 | true std B/S |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in m0["per_device"]:
        lines.append(
            f"| {r['device']} | {r['type']} | {r['blocked_mae_1step']:.2f} | {r['blocked_rmse_1step']:.2f} "
            f"| {r['blocked_r2_1step']:.3f} | {r['starved_mae_1step']:.2f} | {r['starved_rmse_1step']:.2f} "
            f"| {r['starved_r2_1step']:.3f} | {r['blocked_true_std']:.1f}/{r['starved_true_std']:.1f} |"
        )
    (out_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    fig_dir = out_root / "figures"
    fig_dir.mkdir(exist_ok=True)

    def _series(name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        s = np.load(out_root / name / "series.npz", allow_pickle=False)
        devs = [str(x) for x in s["devices"]]
        j = devs.index(key_dev)
        sel = (s["episode"] == key_ep) & s["valid"][:, 0]
        order = np.argsort(s["t_index"][sel])
        t = s["t_index"][sel][order]
        return t, s["true"][sel][order][:, 0, j, :], s["pred"][sel][order][:, 0, j, :], s["persist"][sel][order][:, j, :]

    t, y, p, _ = _series(main_name)
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for c, ch in enumerate(CHANNELS):
        axes[c].plot(t, y[:, c], color="black", lw=1.6, label="True")
        axes[c].plot(t, p[:, c], color="tab:red", lw=1.2, ls="--", label="BNPDFormer")
        axes[c].set_ylabel(f"{ch} time (s / window)")
        axes[c].grid(alpha=0.3)
        axes[c].legend(loc="upper right")
    axes[-1].set_xlabel("window index (60 s)")
    fig.suptitle(f"{key_dev}  |  {key_ep}  |  one-step-ahead")
    fig.tight_layout()
    fig.savefig(fig_dir / "key_device_main.png", dpi=200)
    plt.close(fig)

    families = {
        "start": [n for n in all_m if "12_3_start" in n],
        "struct": [main_name] + [n for n in all_m if n.startswith("ablation_") and "start5" in n],
        "entity": [main_name] + [n for n in all_m if "entity_" in n and "start5" in n],
    }
    for fam, names in families.items():
        names = [n for n in dict.fromkeys(names) if n in all_m]
        if len(names) < 2:
            continue
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        t0, y0, _, _ = _series(names[0])
        for c, ch in enumerate(CHANNELS):
            axes[c].plot(t0, y0[:, c], color="black", lw=1.8, label="True")
            for n in names:
                tn, _, pn, _ = _series(n)
                axes[c].plot(tn, pn[:, c], lw=1.0, ls="--", label=_label(n))
            axes[c].set_ylabel(f"{ch} time (s / window)")
            axes[c].grid(alpha=0.3)
        axes[0].legend(loc="upper right", fontsize=7, ncol=2)
        axes[-1].set_xlabel("window index (60 s)")
        fig.suptitle(f"{key_dev}  |  {key_ep}  |  one-step-ahead ({fam})")
        fig.tight_layout()
        fig.savefig(fig_dir / f"key_device_{fam}.png", dpi=200)
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for c, ch in enumerate(CHANNELS):
        for n in families["start"] or [main_name]:
            m = all_m[n]
            axes[c].plot(np.arange(1, m["horizon_windows"] + 1), m[f"{ch}_all_mae_by_step"], marker="o", ms=3, label=_label(n))
        axes[c].set_xlabel("forecast step k (60 s windows)")
        axes[c].set_ylabel(f"{ch} MAE (s)")
        axes[c].grid(alpha=0.3)
        axes[c].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "mae_by_step.png", dpi=200)
    plt.close(fig)

    (out_root / "selection.json").write_text(
        json.dumps({"main_run": main_name, "key_device": key_dev, "key_episode": key_ep}, indent=2),
        encoding="utf-8",
    )
    print(f"[recon-bs] summary -> {out_root / 'summary.md'}  key_device={key_dev} episode={key_ep}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    ev = sub.add_parser("eval")
    ev.add_argument("--run_dir", required=True)
    ev.add_argument("--out_root", default=str(DEFAULT_OUT))
    ev.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    sm = sub.add_parser("summarize")
    sm.add_argument("--out_root", default=str(DEFAULT_OUT))
    args = ap.parse_args()
    out_root = Path(args.out_root)
    if args.cmd == "eval":
        run_dir = Path(args.run_dir)
        if not run_dir.is_absolute():
            run_dir = (_PDFORMER_ROOT / run_dir).resolve()
        evaluate_run(run_dir, out_root, torch.device(args.device))
    else:
        summarize(out_root)


if __name__ == "__main__":
    main()
