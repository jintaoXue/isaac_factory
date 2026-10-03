"""Bottleneck forecast eval: who / when / how long per device, plus figures.

Uses the same decode path as ``train._epoch_loop`` (``model.predict`` →
``station_report_metrics``), so will15 P/R/F1 on test reproduces
``last_metrics.json``. On top of that it reports bottleneck duration / start
MAE + RMSE (minutes = 60 s windows), one-step bottleneck-state accuracy, and
per-device numbers, and dumps per-window series for plotting.

Example::

    cd source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/PDFormer
    python -m factory_bn.eval_bottleneck eval \\
        --run_dir libcity/cache/model_cache/dense_i1_12_3_start5_min8_cause4_opt_main_seed42
    python -m factory_bn.eval_bottleneck summarize
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
from factory_bn.remain import (
    apply_ongoing_will_force,
    node_event_targets,
    parse_max_start_windows,
    rasterize_node_events,
    station_report_metrics,
)
from factory_bn.train import _near_remain_mask

DEFAULT_OUT = _PDFORMER_ROOT / "libcity/cache/bottleneck_eval"
MAIN_RUN = "dense_i1_12_3_start5_min8_cause4_opt_main_seed42"


def _load_ckpt(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _mae_rmse(err: np.ndarray) -> tuple[float, float]:
    err = np.asarray(err, dtype=np.float64)
    if err.size == 0:
        return float("nan"), float("nan")
    return float(np.abs(err).mean()), float(np.sqrt((err**2).mean()))


def _prf(tp: float, n_pred: float, n_true: float) -> tuple[float, float, float]:
    p = tp / n_pred if n_pred > 0 else 0.0
    r = tp / n_true if n_true > 0 else 0.0
    f = 2 * p * r / (p + r) if p + r > 0 else 0.0
    return p, r, f


def _event_kw(cfg: dict[str, Any]) -> dict[str, Any]:
    """Same knobs as ``train.event_eval_kw`` / ``_epoch_loop`` report_kw."""
    return dict(
        min_windows=int(cfg.get("event_min_windows", cfg.get("hot_min_windows", 8))),
        start_tol_windows=int(cfg.get("start_tol_windows", 3)),
        will_floor=float(cfg.get("ongoing_will_floor", 0.62)),
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
        ongoing_min_windows=int(cfg.get("event_ongoing_min_windows", 1)),
    )


def _build(run_dir: Path, device: torch.device) -> tuple[BNPDFormer, Any, dict[str, Any], dict[str, Any]]:
    ckpt = _load_ckpt(run_dir / "BNPDFormer_best.pt")
    cfg = dict(ckpt["config"])
    meta = ckpt["data_meta"]
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
        train_mode=str(cfg.get("train_mode") or "supervised"),
        observed_resource_types=list(cfg.get("observed_resource_types") or []),
        masked_feature_indices=list(cfg.get("masked_feature_indices") or []),
    )
    data_feature.pop("train_feature_windows", None)
    for key in ("adj_mx", "sh_mx", "sem_mx", "pattern_keys"):
        if meta.get(key) is not None:
            data_feature[key] = meta[key]
    cfg["device"] = device
    model = BNPDFormer(cfg, data_feature).to(device)
    # Older ckpts predate adj_row (derived buffer) and precursor_mlp (zero-init, event heads only).
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    bad = [k for k in missing if k != "adj_row" and not k.startswith("precursor_mlp.")]
    if bad or unexpected:
        raise SystemExit(f"{run_dir.name}: state_dict mismatch missing={bad} unexpected={list(unexpected)}")
    model.eval()
    return model, test_loader, cfg, ckpt


@torch.no_grad()
def evaluate_run(run_dir: Path, out_root: Path, device: torch.device) -> Path:
    model, test_loader, cfg, ckpt = _build(run_dir, device)
    near_k = int(getattr(model, "near_remain_windows", 60) or 60)

    ys, rs, occs, lasts, wills, starts, durs = [], [], [], [], [], [], []
    for batch in test_loader:
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        pred = model.predict(batch)
        ys.append(batch["y_hot"].cpu().numpy())
        rs.append(_near_remain_mask(batch["remain_mask"], near_k).cpu().numpy())
        occs.append(batch["occ_node_mask"].cpu().numpy())
        lasts.append(batch["hist_last_hot"].cpu().numpy())
        wills.append(pred["event_will_prob"].cpu().numpy())
        starts.append(pred["event_start_idx"].cpu().numpy())
        durs.append(pred["event_dur"].cpu().numpy())
    y = np.concatenate(ys)
    r = np.concatenate(rs)
    occ = np.concatenate(occs)
    last = np.concatenate(lasts)
    will_p = np.concatenate(wills)
    start_p = np.concatenate(starts)
    dur_p = np.concatenate(durs)
    n_nodes = will_p.shape[-1]
    dur_p = dur_p[:, :n_nodes]

    kw = _event_kw(cfg)
    run_metrics = json.loads((run_dir / "last_metrics.json").read_text(encoding="utf-8"))
    test_ref = run_metrics.get("test") or {}
    thr = float(test_ref.get("report_threshold_used", cfg.get("event_report_threshold", 0.70)))
    kw["force_to"] = max(float(kw["force_to"] or 0.0), thr)

    official = station_report_metrics(
        y, will_p, start_p, dur_p, r, occ, threshold=thr, hist_last_hot=last, **kw
    )

    y_will, y_start, y_dur = node_event_targets(
        y,
        min_windows=kw["min_windows"],
        remain_mask=r,
        occ_node_mask=occ,
        max_start_windows=kw["max_start_windows"],
        hist_last_hot=last,
        ongoing_min_windows=kw["ongoing_min_windows"],
    )
    wp, sp = apply_ongoing_will_force(
        will_p,
        start_p,
        dur_p,
        last,
        threshold=thr,
        min_windows=kw["min_windows"],
        will_floor=kw["will_floor"],
        force_will=kw["force_ongoing_will"],
        force_to=kw["force_to"],
        require_dur=kw["force_require_dur"],
        report_ongoing_only=kw["report_ongoing_only"],
    )
    node_ok = occ > 0.5
    pred_pos = (wp >= thr) & node_ok
    true_pos = (y_will > 0.5) & node_ok
    tp = pred_pos & true_pos
    p_dur = np.where(pred_pos, dur_p, 0.0)
    t_dur = np.where(true_pos, y_dur, 0.0)
    p_start = np.where(pred_pos, sp, 0)
    t_start = np.where(true_pos, y_start, 0)
    k_len = y.shape[1]
    pred_grid = rasterize_node_events(
        pred_pos.astype(np.float32), sp, dur_p, k_len, threshold=0.5, min_windows=kw["min_windows"]
    )
    true_grid = rasterize_node_events(
        true_pos.astype(np.float32), y_start, y_dur, k_len, threshold=0.5, min_windows=1
    )

    samples = test_loader.dataset.samples
    episodes = np.asarray([str(s["episode_name"]) for s in samples])
    t_index = np.asarray([int(s["t_index"]) for s in samples], dtype=np.int64)
    resource_ids = [str(x) for x in model.data_feature["resource_ids"]]
    resource_types = [str(x) for x in model.data_feature["resource_types"]]
    dev_idx = np.flatnonzero(node_ok.any(axis=0))

    m: dict[str, Any] = {
        "run": run_dir.name,
        "ckpt_epoch": int(ckpt.get("epoch") or 0),
        "report_threshold": thr,
        "horizon_windows": int(k_len),
        "n_test_windows": int(len(samples)),
        "devices": [resource_ids[i] for i in dev_idx],
        "ref_will15_f1": test_ref.get("will15_f1"),
    }
    for key in (
        "will15_precision", "will15_recall", "will15_f1",
        "report_precision", "report_recall", "report_f1",
        "who_recall_ongoing", "who_recall_upcoming",
        "start_mae", "dur_mae", "n_true_who", "n_pred_who",
    ):
        m[key] = float(official.get(key, float("nan")))

    m["dur_mae_tp"], m["dur_rmse_tp"] = _mae_rmse((dur_p - y_dur)[tp])
    m["start_mae_tp"], m["start_rmse_tp"] = _mae_rmse((sp - y_start)[tp])
    union = true_pos | pred_pos
    m["dur_mae_cells"], m["dur_rmse_cells"] = _mae_rmse((p_dur - t_dur)[node_ok])
    m["dur_mae_union"], m["dur_rmse_union"] = _mae_rmse((p_dur - t_dur)[union])
    m["start_mae_union"], m["start_rmse_union"] = _mae_rmse((p_start - t_start)[union])

    cell = (r > 0.5)[:, :, None] & node_ok[:, None, :]
    yb = y > 0.5
    pb = pred_grid > 0.5
    for name, sl in (("1step", slice(0, 1)), ("horizon", slice(None))):
        c = cell[:, sl]
        a, b = yb[:, sl][c], pb[:, sl][c]
        m[f"state_acc_{name}"] = float((a == b).mean()) if a.size else float("nan")
        p, rr, f = _prf(float((a & b).sum()), float(b.sum()), float(a.sum()))
        m[f"state_precision_{name}"], m[f"state_recall_{name}"], m[f"state_f1_{name}"] = p, rr, f
        tg = true_grid[:, sl][c]
        m[f"state_vs_event_acc_{name}"] = float(((tg > 0.5) == b).mean()) if a.size else float("nan")

    per_device = []
    for j in dev_idx:
        ok = node_ok[:, j]
        tpj, ppj, tj = tp[ok, j], pred_pos[ok, j], true_pos[ok, j]
        p, rr, f = _prf(float(tpj.sum()), float(ppj.sum()), float(tj.sum()))
        uj = union[ok, j]
        dmae, drmse = _mae_rmse((p_dur[ok, j] - t_dur[ok, j])[uj])
        tdmae, tdrmse = _mae_rmse((dur_p[ok, j] - y_dur[ok, j])[tpj])
        c1 = cell[:, 0, j]
        acc1 = float((yb[c1, 0, j] == pb[c1, 0, j]).mean()) if c1.any() else float("nan")
        per_device.append(
            {
                "device": resource_ids[j],
                "type": resource_types[j],
                "n_true": int(tj.sum()),
                "n_pred": int(ppj.sum()),
                "precision": p,
                "recall": rr,
                "f1": f,
                "dur_mae_union": dmae,
                "dur_rmse_union": drmse,
                "dur_mae_tp": tdmae,
                "dur_rmse_tp": tdrmse,
                "state_acc_1step": acc1,
            }
        )
    m["per_device"] = per_device

    out_dir = out_root / run_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
    with (out_dir / "per_device.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(per_device[0].keys()))
        writer.writeheader()
        writer.writerows(per_device)
    np.savez_compressed(
        out_dir / "series.npz",
        true_state=yb[:, 0, :] & node_ok,
        pred_state=pb[:, 0, :] & node_ok,
        valid_1step=cell[:, 0, :],
        true_dur=t_dur.astype(np.float32),
        pred_dur=p_dur.astype(np.float32),
        true_will=true_pos,
        pred_will=pred_pos,
        will_prob=wp.astype(np.float32),
        episode=episodes,
        t_index=t_index,
        resource_ids=np.asarray(resource_ids),
        resource_types=np.asarray(resource_types),
    )
    ref = m["ref_will15_f1"]
    print(
        f"[bn-eval] {run_dir.name} thr={thr:.2f} will15 F1={m['will15_f1']:.3f} (ref {ref if ref is None else f'{ref:.3f}'}) "
        f"dur MAE/RMSE union={m['dur_mae_union']:.2f}/{m['dur_rmse_union']:.2f} tp={m['dur_mae_tp']:.2f}/{m['dur_rmse_tp']:.2f} "
        f"state acc 1step={m['state_acc_1step']:.3f}"
    )
    return out_dir


def _label(run: str) -> str:
    name = run.replace("dense_i1_", "").replace("_min8", "").replace("_seed42", "")
    return name.replace("_cause4_opt_main", "").replace("_cold_ep50", "")


def _pick_key_pair(main: dict[str, Any], s: dict[str, np.ndarray]) -> tuple[str, str]:
    """(device, episode): device with ≥ median bottleneck count; episode with the most
    bottleneck onsets for it, ties broken by next-window state F1."""
    rows = [r for r in main["per_device"] if r["n_true"] > 0]
    med = float(np.median([r["n_true"] for r in rows]))
    rids = [str(x) for x in s["resource_ids"]]
    best, best_key = ("", ""), (-1, -1.0)
    for r in rows:
        if r["n_true"] < med:
            continue
        j = rids.index(r["device"])
        for e in np.unique(s["episode"]):
            sel = s["episode"] == e
            order = np.argsort(s["t_index"][sel])
            a = s["true_state"][sel][order][:, j]
            b = s["pred_state"][sel][order][:, j]
            if not a.any():
                continue
            onsets = int((np.diff(np.r_[0, a.astype(int)]) == 1).sum())
            f1 = 2 * float((a & b).sum()) / max(float(a.sum() + b.sum()), 1.0)
            key = (onsets, round(f1, 3))
            if key > best_key:
                best, best_key = (r["device"], str(e)), key
    return best


def _pick_busy_episode(s: dict[str, np.ndarray]) -> str:
    """Episode with the most true bottleneck (window, device) cells."""
    eps = np.unique(s["episode"])
    return str(max(eps, key=lambda e: int(s["true_state"][s["episode"] == e].sum())))


def summarize(out_root: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    runs = sorted(p for p in out_root.iterdir() if (p / "metrics.json").is_file())
    if not runs:
        raise SystemExit(f"No metrics under {out_root}")
    all_m = {p.name: json.loads((p / "metrics.json").read_text(encoding="utf-8")) for p in runs}
    main_name = MAIN_RUN if MAIN_RUN in all_m else runs[0].name
    main_s = dict(np.load(out_root / main_name / "series.npz", allow_pickle=False))
    rids = [str(x) for x in main_s["resource_ids"]]
    key_dev, key_ep = _pick_key_pair(all_m[main_name], main_s)
    key_j = rids.index(key_dev)
    busy_ep = _pick_busy_episode(main_s)

    rows = []
    for name, m in all_m.items():
        kd = next(r for r in m["per_device"] if r["device"] == key_dev)
        rows.append(
            {
                "run": name,
                "label": _label(name),
                "epoch": m["ckpt_epoch"],
                "thr": m["report_threshold"],
                "will15_precision": m["will15_precision"],
                "will15_recall": m["will15_recall"],
                "will15_f1": m["will15_f1"],
                "ref_will15_f1": m["ref_will15_f1"],
                "state_acc_1step": m["state_acc_1step"],
                "state_f1_1step": m["state_f1_1step"],
                "state_f1_horizon": m["state_f1_horizon"],
                "dur_mae_union": m["dur_mae_union"],
                "dur_rmse_union": m["dur_rmse_union"],
                "dur_mae_tp": m["dur_mae_tp"],
                "dur_rmse_tp": m["dur_rmse_tp"],
                "start_mae_tp": m["start_mae_tp"],
                "start_rmse_tp": m["start_rmse_tp"],
                "key_f1": kd["f1"],
                "key_dur_mae_union": kd["dur_mae_union"],
                "key_dur_rmse_union": kd["dur_rmse_union"],
                "key_state_acc_1step": kd["state_acc_1step"],
            }
        )
    with (out_root / "summary.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    m0 = all_m[main_name]
    lines = [
        "# Bottleneck forecast eval summary",
        "",
        f"- horizon K = {m0['horizon_windows']} windows (60 s), test windows = {m0['n_test_windows']}",
        f"- devices = {len(m0['devices'])}: {', '.join(m0['devices'])}",
        f"- key device = `{key_dev}` (picked on `{main_name}`), plot episode = `{key_ep}`, "
        f"all-device heatmap episode = `{busy_ep}`",
        "",
        "Durations / starts in minutes. `dur MAE/RMSE` = over every (window, device) where a bottleneck "
        "is true or predicted (missed → pred 0, false alarm → true 0); `tp` = correctly detected "
        "bottlenecks only. State = bottleneck yes/no per future window.",
        "",
        "| run | thr | P | R | will15 F1 | ref F1 | state acc 1step | state F1 1step | state F1 K | "
        "dur MAE | dur RMSE | dur MAE tp | dur RMSE tp | start MAE tp | start RMSE tp |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        ref = r["ref_will15_f1"]
        lines.append(
            f"| {r['label']} | {r['thr']:.2f} | {r['will15_precision']:.3f} | {r['will15_recall']:.3f} "
            f"| {r['will15_f1']:.3f} | {'-' if ref is None else f'{ref:.3f}'} "
            f"| {r['state_acc_1step']:.3f} | {r['state_f1_1step']:.3f} | {r['state_f1_horizon']:.3f} "
            f"| {r['dur_mae_union']:.2f} | {r['dur_rmse_union']:.2f} | {r['dur_mae_tp']:.2f} | {r['dur_rmse_tp']:.2f} "
            f"| {r['start_mae_tp']:.2f} | {r['start_rmse_tp']:.2f} |"
        )
    lines += [
        "",
        f"## Per-device ({_label(main_name)})",
        "",
        "| device | type | n_true | n_pred | P | R | F1 | dur MAE | dur RMSE | dur MAE tp | dur RMSE tp | state acc 1step |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for d in m0["per_device"]:
        lines.append(
            f"| {d['device']} | {d['type']} | {d['n_true']} | {d['n_pred']} | {d['precision']:.3f} "
            f"| {d['recall']:.3f} | {d['f1']:.3f} | {d['dur_mae_union']:.2f} | {d['dur_rmse_union']:.2f} "
            f"| {d['dur_mae_tp']:.2f} | {d['dur_rmse_tp']:.2f} | {d['state_acc_1step']:.3f} |"
        )
    (out_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    fig_dir = out_root / "figures"
    fig_dir.mkdir(exist_ok=True)

    def _ep(name: str, ep: str | None = None) -> dict[str, np.ndarray]:
        s = np.load(out_root / name / "series.npz", allow_pickle=False)
        sel = s["episode"] == (ep or key_ep)
        order = np.argsort(s["t_index"][sel])
        return {k: s[k][sel][order] for k in ("t_index", "true_state", "pred_state", "true_dur", "pred_dur", "will_prob", "valid_1step")}

    e = _ep(main_name)
    t = e["t_index"]
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax = axes[0]
    ts, ps = e["true_state"][:, key_j].astype(float), e["pred_state"][:, key_j].astype(float)
    ax.fill_between(t, 0, ts, step="post", color="0.75", label="True bottleneck")
    ax.step(t, ps * 0.9, where="post", color="tab:red", lw=1.4, label="Predicted bottleneck")
    wrong = ts != ps
    ax.scatter(t[wrong], np.full(wrong.sum(), 1.05), marker="x", color="black", s=14, label="Wrong")
    ax.set_ylim(-0.05, 1.15)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["normal", "bottleneck"])
    ax.legend(loc="upper right", fontsize=8, ncol=3)
    ax.set_title("next-window bottleneck state")
    ax = axes[1]
    ax.plot(t, e["true_dur"][:, key_j], color="black", lw=1.6, label="True")
    ax.plot(t, e["pred_dur"][:, key_j], color="tab:red", lw=1.2, ls="--", label="BNPDFormer")
    ax.set_ylabel("bottleneck duration in next 20 min (min)")
    ax.set_xlabel("window index (60 s)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")
    fig.suptitle(f"{key_dev}  |  {key_ep}")
    fig.tight_layout()
    fig.savefig(fig_dir / "key_device_main.png", dpi=200)
    plt.close(fig)

    from matplotlib.patches import Patch

    devs = [rids.index(d) for d in m0["devices"]]
    cmap = ListedColormap(["#f2f2f2", "#2ca02c", "#ff7f0e", "#d62728"])

    def _short(rid: str) -> str:
        head, _, ws = rid.partition("_ws")
        return head[:26] + (f"_ws{ws}" if ws else "")

    def _state_heatmap(name: str, tag: str) -> None:
        en = _ep(name, busy_ep)
        tt = en["t_index"]
        code = np.zeros((len(devs), len(tt)), dtype=int)
        for i, j in enumerate(devs):
            a = en["true_state"][:, j]
            b = en["pred_state"][:, j]
            code[i] = np.select([a & b, ~a & b, a & ~b], [1, 2, 3], default=0)
        tp, fp, fn = int((code == 1).sum()), int((code == 2).sum()), int((code == 3).sum())
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        fig, ax = plt.subplots(figsize=(11, 0.35 * len(devs) + 1.6))
        ax.imshow(code, aspect="auto", cmap=cmap, vmin=0, vmax=3, interpolation="nearest",
                  extent=[tt[0] - 0.5, tt[-1] + 0.5, len(devs) - 0.5, -0.5])
        ax.set_yticks(range(len(devs)))
        ax.set_yticklabels([_short(rids[j]) for j in devs], fontsize=7)
        ax.set_xlabel("window index (60 s)")
        ax.legend(
            handles=[Patch(color=c, label=l) for c, l in zip(cmap.colors, ["TN", "TP", "FP", "FN"])],
            loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=4, fontsize=8, frameon=False,
        )
        ax.set_title(
            f"{_label(name)} | {busy_ep} | next-window bottleneck state, F1={f1:.3f}", fontsize=9, pad=22
        )
        fig.tight_layout()
        fig.savefig(fig_dir / f"all_devices_state_{tag}.png", dpi=200)
        plt.close(fig)

    _state_heatmap(main_name, "main")
    for n in all_m:
        if n.endswith(("nograph_start5_seed42", "entity_noinfo_start5_min8_cold_ep50_seed42")):
            _state_heatmap(n, _label(n))

    families = {
        "start": [n for n in all_m if "12_3_start" in n],
        "struct": [main_name] + [n for n in all_m if n.startswith("ablation_") and "start5" in n],
        "entity": [main_name] + [n for n in all_m if "entity_" in n and "start5" in n],
    }
    for fam, names in families.items():
        names = [n for n in dict.fromkeys(names) if n in all_m]
        if len(names) < 2:
            continue
        fig, ax = plt.subplots(figsize=(10, 3.8))
        ax.plot(t, e["true_dur"][:, key_j], color="black", lw=1.8, label="True")
        for n in names:
            en = _ep(n)
            ax.plot(en["t_index"], en["pred_dur"][:, key_j], lw=1.0, ls="--", label=_label(n))
        ax.set_ylabel("bottleneck duration (min)")
        ax.set_xlabel("window index (60 s)")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=7, ncol=2)
        ax.set_title(f"{key_dev} | {key_ep} ({fam})", fontsize=9)
        fig.tight_layout()
        fig.savefig(fig_dir / f"key_device_{fam}.png", dpi=200)
        plt.close(fig)

    arms = {
        "Full": "dense_i1_12_3_start{s}_min8_cause4_opt_main_seed42",
        "NoGraph": "ablation_nograph_start{s}_seed42",
        "NoGroup": "ablation_nogroup_start{s}_seed42",
        "NoCross": "dense_i1_entity_nocross_start{s}_min8_cold_ep50_seed42",
        "NoInfo": "dense_i1_entity_noinfo_start{s}_min8_cold_ep50_seed42",
        "MachineOnly": "dense_i1_entity_machineonly_start{s}_min8_cold_ep50_seed42",
    }
    starts = (5, 10, 15)
    arms = {a: p for a, p in arms.items() if any(p.format(s=s) in all_m for s in starts)}
    x = np.arange(len(arms))
    w = 0.26
    panels = (("will15_f1", "will15 F1"), ("dur_mae_union", "duration MAE (min)"), ("dur_rmse_union", "duration RMSE (min)"))
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    for ax, (key, ylab) in zip(axes, panels):
        for i, s in enumerate(starts):
            vals = [all_m.get(p.format(s=s), {}).get(key, np.nan) for p in arms.values()]
            ax.bar(x + (i - 1) * w, vals, w, label=f"start≤{s}")
        ax.set_xticks(x)
        ax.set_xticklabels(list(arms), fontsize=8)
        ax.set_ylabel(ylab)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylim(0, 1)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "overview_bars.png", dpi=200)
    plt.close(fig)

    (out_root / "selection.json").write_text(
        json.dumps(
            {"main_run": main_name, "key_device": key_dev, "key_episode": key_ep, "heatmap_episode": busy_ep},
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[bn-eval] summary -> {out_root / 'summary.md'}  key_device={key_dev} episode={key_ep}")


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
