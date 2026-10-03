#!/usr/bin/env python3
"""Summarize frozen baseline bottleneck-event test metrics and make overview charts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


FIELDS = (
    "model", "max_start", "selected_threshold", "test_will15_precision",
    "test_will15_recall", "test_will15_f1", "test_state_acc_1step",
    "test_state_f1_1step", "test_state_f1_horizon", "test_state_vs_event_acc_1step",
    "test_state_vs_event_acc_horizon", "test_dur_mae_cells", "test_dur_rmse_cells",
    "test_dur_mae_union",
    "test_dur_rmse_union", "test_dur_mae_tp", "test_dur_rmse_tp",
    "test_start_mae_union", "test_start_rmse_union", "test_start_mae_tp",
    "test_start_rmse_tp", "test_duration_union_support", "test_duration_tp_support",
)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") != "frozen_baseline_test_evaluation_completed":
        raise ValueError(f"Unexpected report status: {path}")
    return value


def _row(task: dict[str, Any]) -> dict[str, Any]:
    return {field: task.get(field) for field in FIELDS}


def _write_summary(report: dict[str, Any], output: Path) -> list[dict[str, Any]]:
    rows = [_row(task) for task in report["tasks"]]
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(FIELDS))
        writer.writeheader()
        writer.writerows(rows)

    best = max(rows, key=lambda row: float(row["test_will15_f1"]))
    lines = [
        "# Frozen baseline bottleneck-event test summary",
        "",
        "This summary follows the latest `dev_tyx` bottleneck event protocol.",
        "Durations and starts are measured in windows; with the current 60-second window they are numerically minutes.",
        "Union errors include missed and false-alarm bottlenecks; TP errors include correctly detected bottlenecks only.",
        "",
        "| model | cap | threshold | will15 P | will15 R | will15 F1 | state F1 1step | state F1 horizon | dur MAE union | dur RMSE union | dur MAE TP | dur RMSE TP | start MAE TP | start RMSE TP |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {model} | {max_start} | {selected_threshold:.2f} | {test_will15_precision:.3f} | "
            "{test_will15_recall:.3f} | {test_will15_f1:.3f} | {test_state_f1_1step:.3f} | "
            "{test_state_f1_horizon:.3f} | {test_dur_mae_union:.2f} | {test_dur_rmse_union:.2f} | "
            "{test_dur_mae_tp:.2f} | {test_dur_rmse_tp:.2f} | {test_start_mae_tp:.2f} | {test_start_rmse_tp:.2f} |".format(**row)
        )
    lines += [
        "",
        f"Best will15 F1 in this report: {best['model']} cap={best['max_start']} ({best['test_will15_f1']:.3f}).",
        "Per-device metrics for every task are retained under `task.test.bottleneck_per_device` in the JSON report.",
    ]
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rows


def _write_per_device(report: dict[str, Any], output: Path, best: dict[str, Any]) -> None:
    task = next(
        item for item in report["tasks"]
        if item["model"] == best["model"] and item["max_start"] == best["max_start"]
    )
    rows = task["test"].get("bottleneck_per_device", [])
    if not rows:
        return
    with (output / "per_device_best.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot(rows: list[dict[str, Any]], output: Path, artifacts_root: Path, best: dict[str, Any]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    labels = [f"{row['model']}\ncap{row['max_start']}" for row in rows]
    x = list(range(len(rows)))
    width = 0.25
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), constrained_layout=True)
    axes[0].bar([i - width for i in x], [row["test_will15_f1"] for row in rows], width, label="will15 F1")
    axes[0].bar(x, [row["test_state_f1_1step"] for row in rows], width, label="state F1 1step")
    axes[0].bar([i + width for i in x], [row["test_state_f1_horizon"] for row in rows], width, label="state F1 horizon")
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("F1")
    axes[0].set_title("Bottleneck event/state quality")
    axes[0].legend(fontsize=8)
    axes[1].bar([i - width / 2 for i in x], [row["test_dur_mae_union"] for row in rows], width, label="duration MAE union")
    axes[1].bar([i + width / 2 for i in x], [row["test_dur_rmse_union"] for row in rows], width, label="duration RMSE union")
    axes[1].set_ylabel("windows / minutes")
    axes[1].set_title("Duration error including misses and false alarms")
    axes[1].legend(fontsize=8)
    for axis in axes:
        axis.set_xticks(x, labels, rotation=45, ha="right")
        axis.grid(axis="y", alpha=0.25)
    fig.savefig(output / "overview_bars.png", dpi=200)
    plt.close(fig)

    series_path = artifacts_root / f"{best['model'].lower()}_cap{best['max_start']}" / "series.npz"
    if not series_path.is_file():
        return
    series = dict(__import__("numpy").load(series_path, allow_pickle=False))
    true_state = series["true_state"]
    pred_state = series["pred_state"]
    true_dur = series["true_dur"]
    pred_dur = series["pred_dur"]
    ids = [str(value) for value in series["resource_ids"]]
    key_index = int(series["true_will"].sum(axis=0).argmax())
    key_label = ids[key_index]
    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True, constrained_layout=True)
    axes[0].step(range(len(true_state)), true_state[:, key_index], where="post", label="true state")
    axes[0].step(range(len(pred_state)), pred_state[:, key_index], where="post", label="predicted state")
    axes[0].set_ylabel("bottleneck")
    axes[0].set_title(f"{best['model']} cap{best['max_start']} key device: {key_label}")
    axes[0].legend()
    axes[1].plot(true_dur[:, key_index], label="true duration")
    axes[1].plot(pred_dur[:, key_index], label="predicted duration")
    axes[1].set_xlabel("test sample index")
    axes[1].set_ylabel("windows / minutes")
    axes[1].legend()
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.savefig(output / "key_device_best.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True, constrained_layout=True)
    axes[0].imshow(true_state.T, aspect="auto", interpolation="nearest", cmap="Greys", vmin=0, vmax=1)
    axes[0].set_title("True next-window bottleneck state")
    axes[1].imshow(pred_state.T, aspect="auto", interpolation="nearest", cmap="Greys", vmin=0, vmax=1)
    axes[1].set_title("Predicted next-window bottleneck state")
    axes[1].set_xlabel("test sample index")
    for axis in axes:
        axis.set_ylabel("device index")
    fig.savefig(output / "all_devices_state_best.png", dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--artifacts_root", type=Path)
    args = parser.parse_args()
    report = _read(args.report.resolve())
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = _write_summary(report, output)
    best = max(rows, key=lambda row: float(row["test_will15_f1"]))
    _write_per_device(report, output, best)
    artifacts_root = (args.artifacts_root or args.report.with_name(args.report.stem + "_artifacts")).resolve()
    _plot(rows, output, artifacts_root, best)
    print(f"SUMMARY_COMPLETE {output}")


if __name__ == "__main__":
    main()
