#!/usr/bin/env python3
"""Train/evaluate a blocked-starved readout on frozen B3-B5 encoders.

The official B4/B5 event checkpoints do not expose the dev_tyx recon head.
This tool keeps those checkpoints frozen and trains only a small, separately
recorded future-window readout on the train split.  It then evaluates the
readout on the untouched test split with the same protocol as
``factory_bn.eval_recon_bs``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from factory_baselines.dataset import FactoryBaselineTensorDataset, load_shared_dataset
from factory_baselines.recon_metrics import CHANNELS, evaluate_reconstruction
from factory_baselines.torch_trainer import (
    _loaders,
    _model_inputs,
    _move_batch,
    _resolve_device,
    load_checkpoint,
)


class ReconReadout(nn.Module):
    """Shared node readout with a learned future-step embedding."""

    def __init__(self, hidden_dim: int, horizon: int, width: int = 128) -> None:
        super().__init__()
        self.horizon = int(horizon)
        self.time_embedding = nn.Embedding(self.horizon, hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, width),
            nn.GELU(),
            nn.Linear(width, len(CHANNELS)),
        )

    def forward(self, node_hidden: torch.Tensor) -> torch.Tensor:
        batch, nodes, hidden = node_hidden.shape
        steps = torch.arange(self.horizon, device=node_hidden.device)
        time = self.time_embedding(steps)[None, :, None, :].expand(
            batch, -1, nodes, -1
        )
        hidden = node_hidden[:, None].expand(-1, self.horizon, -1, -1)
        return self.head(torch.cat((hidden, time), dim=-1))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _make_loader(payload: dict[str, Any], split: str, batch_size: int) -> DataLoader:
    if "recon_target" not in payload or "recon_valid" not in payload:
        raise ValueError(
            "dataset.pt has no reconstruction targets; rebuild the shared baseline dataset "
            "with the current dev_xwt dataset builder"
        )
    indices = payload["split_indices"][split].tolist()
    return DataLoader(
        FactoryBaselineTensorDataset(payload, indices),
        batch_size=batch_size,
        shuffle=split == "train",
        num_workers=0,
    )


@torch.no_grad()
def _predict_hidden(
    encoder: nn.Module,
    readout: ReconReadout,
    loader: DataLoader,
    device: torch.device,
    *,
    collect_targets: bool,
) -> tuple[float, dict[str, np.ndarray]]:
    encoder.eval()
    readout.eval()
    total = 0.0
    count = 0
    arrays: dict[str, list[np.ndarray]] = {"prediction": [], "truth": [], "valid": [], "sample_index": []}
    for cpu_batch in loader:
        batch = _move_batch(cpu_batch, device)
        output = encoder(**_model_inputs(batch, encoder))
        prediction = readout(output["node_hidden"])
        truth = batch["recon_target"].float()
        valid = batch["recon_valid"].bool()[:, :, None, None]
        node = batch["occ_node_mask"].bool()[:, None, :, None]
        mask = valid & node
        error = torch.nn.functional.smooth_l1_loss(prediction, truth, reduction="none")
        denom = mask.expand_as(error).sum().clamp_min(1.0)
        total += float((error * mask).sum().item())
        count += int(denom.item())
        if collect_targets:
            arrays["prediction"].append(prediction.detach().cpu().numpy())
            arrays["truth"].append(truth.detach().cpu().numpy())
            arrays["valid"].append(batch["recon_valid"].detach().cpu().numpy().astype(bool))
            arrays["sample_index"].append(batch["sample_index"].detach().cpu().numpy())
    return total / max(count, 1), {
        key: np.concatenate(values, axis=0) for key, values in arrays.items()
    } if collect_targets else {}


def _train_probe(
    encoder: nn.Module,
    loaders: dict[str, DataLoader],
    device: torch.device,
    *,
    horizon: int,
    epochs: int,
    patience: int,
    learning_rate: float,
) -> tuple[ReconReadout, dict[str, Any]]:
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    encoder.eval()
    hidden_dim = int(encoder.config.node_hidden if hasattr(encoder.config, "node_hidden") else encoder.config.gru_hidden)
    probe = ReconReadout(hidden_dim, horizon).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=learning_rate, weight_decay=1e-3)
    best_state: dict[str, torch.Tensor] | None = None
    best_val = float("inf")
    best_epoch = 0
    stale = 0
    history = []
    for epoch in range(1, epochs + 1):
        probe.train()
        train_loss, train_count = 0.0, 0
        for cpu_batch in loaders["train"]:
            batch = _move_batch(cpu_batch, device)
            with torch.no_grad():
                output = encoder(**_model_inputs(batch, encoder))
            prediction = probe(output["node_hidden"])
            truth = batch["recon_target"].float()
            mask = (
                batch["recon_valid"].bool()[:, :, None, None]
                & batch["occ_node_mask"].bool()[:, None, :, None]
            ).expand_as(prediction)
            # Match the main task's channel emphasis while retaining the raw
            # seconds-per-window target used for final MAE/RMSE.
            channel_weight = prediction.new_tensor([2.0, 1.5]).view(1, 1, 1, 2)
            error = torch.nn.functional.smooth_l1_loss(prediction, truth, reduction="none")
            denom = (mask * channel_weight).sum().clamp_min(1.0)
            loss = (error * mask * channel_weight).sum() / denom
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(probe.parameters(), 1.0)
            optimizer.step()
            train_loss += float(loss.item()) * int(mask[..., 0].sum().item())
            train_count += int(mask[..., 0].sum().item())
        val_loss, _ = _predict_hidden(
            encoder, probe, loaders["validation"], device, collect_targets=False
        )
        row = {"epoch": epoch, "train_loss": train_loss / max(train_count, 1), "validation_loss": val_loss}
        history.append(row)
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in probe.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is None:
        raise RuntimeError("reconstruction readout never produced a validation checkpoint")
    probe.load_state_dict(best_state)
    return probe, {"best_epoch": best_epoch, "best_validation_loss": best_val, "history": history}


def _metadata(dataset_dir: Path) -> tuple[list[str], list[str], dict[str, Any]]:
    manifest = _load_json(dataset_dir / "dataset_manifest.json")
    with (dataset_dir / "node_catalog.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    rows.sort(key=lambda row: int(row["node_index"]))
    return [row["resource_id"] for row in rows], [row["resource_type"] for row in rows], manifest


def evaluate(args: argparse.Namespace) -> None:
    dataset_dir = args.dataset_dir.resolve()
    checkpoint = args.checkpoint.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    payload, manifest = load_shared_dataset(dataset_dir)
    resource_ids, resource_types, manifest = _metadata(dataset_dir)
    horizon = int(payload["recon_target"].shape[1])
    device = _resolve_device(args.device)
    encoder, checkpoint_data = load_checkpoint(checkpoint, device)
    if checkpoint_data["model_kind"] not in {"b4_gcn_gru", "b5_gat_gru", "b3_lstm"}:
        raise ValueError("The recon readout requires a neural B3/B4/B5 checkpoint")
    loaders = {
        split: _make_loader(payload, split, args.batch_size)
        for split in ("train", "validation", "test")
    }
    probe, training = _train_probe(
        encoder,
        loaders,
        device,
        horizon=horizon,
        epochs=args.epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
    )
    _, arrays = _predict_hidden(encoder, probe, loaders["test"], device, collect_targets=True)
    normalization = _load_json(dataset_dir / "normalization.json")
    mean = np.asarray(normalization["feature_mean"], dtype=np.float32)
    std = np.asarray(normalization["feature_std"], dtype=np.float32)
    channel_indices = [
        manifest["feature_names"].index("blocked_time_s"),
        manifest["feature_names"].index("starved_time_s"),
    ]
    # x starts with the normalized continuous features.  Copy the last
    # observed blocked/starved window as the persistence sanity reference.
    test_indices = payload["split_indices"]["test"].numpy()
    last = payload["x"][test_indices, -1, :, channel_indices].numpy()
    persistence = last * std[channel_indices] + mean[channel_indices]
    prediction = np.clip(arrays["prediction"], 0.0, float(payload["window_size_s"]))
    truth = np.clip(arrays["truth"], 0.0, float(payload["window_size_s"]))
    valid = arrays["valid"]
    # Match dev_tyx exactly: report the union of occupancy-supervised nodes
    # (the 15 devices), not labor/storage nodes present in the graph tensor.
    device_mask = payload["occ_node_mask"][test_indices].bool().any(dim=0).numpy()
    device_indices = np.flatnonzero(device_mask)
    prediction = prediction[:, :, device_indices]
    truth = truth[:, :, device_indices]
    persistence = persistence[:, device_indices]
    resource_ids = [resource_ids[index] for index in device_indices]
    resource_types = [resource_types[index] for index in device_indices]
    sample_rows = {int(row["sample_index"]): row for row in csv.DictReader((dataset_dir / "model_sample_index.csv").open(newline="", encoding="utf-8"))}
    episodes = np.asarray([sample_rows[int(i)]["group_id"] for i in arrays["sample_index"]])
    metrics, per_device = evaluate_reconstruction(
        prediction,
        truth,
        valid,
        persistence,
        resource_ids=resource_ids,
        resource_types=resource_types,
        run=checkpoint.parent.name,
        checkpoint_epoch=int(checkpoint_data.get("epoch") or 0),
        window_size_s=float(payload["window_size_s"]),
        episode=episodes,
    )
    metrics.update({
        "readout_kind": "frozen_encoder_recon_readout",
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_sha256": _sha256(checkpoint),
        "dataset_manifest_sha256": _sha256(dataset_dir / "dataset_manifest.json"),
        "readout_training": training,
        "protocol": {
            "channels": list(CHANNELS),
            "units": "seconds_per_window",
            "validity": "remain_len_until_episode_done",
            "selection": "validation smooth_l1 readout loss; event checkpoint remains frozen",
        },
    })
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False, allow_nan=True) + "\n", encoding="utf-8"
    )
    with (output_dir / "per_device.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(per_device[0]))
        writer.writeheader()
        writer.writerows(per_device)
    np.savez_compressed(
        output_dir / "series.npz",
        pred=prediction.astype(np.float32),
        true=truth.astype(np.float32),
        valid=valid,
        persist=persistence.astype(np.float32),
        episode=episodes,
        sample_index=arrays["sample_index"],
        devices=np.asarray(resource_ids),
        types=np.asarray(resource_types),
        channels=np.asarray(CHANNELS),
    )
    torch.save(
        {
            "probe_state_dict": probe.state_dict(),
            "source_checkpoint": str(checkpoint),
            "source_checkpoint_sha256": _sha256(checkpoint),
            "dataset_manifest_sha256": _sha256(dataset_dir / "dataset_manifest.json"),
            "horizon_windows": horizon,
            "hidden_dim": next(iter(probe.parameters())).shape[-1],
            "training": training,
        },
        output_dir / "readout.pt",
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "source_checkpoint": str(checkpoint),
                "blocked_mae_1step": metrics["blocked_all_mae_1step"],
                "blocked_rmse_1step": metrics["blocked_all_rmse_1step"],
                "starved_mae_1step": metrics["starved_all_mae_1step"],
                "starved_rmse_1step": metrics["starved_all_rmse_1step"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


def summarize(args: argparse.Namespace) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = args.input_root.resolve()
    runs = sorted(path.parent for path in root.glob("*/metrics.json"))
    if not runs:
        raise SystemExit(f"No baseline recon outputs under {root}")
    rows = []
    for run in runs:
        metrics = _load_json(run / "metrics.json")
        row = {"run": run.name}
        for channel in CHANNELS:
            for metric in ("mae_1step", "rmse_1step", "mae_horizon", "rmse_horizon", "mae_1step_active", "rmse_1step_active", "r2_1step"):
                row[f"{channel}_{metric}"] = metrics[f"{channel}_all_{metric}"]
        row["starved_machine_r2_1step"] = metrics["starved_machine_r2_1step"]
        rows.append(row)
    with (root / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Baseline blocked/starved reconstruction summary",
        "",
        "The readout is trained on the train split over a frozen B3/B4/B5 encoder; all values are seconds per 60-second window.",
        "",
        "| run | Blocked MAE/RMSE 1step | Starved MAE/RMSE 1step | Blocked MAE/RMSE horizon | Starved MAE/RMSE horizon | Blocked active MAE | Starved active MAE | Starved R² | Machine Starved R² |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['run']} | {row['blocked_mae_1step']:.2f}/{row['blocked_rmse_1step']:.2f} "
            f"| {row['starved_mae_1step']:.2f}/{row['starved_rmse_1step']:.2f} "
            f"| {row['blocked_mae_horizon']:.2f}/{row['blocked_rmse_horizon']:.2f} "
            f"| {row['starved_mae_horizon']:.2f}/{row['starved_rmse_horizon']:.2f} "
            f"| {row['blocked_mae_1step_active']:.2f} | {row['starved_mae_1step_active']:.2f} "
            f"| {row['starved_r2_1step']:.3f} | {row['starved_machine_r2_1step']:.3f} |"
        )
    (root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    figure_dir = root / "figures"
    figure_dir.mkdir(exist_ok=True)
    for run in runs:
        metrics = _load_json(run / "metrics.json")
        series = dict(np.load(run / "series.npz", allow_pickle=False))
        device_index = 0
        if "num01_weldingRobot_ws0" in [str(x) for x in series["devices"]]:
            device_index = [str(x) for x in series["devices"]].index("num01_weldingRobot_ws0")
        for channel_index, channel in enumerate(CHANNELS):
            mask = series["valid"][:, 0]
            truth = series["true"][mask, 0, device_index, channel_index]
            prediction = series["pred"][mask, 0, device_index, channel_index]
            plt.figure(figsize=(10, 3))
            plt.plot(truth, label="true", linewidth=1.0)
            plt.plot(prediction, label="baseline readout", linewidth=1.0)
            plt.title(f"{run.name}: {channel} 1-step, key device")
            plt.xlabel("test window")
            plt.ylabel("seconds / 60 s window")
            plt.legend()
            plt.tight_layout()
            plt.savefig(figure_dir / f"{run.name}_{channel}.png", dpi=160)
            plt.close()
    plt.figure(figsize=(8, 4))
    for run in runs:
        metrics = _load_json(run / "metrics.json")
        values = metrics["starved_all_mae_by_step"]
        plt.plot(np.arange(1, len(values) + 1), values, marker=".", label=run.name)
    plt.xlabel("future window step")
    plt.ylabel("starved MAE (seconds / 60 s window)")
    plt.title("Baseline starved MAE by prediction step")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(figure_dir / "mae_by_step.png", dpi=160)
    plt.close()
    print(f"WROTE_BASELINE_RECON_SUMMARY {root}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    evaluate_parser = sub.add_parser("evaluate")
    evaluate_parser.add_argument("--dataset_dir", type=Path, required=True)
    evaluate_parser.add_argument("--checkpoint", type=Path, required=True)
    evaluate_parser.add_argument("--output_dir", type=Path, required=True)
    evaluate_parser.add_argument("--device", default="auto")
    evaluate_parser.add_argument("--batch_size", type=int, default=32)
    evaluate_parser.add_argument("--epochs", type=int, default=40)
    evaluate_parser.add_argument("--patience", type=int, default=8)
    evaluate_parser.add_argument("--learning_rate", type=float, default=3e-4)
    evaluate_parser.set_defaults(func=evaluate)
    summary_parser = sub.add_parser("summarize")
    summary_parser.add_argument("--input_root", type=Path, required=True)
    summary_parser.set_defaults(func=summarize)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
