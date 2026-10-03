"""Metrics shared by baseline blocked/starved reconstruction evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np


CHANNELS = ("blocked", "starved")


def mae_rmse(error: np.ndarray) -> tuple[float, float]:
    error = np.asarray(error, dtype=np.float64)
    if error.size == 0:
        return float("nan"), float("nan")
    return float(np.abs(error).mean()), float(np.sqrt(np.square(error).mean()))


def r2(y: np.ndarray, prediction: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    if y.size < 2:
        return float("nan")
    total = float(np.square(y - y.mean()).sum())
    if total <= 1.0e-9:
        return float("nan")
    return float(1.0 - np.square(y - prediction).sum() / total)


def _masked_error(
    prediction: np.ndarray,
    truth: np.ndarray,
    valid: np.ndarray,
    node_mask: np.ndarray,
) -> np.ndarray:
    # prediction/truth: [sample, step, node, channel]
    mask = valid[:, :, None, None] & node_mask[None, None, :, None]
    return (prediction - truth)[mask]


def evaluate_reconstruction(
    prediction: np.ndarray,
    truth: np.ndarray,
    valid: np.ndarray,
    persistence: np.ndarray,
    *,
    resource_ids: list[str],
    resource_types: list[str],
    run: str,
    checkpoint_epoch: int | None = None,
    window_size_s: float = 60.0,
    episode: np.ndarray | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Compute the dev_tyx recon protocol on baseline predictions.

    Arrays use the same layout as ``eval_recon_bs.py``: prediction/truth are
    ``[sample, horizon, device, channel]``, valid is ``[sample, horizon]``,
    and persistence is ``[sample, device, channel]``.  Values are raw seconds
    per window and are expected to have been clipped to ``[0, window_size_s]``
    by the caller.
    """
    prediction = np.asarray(prediction, dtype=np.float32)
    truth = np.asarray(truth, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    persistence = np.asarray(persistence, dtype=np.float32)
    if prediction.shape != truth.shape or prediction.ndim != 4:
        raise ValueError("prediction and truth must share [sample, step, node, channel] shape")
    if valid.shape != prediction.shape[:2]:
        raise ValueError("valid must have [sample, step] shape")
    if persistence.shape != (prediction.shape[0], prediction.shape[2], prediction.shape[3]):
        raise ValueError("persistence must have [sample, node, channel] shape")
    if prediction.shape[-1] != len(CHANNELS):
        raise ValueError("reconstruction metrics require blocked and starved channels")
    if len(resource_ids) != prediction.shape[2] or len(resource_types) != prediction.shape[2]:
        raise ValueError("resource metadata does not match prediction node count")

    groups = {
        "all": np.ones(prediction.shape[2], dtype=bool),
        "machine": np.asarray([str(kind) == "machine" for kind in resource_types]),
    }
    result: dict[str, Any] = {
        "run": run,
        "window_size_s": float(window_size_s),
        "horizon_windows": int(prediction.shape[1]),
        "n_test_windows": int(prediction.shape[0]),
        "devices": list(resource_ids),
    }
    if checkpoint_epoch is not None:
        result["ckpt_epoch"] = int(checkpoint_epoch)
    if episode is not None:
        result["n_test_episodes"] = int(len(set(str(x) for x in episode.tolist())))

    first_valid = valid[:, 0]
    for channel_index, channel in enumerate(CHANNELS):
        p = prediction[..., channel_index]
        y = truth[..., channel_index]
        persist = persistence[..., channel_index]
        for group_name, node_mask in groups.items():
            one_mask = first_valid[:, None] & node_mask[None, :]
            horizon_mask = valid[:, :, None] & node_mask[None, None, :]
            one_error = (p[:, 0] - y[:, 0])[one_mask]
            horizon_error = (p - y)[horizon_mask]
            persist_error = (persist - y[:, 0])[one_mask]
            result[f"{channel}_{group_name}_mae_1step"], result[
                f"{channel}_{group_name}_rmse_1step"
            ] = mae_rmse(one_error)
            result[f"{channel}_{group_name}_mae_horizon"], result[
                f"{channel}_{group_name}_rmse_horizon"
            ] = mae_rmse(horizon_error)
            result[f"{channel}_{group_name}_mae_persist"], result[
                f"{channel}_{group_name}_rmse_persist"
            ] = mae_rmse(persist_error)
            y_one = y[:, 0][one_mask]
            p_one = p[:, 0][one_mask]
            active = y_one > 0.0
            result[f"{channel}_{group_name}_mae_1step_active"], result[
                f"{channel}_{group_name}_rmse_1step_active"
            ] = mae_rmse((p_one - y_one)[active])
            result[f"{channel}_{group_name}_active_frac"] = (
                float(active.mean()) if active.size else float("nan")
            )
            result[f"{channel}_{group_name}_r2_1step"] = r2(y_one, p_one)
        result[f"{channel}_all_mae_by_step"] = [
            mae_rmse((p[:, step] - y[:, step])[first_valid & valid[:, step]][:, :])[0]
            for step in range(prediction.shape[1])
        ]
        result[f"{channel}_all_rmse_by_step"] = [
            mae_rmse((p[:, step] - y[:, step])[first_valid & valid[:, step]][:, :])[1]
            for step in range(prediction.shape[1])
        ]

    per_device: list[dict[str, Any]] = []
    for node_index, (device, kind) in enumerate(zip(resource_ids, resource_types)):
        row: dict[str, Any] = {"device": device, "type": kind}
        for channel_index, channel in enumerate(CHANNELS):
            y_one = truth[first_valid, 0, node_index, channel_index]
            p_one = prediction[first_valid, 0, node_index, channel_index]
            row[f"{channel}_mae_1step"], row[f"{channel}_rmse_1step"] = mae_rmse(p_one - y_one)
            row[f"{channel}_r2_1step"] = r2(y_one, p_one)
            row[f"{channel}_true_mean"] = float(y_one.mean()) if y_one.size else float("nan")
            row[f"{channel}_true_std"] = float(y_one.std()) if y_one.size else float("nan")
            row[f"{channel}_mae_horizon"], row[f"{channel}_rmse_horizon"] = mae_rmse(
                (prediction[:, :, node_index, channel_index] - truth[:, :, node_index, channel_index])[valid]
            )
        per_device.append(row)
    result["per_device"] = per_device
    return result, per_device
