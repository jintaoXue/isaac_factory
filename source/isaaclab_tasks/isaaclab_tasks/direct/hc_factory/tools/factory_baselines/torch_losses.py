"""Masked multi-task losses shared by B3-B5."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch.nn import functional as F


@dataclass
class MultiTaskLossConfig:
    lambda_occurrence: float = 1.0
    lambda_node: float = 1.0
    lambda_time_to_start: float = 1.0
    lambda_remain_score: float = 1.0
    lambda_remain_hot: float = 1.0
    lambda_remain_len: float = 0.3
    lambda_cause: float = 0.4
    hot_pos_weight: float = 32.0
    near_remain_windows: int = 60
    remain_loss_tau: float = 20.0
    prediction_horizon: float = 180.0

    def __post_init__(self) -> None:
        if self.prediction_horizon <= 0:
            raise ValueError("prediction_horizon must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "MultiTaskLossConfig":
        return cls(**values)


def _zero_from(outputs: dict[str, torch.Tensor]) -> torch.Tensor:
    return outputs["occurrence_logit"].sum() * 0.0


def compute_multitask_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    config: MultiTaskLossConfig,
    pos_weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    occurrence = F.binary_cross_entropy_with_logits(
        outputs["occurrence_logit"],
        batch["y_occurrence"].float(),
        pos_weight=pos_weight,
    )
    positive = batch["positive_mask"].bool()
    zero = _zero_from(outputs)

    remain_mask = batch["remain_mask"].float()
    steps = torch.arange(remain_mask.shape[1], device=remain_mask.device).float()
    step_weight = (
        torch.exp(-steps / max(config.remain_loss_tau, 1.0))
        * (steps < float(max(config.near_remain_windows, 1))).float()
    )
    step_weight = remain_mask * step_weight[None]
    score_weight = step_weight[:, :, None, None]
    score_error = F.smooth_l1_loss(
        outputs["remain_score"], batch["y_score"].float(), reduction="none"
    )
    remain_score = (score_error * score_weight).sum() / (
        score_weight.sum() * outputs["remain_score"].shape[2]
    ).clamp_min(1.0)
    hot_error = F.binary_cross_entropy_with_logits(
        outputs["remain_hot_logit"], batch["y_hot"].float(), reduction="none"
    )
    hot_class_weight = 1.0 + (config.hot_pos_weight - 1.0) * batch["y_hot"].float()
    hot_weight = step_weight[:, :, None]
    remain_hot = (hot_error * hot_class_weight * hot_weight).sum() / (
        hot_weight.sum() * outputs["remain_hot_logit"].shape[2]
    ).clamp_min(1.0)
    remain_len = F.smooth_l1_loss(
        torch.log1p(outputs["remain_len"]),
        torch.log1p(batch["target_remain_len"].float()),
    )
    valid_cause = batch["y_cause"] >= 0
    cause = (
        F.cross_entropy(
            outputs["cause_logits"][valid_cause], batch["y_cause"][valid_cause]
        )
        if valid_cause.any()
        else zero
    )

    if positive.any():
        node = F.cross_entropy(
            outputs["node_logits"][positive], batch["y_node"][positive]
        )
        time_to_start = F.smooth_l1_loss(
            outputs["time_to_start"][positive] / config.prediction_horizon,
            batch["y_time_to_start"][positive].float() / config.prediction_horizon,
        )
    else:
        node = zero
        time_to_start = zero

    components = {
        "occurrence": occurrence,
        "node": node,
        "time_to_start": time_to_start,
        "remain_score": remain_score,
        "remain_hot": remain_hot,
        "remain_len": remain_len,
        "cause": cause,
    }
    total = (
        config.lambda_occurrence * occurrence
        + config.lambda_node * node
        + config.lambda_time_to_start * time_to_start
        + config.lambda_remain_score * remain_score
        + config.lambda_remain_hot * remain_hot
        + config.lambda_remain_len * remain_len
        + config.lambda_cause * cause
    )
    return total, components
