"""Masked multi-task losses for BSTAN training."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch.nn import functional as F


@dataclass
class BstanLossConfig:
    lambda_occurrence: float = 1.0
    lambda_node: float = 1.0
    lambda_type: float = 1.0
    lambda_time_to_start: float = 1.0
    lambda_duration: float = 1.0
    lambda_severity: float = 1.0
    prediction_horizon: float = 120.0

    def __post_init__(self) -> None:
        if self.prediction_horizon <= 0:
            raise ValueError("prediction_horizon must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "BstanLossConfig":
        return cls(**values)


def _zero_from(outputs: dict[str, torch.Tensor]) -> torch.Tensor:
    return outputs["occurrence_logit"].sum() * 0.0


def compute_multitask_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    config: BstanLossConfig,
    pos_weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    occurrence = F.binary_cross_entropy_with_logits(
        outputs["occurrence_logit"],
        batch["y_occurrence"].float(),
        pos_weight=pos_weight,
    )
    positive = batch["positive_mask"].bool()
    duration_mask = batch["duration_mask"].bool() & positive
    zero = _zero_from(outputs)

    if positive.any():
        node = F.cross_entropy(
            outputs["node_logits"][positive], batch["y_node"][positive]
        )
        bottleneck_type = F.cross_entropy(
            outputs["type_logits"][positive], batch["y_type"][positive]
        )
        time_to_start = F.smooth_l1_loss(
            outputs["time_to_start"][positive] / config.prediction_horizon,
            batch["y_time_to_start"][positive].float() / config.prediction_horizon,
        )
        severity = F.smooth_l1_loss(
            outputs["severity"][positive], batch["y_severity"][positive].float()
        )
    else:
        node = zero
        bottleneck_type = zero
        time_to_start = zero
        severity = zero

    if duration_mask.any():
        duration = F.smooth_l1_loss(
            outputs["duration"][duration_mask] / config.prediction_horizon,
            batch["y_duration"][duration_mask].float() / config.prediction_horizon,
        )
    else:
        duration = zero

    components = {
        "occurrence": occurrence,
        "node": node,
        "type": bottleneck_type,
        "time_to_start": time_to_start,
        "duration": duration,
        "severity": severity,
    }
    total = (
        config.lambda_occurrence * occurrence
        + config.lambda_node * node
        + config.lambda_type * bottleneck_type
        + config.lambda_time_to_start * time_to_start
        + config.lambda_duration * duration
        + config.lambda_severity * severity
    )
    return total, components
