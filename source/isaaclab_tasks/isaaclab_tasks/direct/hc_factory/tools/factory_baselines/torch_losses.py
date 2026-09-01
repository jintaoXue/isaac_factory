"""Masked multi-task losses shared by B3-B5."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch.nn import functional as F

from factory_bn_shared.causes import cause_ignore_ids


@dataclass
class MultiTaskLossConfig:
    lambda_remain_score: float = 0.0
    lambda_remain_hot: float = 0.5
    lambda_remain_len: float = 0.4
    lambda_cause: float = 0.1
    lambda_event_will: float = 2.5
    lambda_event_start: float = 1.5
    lambda_event_duration: float = 1.0
    hot_pos_weight: float = 4.0
    event_will_pos_weight: float = 3.0
    event_will_fp_weight: float = 2.0
    event_will_upcoming_pos_weight: float = 4.0
    near_remain_windows: int = 15
    remain_loss_tau: float = 40.0
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
    return outputs["remain_hot_logit"].sum() * 0.0


def compute_multitask_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    config: MultiTaskLossConfig,
    pos_weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    del pos_weight
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
    for cause_id in cause_ignore_ids():
        valid_cause = valid_cause & (batch["y_cause"] != int(cause_id))
    cause = (
        F.cross_entropy(
            outputs["cause_logits"][valid_cause], batch["y_cause"][valid_cause]
        )
        if valid_cause.any()
        else zero
    )

    occ_mask = batch["occ_node_mask"].bool()
    event_will_target = batch["event_will"].float()
    hist_hot = batch["hist_last_hot"].bool()
    upcoming = (event_will_target > 0.5) & ~hist_hot & occ_mask
    ongoing = (event_will_target > 0.5) & hist_hot & occ_mask
    event_weight = torch.where(
        event_will_target > 0.5,
        torch.full_like(event_will_target, config.event_will_pos_weight),
        torch.full_like(event_will_target, config.event_will_fp_weight),
    )
    event_weight = torch.where(
        upcoming,
        torch.full_like(event_weight, config.event_will_upcoming_pos_weight),
        event_weight,
    )
    event_will_raw = F.binary_cross_entropy_with_logits(
        outputs["event_will_logit"], event_will_target, reduction="none"
    )
    event_will = (event_will_raw * event_weight * occ_mask.float()).sum() / (
        event_weight * occ_mask.float()
    ).sum().clamp_min(1.0)
    event_start = (
        F.cross_entropy(
            outputs["event_start_logit"][upcoming],
            batch["event_start"][upcoming].long(),
        )
        if upcoming.any()
        else zero
    )
    positive_event = (upcoming | ongoing) & occ_mask
    event_duration = (
        F.smooth_l1_loss(
            outputs["event_duration"][positive_event],
            batch["event_duration"][positive_event].float(),
        )
        if positive_event.any()
        else zero
    )

    components = {
        "remain_score": remain_score,
        "remain_hot": remain_hot,
        "remain_len": remain_len,
        "cause": cause,
        "event_will": event_will,
        "event_start": event_start,
        "event_duration": event_duration,
    }
    total = (
        config.lambda_remain_score * remain_score
        + config.lambda_remain_hot * remain_hot
        + config.lambda_remain_len * remain_len
        + config.lambda_cause * cause
        + config.lambda_event_will * event_will
        + config.lambda_event_start * event_start
        + config.lambda_event_duration * event_duration
    )
    return total, components
