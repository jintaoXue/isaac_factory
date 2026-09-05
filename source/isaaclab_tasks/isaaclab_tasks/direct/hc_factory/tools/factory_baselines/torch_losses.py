"""Masked multi-task losses shared by B3-B5."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any

import torch
from torch.nn import functional as F

from factory_bn_shared.causes import cause_ignore_ids
from factory_bn_shared.remain import gaussian_start_soft_labels


@dataclass
class MultiTaskLossConfig:
    lambda_remain_score: float = 0.0
    lambda_remain_hot: float = 0.5
    lambda_remain_dice: float = 0.25
    lambda_remain_iou: float = 0.25
    lambda_remain_len: float = 0.4
    lambda_cause: float = 0.1
    lambda_event_will: float = 2.5
    lambda_event_start: float = 1.5
    lambda_event_duration: float = 1.0
    hot_pos_weight: float = 4.0
    hot_pos_weight_by_type: dict[str, float] = field(
        default_factory=lambda: {
            "machine": 4.0,
            "workbench": 2.0,
            "gantry": 1.0,
            "agv": 4.0,
        }
    )
    hot_fp_weight_by_type: dict[str, float] = field(
        default_factory=lambda: {
            "workbench": 2.0,
            "gantry": 2.0,
            "agv": 2.0,
        }
    )
    type_balanced_occupancy: bool = True
    event_will_pos_weight: float = 3.0
    event_will_fp_weight: float = 2.0
    event_will_upcoming_pos_weight: float = 4.0
    event_will_ongoing_pos_weight: float = 3.0
    event_focal_gamma: float = 0.0
    event_start_sigma: float = 1.0
    near_remain_windows: int = 15
    remain_loss_tau: float = 40.0
    prediction_horizon: float = 180.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.event_focal_gamma) or self.event_focal_gamma < 0:
            raise ValueError("event_focal_gamma must be finite and non-negative")
        if self.prediction_horizon <= 0:
            raise ValueError("prediction_horizon must be positive")
        for name in (
            "lambda_remain_score",
            "lambda_remain_hot",
            "lambda_remain_dice",
            "lambda_remain_iou",
            "lambda_remain_len",
            "lambda_cause",
            "lambda_event_will",
            "lambda_event_start",
            "lambda_event_duration",
        ):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        for name in (
            "hot_pos_weight",
            "event_will_pos_weight",
            "event_will_fp_weight",
            "event_will_upcoming_pos_weight",
            "event_will_ongoing_pos_weight",
            "event_start_sigma",
            "remain_loss_tau",
        ):
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if self.near_remain_windows <= 0:
            raise ValueError("near_remain_windows must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "MultiTaskLossConfig":
        return cls(**values)


def _zero_from(outputs: dict[str, torch.Tensor]) -> torch.Tensor:
    return outputs["remain_hot_logit"].sum() * 0.0


def _event_binary_loss(logits: torch.Tensor, target: torch.Tensor, gamma: float) -> torch.Tensor:
    error = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    if gamma == 0.0:
        return error
    # exp(-BCE) is the probability assigned to the true binary class.
    return error * (-torch.expm1(-error)).pow(gamma)


def _soft_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1.0,
) -> torch.Tensor:
    prediction = torch.sigmoid(logits)
    intersection = (prediction * target * weight).sum()
    denominator = (prediction * weight).sum() + (target * weight).sum()
    return 1.0 - (2.0 * intersection + eps) / (denominator + eps)


def _soft_iou_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1.0,
) -> torch.Tensor:
    prediction = torch.sigmoid(logits)
    intersection = (prediction * target * weight).sum()
    union = (prediction * weight).sum() + (target * weight).sum() - intersection
    return 1.0 - (intersection + eps) / (union + eps)


def _occupancy_cell_weight(
    target: torch.Tensor,
    type_masks: dict[str, torch.Tensor],
    config: MultiTaskLossConfig,
) -> torch.Tensor:
    default = 1.0 + (config.hot_pos_weight - 1.0) * target
    weight = default
    tagged = torch.zeros_like(target)
    for name, raw_mask in type_masks.items():
        node_mask = raw_mask.to(device=target.device, dtype=target.dtype).reshape(-1)
        if node_mask.numel() != target.shape[-1] or float(node_mask.sum()) <= 0:
            continue
        mask = node_mask.view(1, 1, -1)
        positive_weight = float(
            config.hot_pos_weight_by_type.get(name, config.hot_pos_weight)
        )
        negative_weight = float(config.hot_fp_weight_by_type.get(name, 1.0))
        typed = target * positive_weight + (1.0 - target) * negative_weight
        weight = torch.where(mask > 0.5, typed, weight)
        tagged = torch.maximum(tagged, mask.expand_as(target))
    return torch.where(tagged > 0.5, weight, default)


def _type_balanced_occupancy_losses(
    logits: torch.Tensor,
    target: torch.Tensor,
    cell_mask: torch.Tensor,
    type_masks: dict[str, torch.Tensor],
    config: MultiTaskLossConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    zero = logits.sum() * 0.0
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    class_weight = _occupancy_cell_weight(target, type_masks, config)
    bce_parts = []
    dice_parts = []
    iou_parts = []
    for raw_mask in type_masks.values():
        node_mask = raw_mask.to(device=logits.device, dtype=logits.dtype).reshape(-1)
        if node_mask.numel() != logits.shape[-1] or float(node_mask.sum()) <= 0:
            continue
        typed_mask = cell_mask * node_mask.view(1, 1, -1)
        if float(typed_mask.sum()) <= 0:
            continue
        bce_parts.append(
            (bce * class_weight * typed_mask).sum() / typed_mask.sum().clamp_min(1.0)
        )
        dice_parts.append(_soft_dice_loss(logits, target, typed_mask))
        iou_parts.append(_soft_iou_loss(logits, target, typed_mask))
    if not bce_parts:
        denominator = cell_mask.sum().clamp_min(1.0)
        return (
            (bce * class_weight * cell_mask).sum() / denominator,
            _soft_dice_loss(logits, target, cell_mask),
            _soft_iou_loss(logits, target, cell_mask),
        )
    return (
        torch.stack(bce_parts).mean(),
        torch.stack(dice_parts).mean() if dice_parts else zero,
        torch.stack(iou_parts).mean() if iou_parts else zero,
    )


def compute_multitask_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    config: MultiTaskLossConfig,
    pos_weight: torch.Tensor | None = None,
    occupancy_type_masks: dict[str, torch.Tensor] | None = None,
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
    hot_target = batch["y_hot"].float()
    hot_weight = step_weight[:, :, None] * batch["occ_node_mask"].float()[:, None, :]
    type_masks = occupancy_type_masks or {}
    if config.type_balanced_occupancy and type_masks:
        remain_hot, remain_dice, remain_iou = _type_balanced_occupancy_losses(
            outputs["remain_hot_logit"],
            hot_target,
            hot_weight,
            type_masks,
            config,
        )
    else:
        hot_error = F.binary_cross_entropy_with_logits(
            outputs["remain_hot_logit"], hot_target, reduction="none"
        )
        hot_class_weight = 1.0 + (config.hot_pos_weight - 1.0) * hot_target
        remain_hot = (hot_error * hot_class_weight * hot_weight).sum() / (
            hot_weight.sum().clamp_min(1.0)
        )
        remain_dice = _soft_dice_loss(
            outputs["remain_hot_logit"], hot_target, hot_weight
        )
        remain_iou = _soft_iou_loss(outputs["remain_hot_logit"], hot_target, hot_weight)
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
    ongoing = (
        hist_hot | ((batch["event_start"] == 0) & (event_will_target > 0.5))
    ) & occ_mask
    ongoing = ongoing & (event_will_target > 0.5)
    upcoming = (event_will_target > 0.5) & ~ongoing & occ_mask
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
    event_weight = torch.where(
        ongoing,
        torch.full_like(event_weight, config.event_will_ongoing_pos_weight),
        event_weight,
    )
    event_will_raw = _event_binary_loss(
        outputs["event_will_logit"], event_will_target, config.event_focal_gamma
    )
    event_weight = event_weight * occ_mask.float()
    event_parts = []
    for raw_mask in type_masks.values():
        node_mask = raw_mask.to(
            device=event_weight.device, dtype=event_weight.dtype
        ).reshape(-1)
        if node_mask.numel() != event_weight.shape[-1]:
            continue
        typed_weight = event_weight * node_mask.view(1, -1)
        if float(typed_weight.sum()) <= 0:
            continue
        event_parts.append(
            (event_will_raw * typed_weight).sum() / typed_weight.sum().clamp_min(1.0)
        )
    event_will = (
        torch.stack(event_parts).mean()
        if event_parts
        else (event_will_raw * event_weight).sum() / event_weight.sum().clamp_min(1.0)
    )
    if upcoming.any():
        start_logits = outputs["event_start_logit"][upcoming]
        soft_start = gaussian_start_soft_labels(
            batch["event_start"][upcoming].detach().cpu().numpy(),
            start_logits.shape[-1],
            sigma=config.event_start_sigma,
        )
        soft_start_target = torch.from_numpy(soft_start).to(
            device=start_logits.device, dtype=start_logits.dtype
        )
        event_start = (
            -(soft_start_target * F.log_softmax(start_logits, dim=-1))
            .sum(dim=-1)
            .mean()
        )
    else:
        event_start = zero
    positive_event = (upcoming | ongoing) & occ_mask
    event_duration = (
        F.smooth_l1_loss(
            torch.log1p(outputs["event_duration"][positive_event].clamp_min(0.0)),
            torch.log1p(batch["event_duration"][positive_event].float().clamp_min(0.0)),
        )
        if positive_event.any()
        else zero
    )

    components = {
        "remain_score": remain_score,
        "remain_hot": remain_hot,
        "remain_dice": remain_dice,
        "remain_iou": remain_iou,
        "remain_len": remain_len,
        "cause": cause,
        "event_will": event_will,
        "event_start": event_start,
        "event_duration": event_duration,
    }
    total = (
        config.lambda_remain_score * remain_score
        + config.lambda_remain_hot * remain_hot
        + config.lambda_remain_dice * remain_dice
        + config.lambda_remain_iou * remain_iou
        + config.lambda_remain_len * remain_len
        + config.lambda_cause * cause
        + config.lambda_event_will * event_will
        + config.lambda_event_start * event_start
        + config.lambda_event_duration * event_duration
    )
    return total, components
