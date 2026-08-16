"""Remaining-jobs occupancy targets shared with dev_tyx PDFormer A.1."""

from __future__ import annotations

import torch


def node_hot_mask(
    features: torch.Tensor,
    scores: torch.Tensor,
    score_threshold: float = 0.55,
) -> torch.Tensor:
    """Score threshold plus turning-point and active L2 indicators."""
    hot = scores[..., 0] >= score_threshold
    disturbance_index = 18
    turning_point_index = 19
    if features.shape[-1] > turning_point_index:
        hot = (
            hot
            | (features[..., disturbance_index] > 0.5)
            | (features[..., turning_point_index] > 0.5)
        )
    return hot.float()


def pack_remain_target(
    scores: torch.Tensor,
    hot: torch.Tensor,
    start: int,
    done: int,
    max_remain_windows: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Pad future ``[start, done)`` to the shared fixed remain horizon."""
    node_count = scores.shape[1]
    y_score = scores.new_zeros((max_remain_windows, node_count, 1))
    y_hot = hot.new_zeros((max_remain_windows, node_count))
    remain_mask = hot.new_zeros((max_remain_windows,))
    remain_len = max(done - start, 0)
    length = min(remain_len, max_remain_windows)
    if length:
        y_score[:length] = scores[start : start + length]
        y_hot[:length] = hot[start : start + length]
        remain_mask[:length] = 1.0
    return y_score, y_hot, remain_mask, remain_len
