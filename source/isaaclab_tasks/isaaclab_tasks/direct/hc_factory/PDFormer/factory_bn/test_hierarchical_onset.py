"""Tests for the Start<=15 hierarchical onset contract."""

from __future__ import annotations

import torch

from factory_bn.model import decode_hierarchical_onset, hierarchical_onset_targets
from factory_bn.remain import station_report_metrics


def test_hierarchical_onset_boundaries() -> None:
    start = torch.tensor([0, 1, 5, 6, 10, 11, 14])
    ongoing = torch.tensor([True, False, False, False, False, False, False])
    bucket, minute = hierarchical_onset_targets(start, ongoing)
    assert bucket.tolist() == [0, 1, 1, 2, 2, 3, 3]
    assert minute.tolist() == [0, 0, 4, 0, 4, 0, 3]


def test_hierarchical_onset_decode_and_horizon_clamp() -> None:
    bucket_id = torch.tensor([[0, 1, 1, 2, 2, 3, 3]])
    minute_id = torch.tensor([[0, 0, 4, 0, 4, 0, 4]])
    bucket_logits = torch.full((1, 7, 4), -10.0)
    minute_logits = torch.full((1, 7, 5), -10.0)
    bucket_logits.scatter_(-1, bucket_id.unsqueeze(-1), 10.0)
    minute_logits.scatter_(-1, minute_id.unsqueeze(-1), 10.0)
    decoded, bucket_prob, minute_prob = decode_hierarchical_onset(
        bucket_logits,
        minute_logits,
        max_start_windows=14,
    )
    assert decoded.tolist() == [[0, 1, 5, 6, 10, 11, 14]]
    assert bucket_prob.shape == (1, 7, 4)
    assert minute_prob.shape == (1, 7, 5)


def test_no_event_is_owned_by_will15_not_onset_bucket() -> None:
    y_hot = torch.zeros(1, 20, 1).numpy()
    metrics = station_report_metrics(
        y_hot,
        will_prob=torch.tensor([[0.1]]).numpy(),
        start_idx=torch.tensor([[14]]).numpy(),
        dur=torch.tensor([[5.0]]).numpy(),
        remain_mask=torch.ones(1, 20).numpy(),
        occ_node_mask=torch.ones(1).numpy(),
        threshold=0.5,
        min_windows=5,
        max_start_windows=14,
    )
    assert metrics["n_true_who"] == 0.0
    assert metrics["n_pred_who"] == 0.0


if __name__ == "__main__":
    test_hierarchical_onset_boundaries()
    test_hierarchical_onset_decode_and_horizon_clamp()
    test_no_event_is_owned_by_will15_not_onset_bucket()
    print("ok")
