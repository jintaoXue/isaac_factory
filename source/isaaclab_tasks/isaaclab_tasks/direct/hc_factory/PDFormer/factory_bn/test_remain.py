"""Remaining-jobs occupancy packing (no GPU)."""

from __future__ import annotations

import numpy as np

from factory_bn.dataset import _build_samples
from factory_bn.remain import (
    first_done_index,
    jobs_remaining_series,
    node_hot_mask,
    occupancy_to_events,
    pack_remain_target,
)


def test_node_hot_includes_score() -> None:
    t_len, n, f = 3, 2, 22
    feats = np.zeros((t_len, n, f), dtype=np.float32)
    scores = np.zeros((t_len, n, 1), dtype=np.float32)
    scores[1, 0, 0] = 0.7
    feats[2, 1, 19] = 1.0  # turning point
    hot = node_hot_mask(feats, scores, score_threshold=0.55)
    assert hot[1, 0] == 1.0
    assert hot[2, 1] == 1.0
    assert hot[0].sum() == 0.0


def test_jobs_remaining_series() -> None:
    kpi = [
        {"complete_s": "100"},
        {"complete_s": "250"},
        {"complete_s": ""},
    ]
    w = np.array([0.0, 60.0, 120.0, 240.0], dtype=np.float32)
    rem, total = jobs_remaining_series(kpi, w)
    assert total == 3.0
    assert rem.tolist() == [3.0, 3.0, 2.0, 2.0]
    assert first_done_index(rem) == 4


def test_pack_remain_and_events() -> None:
    t_len, n = 8, 2
    scores = np.zeros((t_len, n, 1), dtype=np.float32)
    hot = np.zeros((t_len, n), dtype=np.float32)
    hot[3:6, 0] = 1.0
    scores[3:6, 0, 0] = 0.9
    y_s, y_h, mask, remain_len = pack_remain_target(
        scores, hot, t=2, done_ti=7, max_remain_windows=10
    )
    assert remain_len == 5
    assert mask[:5].sum() == 5
    assert mask[5:].sum() == 0
    assert y_h[1:4, 0].sum() == 3
    events = occupancy_to_events(
        y_h[:5],
        resource_ids=["a", "b"],
        first_future_start_s=120.0,
        window_size_s=60.0,
    )
    assert len(events) == 1
    assert events[0]["resource_id"] == "a"
    assert events[0]["n_windows"] == 3
    assert abs(events[0]["duration_s"] - 180.0) < 1e-6


def test_build_samples_remain_horizon() -> None:
    t_len, n_nodes = 20, 3
    feats = np.zeros((t_len, n_nodes, 4), dtype=np.float32)
    scores = np.zeros((t_len, n_nodes, 1), dtype=np.float32)
    scores[15:, 1, 0] = 0.8
    jobs = np.array([3] * 16 + [2, 1, 0, 0], dtype=np.float32)
    ep = {
        "name": "toy",
        "episode_id": 0,
        "features": feats,
        "scores": scores,
        "will": np.zeros((t_len,), dtype=np.float32),
        "mark": np.full((t_len,), -1, dtype=np.int64),
        "cause": np.full((t_len,), -1, dtype=np.int64),
        "tts": np.zeros((t_len,), dtype=np.float32),
        "duration": np.zeros((t_len,), dtype=np.float32),
        "window_start_s": np.arange(t_len, dtype=np.float32) * 60.0,
        "event_node": np.zeros((0,), dtype=np.int64),
        "event_start_s": np.zeros((0,), dtype=np.float32),
        "event_duration_s": np.zeros((0,), dtype=np.float32),
        "event_start_ti": np.zeros((0,), dtype=np.int64),
        "jobs_remaining": jobs,
        "jobs_total": 3.0,
    }
    samples = _build_samples(
        [ep],
        input_window=12,
        output_window=1,
        horizon_windows=3,
        remain_to_jobs_done=True,
        max_remain_windows=16,
        window_size_s=60.0,
        horizon_s=180.0,
    )
    assert samples
    first = samples[0]
    assert first["jobs_remaining"] == 3.0
    assert first["remain_len"] == 6
    assert float(first["remain_mask"].sum()) == 6
    last_ok = [s for s in samples if s["jobs_remaining"] > 0]
    assert all(s["remain_len"] > 0 for s in last_ok)


if __name__ == "__main__":
    test_node_hot_includes_score()
    test_jobs_remaining_series()
    test_pack_remain_and_events()
    test_build_samples_remain_horizon()
    print("ok")
