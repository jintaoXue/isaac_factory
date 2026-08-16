"""Infer sample packing (no GPU). Checkpoint forward is optional."""

from __future__ import annotations

import numpy as np

from factory_bn.dataset import build_infer_sample
from factory_bn.infer import align_episode


def test_build_infer_sample_last_window() -> None:
    t_len, n, f = 15, 3, 4
    feats = np.arange(t_len * n * f, dtype=np.float32).reshape(t_len, n, f)
    wstart = np.arange(t_len, dtype=np.float32) * 60.0
    sample = build_infer_sample(
        feats,
        wstart,
        t=t_len,
        event_node=np.array([0], dtype=np.int64),
        event_start_s=np.array([120.0], dtype=np.float32),
        event_duration_s=np.array([60.0], dtype=np.float32),
        event_start_ti=np.array([2], dtype=np.int64),
        input_window=12,
        horizon_s=180.0,
    )
    assert sample["x"].shape == (12, 3, 4)
    np.testing.assert_array_equal(sample["x"][0], feats[3, :, :])
    assert sample["has_future_score"] is False
    assert float(sample["next_tau"][0]) == 3.0
    # event at ti=2 is before hist_start=3, so it is not in the 12-window history
    assert float(sample["event_mask"][0].sum()) == 0.0


def test_hist_event_inside_window() -> None:
    t_len, n, f = 14, 2, 3
    feats = np.zeros((t_len, n, f), dtype=np.float32)
    wstart = np.arange(t_len, dtype=np.float32) * 60.0
    # t=14, hist_start=2; event at ti=5 is inside
    sample = build_infer_sample(
        feats,
        wstart,
        t=14,
        event_node=np.array([1], dtype=np.int64),
        event_start_s=np.array([5 * 60.0], dtype=np.float32),
        event_duration_s=np.array([90.0], dtype=np.float32),
        event_start_ti=np.array([5], dtype=np.int64),
        input_window=12,
    )
    assert float(sample["event_mask"][1, 0]) == 1.0
    assert int(sample["event_idx"][1, 0]) == 3  # 5 - 2
    assert abs(float(sample["event_dur"][1, 0]) - 1.5) < 1e-5
    assert sample["x"].shape == (12, 2, 3)


def test_align_episode_missing_node() -> None:
    ep = {
        "name": "toy",
        "resource_ids": ["a", "b"],
        "features": np.ones((5, 2, 2), dtype=np.float32),
        "scores": np.ones((5, 2, 1), dtype=np.float32),
        "window_start_s": np.arange(5, dtype=np.float32) * 60,
        "windows": np.arange(5),
        "will_bottleneck": np.zeros(5, dtype=np.float32),
        "mark_node": np.full(5, -1, dtype=np.int64),
        "cause": np.full(5, -1, dtype=np.int64),
        "event_node": np.array([1], dtype=np.int64),
        "event_start_s": np.array([10.0], dtype=np.float32),
        "event_duration_s": np.array([1.0], dtype=np.float32),
        "event_start_ti": np.array([0], dtype=np.int64),
    }
    aligned = align_episode(ep, ["b", "c"])
    assert aligned["features"].shape == (5, 2, 2)
    np.testing.assert_array_equal(aligned["features"][:, 0], 1.0)  # b
    np.testing.assert_array_equal(aligned["features"][:, 1], 0.0)  # c missing
    assert aligned["event_node"].tolist() == [0]


def test_build_infer_sample_too_short() -> None:
    feats = np.zeros((5, 2, 1), dtype=np.float32)
    wstart = np.arange(5, dtype=np.float32)
    try:
        build_infer_sample(feats, wstart, t=5, input_window=12)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


if __name__ == "__main__":
    test_build_infer_sample_last_window()
    test_hist_event_inside_window()
    test_align_episode_missing_node()
    test_build_infer_sample_too_short()
    print("ok")
