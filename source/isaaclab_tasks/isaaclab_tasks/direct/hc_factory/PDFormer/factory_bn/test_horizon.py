"""Horizon censoring for STGNPP next-event labels (no GPU)."""

from __future__ import annotations

import numpy as np

from factory_bn.dataset import _build_samples


def _episode(n_nodes: int = 4, t_len: int = 20) -> dict:
    feats = np.zeros((t_len, n_nodes, 2), dtype=np.float32)
    scores = np.zeros((t_len, n_nodes, 1), dtype=np.float32)
    will = np.zeros((t_len,), dtype=np.float32)
    return {
        "name": "toy",
        "episode_id": 0,
        "features": feats,
        "scores": scores,
        "will": will,
        "mark": np.full((t_len,), -1, dtype=np.int64),
        "cause": np.full((t_len,), -1, dtype=np.int64),
        "tts": np.zeros((t_len,), dtype=np.float32),
        "duration": np.zeros((t_len,), dtype=np.float32),
        "window_start_s": np.arange(t_len, dtype=np.float32) * 60.0,
        "event_node": np.array([0, 1], dtype=np.int64),
        # node 0: event 2 min after window 12's label time (t=12, label_idx=11 → ref=660s → event at 780s)
        # node 1: event 20 min later (beyond 180s horizon)
        "event_start_s": np.array([780.0, 1860.0], dtype=np.float32),
        "event_duration_s": np.array([60.0, 60.0], dtype=np.float32),
        "event_start_ti": np.array([13, 31], dtype=np.int64),
    }


def test_horizon_censors_far_events() -> None:
    samples = _build_samples(
        [_episode()],
        input_window=12,
        output_window=1,
        horizon_windows=3,
        max_hist_events=4,
        window_size_s=60.0,
        horizon_s=180.0,
    )
    assert samples, "expected sliding windows"
    # t=12 → label_idx=11, ref_s=660. node0 tau=(780-660)/60=2 min ≤ 3; node1 far.
    first = samples[0]
    assert first["next_mask"][0] == 1.0
    assert first["surv_mask"][0] == 0.0
    assert abs(float(first["next_tau"][0]) - 2.0) < 1e-5
    assert first["next_mask"][1] == 0.0
    assert first["surv_mask"][1] == 1.0
    assert abs(float(first["next_tau"][1]) - 3.0) < 1e-5
    # nodes 2,3 never have events → survival at H
    assert first["surv_mask"][2] == 1.0 and first["next_mask"][2] == 0.0
    n_pos = int(sum(s["next_mask"].sum() for s in samples))
    n_surv = int(sum(s["surv_mask"].sum() for s in samples))
    assert n_pos + n_surv == len(samples) * 4
    assert n_pos < n_surv


if __name__ == "__main__":
    test_horizon_censors_far_events()
    print("ok")
