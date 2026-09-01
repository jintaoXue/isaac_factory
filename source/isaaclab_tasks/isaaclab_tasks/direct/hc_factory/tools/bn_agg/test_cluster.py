"""K-means clustering of window features without bottleneck scores."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_TOOLS = Path(__file__).resolve().parent.parent
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

from bn_agg.cluster import (
    CLUSTER_FEATURE_COLS,
    assign_rows,
    fit_cause_aligned_cluster_model,
    fit_cluster_model,
    kmeans_pp,
    row_vector,
    seed_cluster_name,
)


def test_kmeans_separates_blobs() -> None:
    rng = np.random.default_rng(0)
    a = rng.normal(0.0, 0.1, size=(40, 2))
    b = rng.normal(5.0, 0.1, size=(40, 2))
    x = np.concatenate([a, b], axis=0)
    labels, _c, inertia = kmeans_pp(x, 2, seed=0)
    assert inertia > 0
    assert set(labels[:40].tolist()) != set(labels[40:].tolist()) or (
        labels[:40].min() == labels[:40].max() and labels[40:].min() == labels[40:].max()
        and labels[0] != labels[40]
    )
    assert labels[:40].min() == labels[:40].max()
    assert labels[40:].min() == labels[40:].max()
    assert int(labels[0]) != int(labels[40])


def test_cluster_ignores_bottleneck_score() -> None:
    rows = []
    for i in range(20):
        row = {c: 0.0 for c in CLUSTER_FEATURE_COLS}
        row.update(
            {
                "run_id": "t",
                "env_id": "0",
                "window_size_s": 60.0,
                "window_index": i,
                "resource_id": "m0",
                "resource_type": "machine",
                "queue_length_s": 0.0 if i < 10 else 4.0,
                "blocked_time_s": 0.0 if i < 10 else 40.0,
                "bottleneck_score_s": 99.0,
                "is_window_peak": 1,
            }
        )
        rows.append(row)
    v0 = row_vector(rows[0])
    assert "bottleneck_score_s" not in CLUSTER_FEATURE_COLS
    model = fit_cluster_model(rows, n_clusters=2, seed=0, window_size=60.0)
    out = assign_rows(rows, model)
    idle = {r["window_cluster_id"] for r in out if r["window_index"] < 10}
    busy = {r["window_cluster_id"] for r in out if r["window_index"] >= 10}
    assert idle.isdisjoint(busy)
    assert all(int(r["cluster_id"]) >= 0 for r in out)
    assert v0.shape[0] == len(CLUSTER_FEATURE_COLS) + 5


def _blank_row(i: int, **kw: float) -> dict:
    row = {c: 0.0 for c in CLUSTER_FEATURE_COLS}
    row.update(
        {
            "run_id": "t",
            "env_id": "0",
            "window_size_s": 60.0,
            "window_index": i,
            "window_start_s": float(i * 60),
            "window_end_s": float((i + 1) * 60),
            "resource_id": "m0",
            "resource_type": "machine",
        }
    )
    row.update(kw)
    return row


def test_cause_aligned_separates_normal_and_causes() -> None:
    rows = []
    for i in range(8):
        rows.append(_blank_row(i, active_pct_s=0.1))
    for i in range(8, 16):
        rows.append(_blank_row(i, material_shortage_propagation_s=0.8, starved_time_s=40.0))
    for i in range(16, 24):
        rows.append(_blank_row(i, inbound_wait_s=30.0))
    for i in range(24, 32):
        rows.append(_blank_row(i, queue_length_s=3.0, avg_waiting_time_s=25.0))
    for i in range(32, 40):
        rows.append(_blank_row(i, blocked_time_s=40.0, starved_time_s=5.0))
    for i in range(40, 48):
        rows.append(_blank_row(i, starved_time_s=20.0))
    for i in range(48, 56):
        rows.append(_blank_row(i, active_pct_s=0.95))
    assert seed_cluster_name(rows[0]) == "normal"
    assert seed_cluster_name(rows[8]) == "material_shortage"
    assert seed_cluster_name(rows[16]) == "transport_delay"
    assert seed_cluster_name(rows[24]) == "queue_buildup"
    assert seed_cluster_name(rows[32]) == "blocked_downstream"
    assert seed_cluster_name(rows[40]) == "starved_upstream"
    assert seed_cluster_name(rows[48]) == "high_utilization"
    model = fit_cause_aligned_cluster_model(rows, seed=0, window_size=60.0)
    names = model["cluster_names"]
    assert "normal" in names
    assert "material_shortage" in names
    assert "queue_buildup" in names
    out = assign_rows(rows, model)
    by_name = {}
    for r in out:
        by_name.setdefault(r["cluster_name"], set()).add(r["window_index"] // 8)
    # each cause block (8 rows) should land on its own named cluster
    assert by_name["normal"] == {0}
    assert by_name["material_shortage"] == {1}
    assert by_name["transport_delay"] == {2}
    assert by_name["queue_buildup"] == {3}
    assert by_name["blocked_downstream"] == {4}
    assert by_name["starved_upstream"] == {5}
    assert by_name["high_utilization"] == {6}
    assert min(model["cluster_purity"]) >= 0.99


if __name__ == "__main__":
    test_kmeans_separates_blobs()
    test_cluster_ignores_bottleneck_score()
    test_cause_aligned_separates_normal_and_causes()
    print("ok")
