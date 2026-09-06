import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "isaaclab_tasks/direct/hc_factory/tools"))
from audit_baseline_validation_contract import record_difference


def test_exact_targets_and_float_round_trip_tolerance():
    stats = {}
    record_difference(stats, "hot", [0, 1, 0], [0, 1, 0], 0)
    record_difference(stats, "features", [1, 2.000001], [1, 2], 0)
    assert all(row["mismatched_samples"] == 0 for row in stats.values())
    record_difference(stats, "jobs_remaining", 9, 10, 704)
    assert stats["jobs_remaining"] == {
        "mismatched_samples": 1, "mismatched_cells": 1,
        "max_absolute_difference": 1.0, "example_samples": [704],
    }


def test_differences_aggregate_and_invalid_arrays_fail():
    stats = {}
    for index in range(10):
        record_difference(stats, "event_will", [0, 0], [1, 1], index)
    assert stats["event_will"]["mismatched_samples"] == 10
    assert stats["event_will"]["mismatched_cells"] == 20
    assert stats["event_will"]["example_samples"] == list(range(8))
    with pytest.raises(ValueError, match="shape"):
        record_difference(stats, "mask", [0], [0, 1], 0)
    with pytest.raises(ValueError, match="non-finite"):
        record_difference(stats, "input", [np.nan], [0], 0)
