"""Checks against overstating retrospective validation trajectories."""

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "isaaclab_tasks/direct/hc_factory/tools"))
from analyze_baseline_matched_history import summarize_history


def fixture():
    training = {"checkpoint_min_report_precision": .8, "checkpoint_min_report_recall": .7}
    rows = [{"epoch": epoch, "train_total": 4. - epoch, "validation_total_loss": float(epoch),
             "validation_primary_f1": f1, "validation_primary_precision": p,
             "validation_primary_recall": r, "validation_report_recall_upcoming": up,
             "validation_event_threshold": .5, "validation_event_will_pr_auc": .6}
            for epoch, f1, p, r, up in [(1, .79, .79, .79, .1), (2, .75, .8, .71, .2), (3, .6, .5, .75, .5)]]
    return rows, training


def test_retrospective_recall_maximum_does_not_replace_feasible_selected_epoch():
    rows, training = fixture()
    result = summarize_history(rows, training, 2, 10)
    assert result["selected"]["epoch"] == 2 and result["selected"]["upcoming_strict_hits"] == 2
    assert result["retrospective_maximum_epochs"][0]["epoch"] == 3
    assert not result["retrospective_maximum_epochs"][0]["checkpoint_constraints_met"]
    assert result["extra_hits_over_selected_at_observed_thresholds"] == 3
    assert result["epochs_with_more_hits_than_selected"] == 1


@pytest.mark.parametrize("damage", ["wrong_selection", "fractional_support", "missing_epoch", "nan"])
def test_reject_unverifiable_trajectory_claims(damage):
    rows, training = fixture()
    selected = 2
    if damage == "wrong_selection": selected = 3
    elif damage == "fractional_support": rows[2]["validation_report_recall_upcoming"] = .555
    elif damage == "missing_epoch": rows.pop(0)
    else: rows[2]["train_total"] = float("nan")
    with pytest.raises(ValueError): summarize_history(rows, training, selected, 10)
