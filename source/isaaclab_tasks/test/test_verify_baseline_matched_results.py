"""Independent artifact checks use memory fixtures; no experiment directories."""

import io
import json
from pathlib import Path
import sys
import zipfile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "isaaclab_tasks/direct/hc_factory/tools"))
import verify_baseline_matched_results as verification


def test_selection_keeps_first_epsilon_tie_and_enforces_both_gates():
    history = [dict(epoch=i, validation_primary_f1=f, validation_primary_precision=p,
                    validation_primary_recall=r)
               for i, f, p, r in ((1, .91, .79, 1.), (2, .85, .90, .8),
                                 (3, .8500009, .95, .8), (4, .850002, .90, .8), (5, .95, 1., .65))]
    chosen, feasible = verification.selected_epoch(history[:3],
        {"checkpoint_min_report_precision": .8, "checkpoint_min_report_recall": .7})
    assert feasible and chosen["epoch"] == 2
    chosen, feasible = verification.selected_epoch(history,
        {"checkpoint_min_report_precision": .8, "checkpoint_min_report_recall": .7})
    assert feasible and chosen["epoch"] == 4
    chosen, feasible = verification.selected_epoch(history,
        {"checkpoint_min_report_precision": 1., "checkpoint_min_report_recall": 1.})
    assert not feasible and chosen["epoch"] == 5


def station_fixture():
    report = dict(n_pred_who=10, n_true_who=10, n_matched_who=8, n_matched_report=6,
                  report_threshold_used=.7)
    for prefix, value in (("who", .8), ("will15", .8), ("report", .6)):
        report.update({f"{prefix}_{name}": value for name in ("precision", "recall", "f1")})
    for group, n, who, strict in (("ongoing", 6, 5, 5), ("upcoming", 4, 3, 1)):
        report.update({f"n_true_{group}": n, f"n_matched_who_{group}": who,
                       f"n_matched_report_{group}": strict, f"who_recall_{group}": who / n,
                       f"report_recall_{group}": strict / n})
    for tolerance, hits in ((1, 5), (2, 5), (3, 6)):
        report.update({f"report_{name}_at_{tolerance}": hits / 10 for name in ("precision", "recall", "f1")})
    for name, n, hits in (("start_le_5", 3, 1), ("start_6_10", 1, 0), ("start_gt_10", 0, 0)):
        report[f"n_true_upcoming_{name}"] = n
        report[f"report_recall_upcoming_{name}"] = hits / n if n else 0.
    return report, {"ongoing": 6, "upcoming": 4, "negative": 6}


def test_separate_low_station_recognition_from_start_misses():
    report, expected = station_fixture()
    values = verification.station_counts(report, expected)
    up = values["groups"]["upcoming"]
    assert up == dict(true=4, who_hits=3, strict_hits=1, score_misses=1,
                      start_tolerance_misses=2, who_recall=.75, strict_recall=.25)
    assert values["will15_f1"] == .8 and values["strict_f1_at_1_2_3"] == [.5, .5, .6]


@pytest.mark.parametrize("key,value", [("n_true_upcoming", 4.5), ("will15_f1", .6),
    ("n_matched_report_upcoming", 4), ("report_recall_at_1", .7), ("n_true_upcoming_start_le_5", 4)])
def test_reject_inconsistent_station_metrics(key, value):
    report, expected = station_fixture(); report[key] = value
    with pytest.raises(ValueError):
        verification.station_counts(report, expected)


def prediction_fixture():
    labels = ["transport_delay", "material_shortage", "queue_buildup", "starved_upstream", "ignored"]
    pairs = [(labels[0], labels[0]), (labels[0], labels[2]), (labels[1], labels[1]), (labels[4], labels[2])]
    rows = [dict(sample_index=i, split="train", target_cause=a, predicted_cause=b,
                 target_remain_len_windows=5, predicted_remain_len_windows=6)
            for i, (a, b) in enumerate(pairs)]
    metrics = dict(sample_count=4, remain={"remain_len_mae": 1.},
                   cause={"cause_n": 3, "cause_acc": 2/3, "cause_macro_recall": .75,
                          "cause_recall_transport_delay": .5, "cause_recall_material_shortage": 1.})
    return rows, metrics, labels


def test_recompute_supported_cause_confusion_and_unweighted_remaining_time():
    rows, metrics, classes = prediction_fixture()
    confusion = verification.verify_prediction_rows(rows, set(range(4)), "train", metrics, classes)
    assert sum(confusion.values()) == 3
    assert confusion["transport_delay", "queue_buildup"] == 1
    assert confusion["ignored", "queue_buildup"] == 0


@pytest.mark.parametrize("damage", ["repeat", "split", "cause", "remain"])
def test_prediction_evaluation_rejects_duplicates_cross_split_and_wrong_metrics(damage):
    rows, metrics, classes = prediction_fixture()
    if damage == "repeat": rows[1]["sample_index"] = 0
    elif damage == "split": rows[1]["split"] = "test"
    elif damage == "cause": rows[1]["predicted_cause"] = "ignored"
    else: metrics["remain"]["remain_len_mae"] = 2.
    with pytest.raises(ValueError):
        verification.verify_prediction_rows(rows, set(range(4)), "train", metrics, classes)


@pytest.mark.parametrize("corrupt", [False, True])
def test_archive_hash_mismatch_is_not_misclassified_as_live_transition(monkeypatch, corrupt):
    expected = {"best.pt": verification.digest(b"weights")}
    raw = io.BytesIO()
    real_zip = zipfile.ZipFile
    with real_zip(raw, "w") as archive:
        archive.writestr("archive_manifest.json", json.dumps(expected))
        archive.writestr("best.pt", b"damaged" if corrupt else b"weights")
    monkeypatch.setattr(verification.zipfile, "ZipFile", lambda path: real_zip(io.BytesIO(raw.getvalue())))
    monkeypatch.setattr(Path, "exists", lambda path: True)
    if corrupt:
        with pytest.raises(ValueError, match="SHA differs"):
            verification.artifact_snapshot(Path("existing"), Path("existing/model.zip"), expected)
    else:
        files, origin = verification.artifact_snapshot(Path("existing"), Path("existing/model.zip"), expected)
        assert files["best.pt"] == b"weights" and origin == "existing/model.zip"
