"""Check old loose-checkpoint and archived-parent diagnostic provenance."""

from copy import deepcopy
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "isaaclab_tasks/direct/hc_factory/tools"))
from compare_baseline_readout_parent import validate_diagnostic


def fixture():
    run = dict(record=dict(summary=dict(best_epoch=5, epochs_trained=25)), files_sha256={"last.pt": "member"})
    result = dict(split="train", test_evaluated=False, sample_count=23859, dataset_manifest_sha256="manifest",
        epoch=25, checkpoint_archive_member=None, checkpoint_file_sha256="member",
        groups={k: dict(count=n) for k, n in zip(("ongoing", "upcoming", "negative"), (4191, 595, 297152))},
        thresholds=[dict(threshold=.7)], saved_report_threshold=.7,
        ranking=dict(upcoming_vs_negative=dict(tie_aware_average_precision=.1)))
    return run, result


def reject(run, result, archive_sha=None):
    try: validate_diagnostic(result, run, "last", "train", "manifest", archive_sha)
    except (AssertionError, KeyError): pass
    else: raise AssertionError("Mismatched diagnostic was accepted")


def test_loose_parent_uses_checkpoint_hash():
    run, result = fixture()
    assert validate_diagnostic(result, run, "last", "train", "manifest") == .1


def test_archived_parent_checks_container_and_member_separately():
    run, result = fixture()
    result.update(checkpoint_file_sha256="container", checkpoint_member_sha256="member", checkpoint_archive_member="last.pt")
    assert validate_diagnostic(result, run, "last", "train", "manifest", "container") == .1
    bad = deepcopy(result); bad["checkpoint_file_sha256"] = "member"; reject(run, bad, "container")
    bad = deepcopy(result); bad["checkpoint_member_sha256"] = "container"; reject(run, bad, "container")


def test_mismatched_split_epoch_manifest_and_counts_are_rejected():
    run, result = fixture()
    for key, value in (("split", "validation"), ("epoch", 5), ("sample_count", 5439),
                       ("dataset_manifest_sha256", "different"), ("test_evaluated", True)):
        bad = deepcopy(result); bad[key] = value; reject(run, bad)
    bad = deepcopy(result); bad["groups"]["upcoming"]["count"] = 145; reject(run, bad)


def test_best_validation_reproduces_original_canonical_counts_and_threshold():
    run, result = fixture()
    run["files_sha256"]["best.pt"] = "best"
    original = dict(report_threshold_used=.6, n_matched_who_upcoming=2, report_f1=.75)
    run["metrics"] = dict(station_report=original)
    result.update(split="validation", sample_count=5439, epoch=5, checkpoint_file_sha256="best",
        groups={k: dict(count=n) for k, n in zip(("ongoing", "upcoming", "negative"), (950, 145, 67331))},
        saved_report_threshold=.6, thresholds=[dict(threshold=.6, n_matched_who_upcoming=2, report_f1=.75)])
    assert validate_diagnostic(result, run, "best", "validation", "manifest") == .1
    result["thresholds"][0]["n_matched_who_upcoming"] = 3
    try: validate_diagnostic(result, run, "best", "validation", "manifest")
    except AssertionError: pass
    else: raise AssertionError("Changed formal hit count accepted")
