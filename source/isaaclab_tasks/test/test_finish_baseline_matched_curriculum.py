"""Reject false completion of an existing six-stage training queue."""

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "isaaclab_tasks/direct/hc_factory/tools"))
from finish_baseline_matched_curriculum import RUNTIME, VERIFIER_SOURCE, TASKS, pane_finished, validate_batch


def fixture():
    records = [{"model": model, "max_start": cap, "status": "validation_completed", "source_commit": RUNTIME,
                "test_evaluated": False, "artifact_sha256": {"best.pt": str(i)}} for i, (model, cap) in enumerate(TASKS)]
    proofs = [{"model": r["model"], "max_start": r["max_start"], "status": "completed_stage_verified",
               "runtime_commit": RUNTIME, "test_evaluated": False, "artifact_sha256": r["artifact_sha256"],
               "verification_source_commit": VERIFIER_SOURCE,
               "record_sha256": str(i)} for i, r in enumerate(records)]
    plan = {"status": "completed", "source_commit": RUNTIME, "tasks": records}
    batch = {**plan, "test_evaluated": False}
    return plan, batch, records, proofs, [str(i) for i in range(6)]


def test_live_pane_is_not_completion_and_all_six_stages_are_required():
    assert not pane_finished("1037530 0")
    assert pane_finished("1037530 1 0")
    validate_batch(*fixture())


@pytest.mark.parametrize("raw", ["1037530 1 1", "1037530 1", "different 1 0", ""])
def test_missing_changed_or_failed_handles_are_not_success(raw):
    with pytest.raises(ValueError): pane_finished(raw)


@pytest.mark.parametrize("damage", ["missing", "duplicate", "stale_proof", "test", "running"])
def test_incomplete_or_unverified_batches_cannot_be_final_results(damage):
    plan, batch, records, proofs, hashes = fixture()
    if damage == "missing": proofs.pop()
    elif damage == "duplicate": proofs[1] = proofs[0]
    elif damage == "stale_proof": hashes[0] = "changed"
    elif damage == "test": records[0]["test_evaluated"] = True
    else: plan["status"] = "running"
    with pytest.raises(ValueError): validate_batch(plan, batch, records, proofs, hashes)
