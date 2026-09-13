#!/usr/bin/env python3
"""Wait for the existing six-task queue, then verify and collect its official results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import zipfile

from verify_baseline_matched_results import RUNTIME, RELATIVE as VERIFIER_PATH, artifact_snapshot, digest, sha


VERIFIER_SOURCE = "8c9507645607ea48cd143b0d914e5fe26bac8028"
TRAINING_PANE = "baseline_dense_v6:0.0"
TRAINING_PID = "1037530"
TASKS = [(model, cap) for cap in (5, 10, 15) for model in ("B4", "B5")]
FINAL = "baseline_matched_protocol_training20260913_final_verification.json"
STATE = "baseline_matched_protocol_finish20260914_state.json"
SELF_PATH = "source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/tools/finish_baseline_matched_curriculum.py"


def pane_finished(raw):
    fields = raw.split()
    if len(fields) < 2 or fields[0] != TRAINING_PID or fields[1] not in ("0", "1"):
        raise ValueError("The original training handle changed; inspect it without restarting")
    if fields[1] == "0":
        return False
    if fields != [TRAINING_PID, "1", "0"]:
        raise ValueError("Original queue ended unsuccessfully; do not verify it as complete")
    return True


def validate_batch(plan, batch, records, proofs, record_hashes):
    identities = lambda values: [(x["model"], x["max_start"]) for x in values]
    if (plan["status"] != "completed" or batch["status"] != "completed"
            or plan["source_commit"] != RUNTIME or batch["source_commit"] != RUNTIME
            or batch["test_evaluated"] or identities(plan["tasks"]) != TASKS
            or identities(records) != TASKS or identities(proofs) != TASKS
            or batch["tasks"] != records or len(record_hashes) != 6):
        raise ValueError("The six registered tasks are not complete and consistent")
    for record, proof, expected in zip(records, proofs, record_hashes):
        if (record["status"] != "validation_completed" or record["source_commit"] != RUNTIME
                or record["test_evaluated"] or proof["status"] != "completed_stage_verified"
                or proof["runtime_commit"] != RUNTIME or proof["test_evaluated"]
                or proof["verification_source_commit"] != VERIFIER_SOURCE
                or proof["record_sha256"] != expected or proof["artifact_sha256"] != record["artifact_sha256"]):
            raise ValueError("A stage lacks a matching completed verification")


def write(path, value, exclusive=False):
    with path.open("x" if exclusive else "w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write("\n")


def finish(dataset, source_commit):
    final, state = dataset / FINAL, dataset / STATE
    if final.exists() or state.exists():
        raise FileExistsError("An existing finisher must be inspected, not duplicated")
    metadata = {"source_commit": source_commit, "runtime_commit": RUNTIME,
                "training_pane": TRAINING_PANE, "training_pid": int(TRAINING_PID),
                "new_training": False, "new_diagnostics": False, "test_evaluated": False}
    write(state, {**metadata, "status": "waiting_for_original_training"}, exclusive=True)
    print("WAITING_FOR_ORIGINAL_MATCHED_QUEUE", TRAINING_PID, flush=True)
    previous_error = None
    while True:
        try:
            raw = subprocess.check_output(["tmux", "display-message", "-p", "-t", TRAINING_PANE,
                                           "#{pane_pid} #{pane_dead} #{pane_dead_status}"], text=True, timeout=10).strip()
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
            # An observer failure is not evidence of a terminal training job.
            message = type(error).__name__
            if message != previous_error:
                print("TRAINING_OBSERVATION_UNAVAILABLE", message, flush=True)
                previous_error = message
            time.sleep(20)
            continue
        previous_error = None
        if pane_finished(raw):
            break
        time.sleep(20)
    write(state, {**metadata, "status": "verifying_completed_queue"})
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip() == RUNTIME
    assert not subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], text=True).strip()
    plan_path = dataset / "baseline_matched_protocol_training20260913_plan.json"
    batch_path = dataset / "baseline_matched_protocol_training20260913_results.json"
    plan, batch = json.loads(plan_path.read_text()), json.loads(batch_path.read_text())
    if plan["status"] != "completed" or batch["status"] != "completed":
        raise ValueError("Successful pane exit has no complete six-task result")
    proofs, records, record_hashes, verification_hashes = [], [], [], {}
    for task in plan["tasks"]:
        model, cap = task["model"], task["max_start"]
        output = dataset / f"baseline_matched_{model.lower()}_start{cap}_verification20260913.json"
        if not output.exists():
            loader = ("import sys,subprocess; from pathlib import Path; "
                      f"p={VERIFIER_PATH!r}; s={VERIFIER_SOURCE!r}; "
                      f"sys.argv=[p,'--source_commit',s,'--model',{model!r},'--max_start',{str(cap)!r}]; "
                      "exec(compile(subprocess.check_output(['git','show',s+':'+p]),p,'exec'),"
                      "{'__name__':'__main__','__file__':str(Path.cwd()/p)})")
            subprocess.run([sys.executable, "-B", "-u", "-c", loader], check=True)
        raw_record = Path(task["record"]).read_bytes()
        record, proof = json.loads(raw_record), json.loads(output.read_text())
        directory = Path(task["output_dir"])
        archive = directory / f"model_before_matched20260913_s{cap + 5}.zip" if cap < 15 else None
        # Reuse completed numerical checks; confirm their bytes survived later-stage archiving.
        artifact_snapshot(directory, archive, record["artifact_sha256"])
        records.append(record); proofs.append(proof); record_hashes.append(digest(raw_record))
        verification_hashes[output.name] = sha(output)
    validate_batch(plan, batch, records, proofs, record_hashes)
    archived = {}
    for task in plan["tasks"][:2]:
        path = Path(task["archive"])
        record = json.loads(Path(task["record"]).read_text())
        if sha(path) != record["prior_artifact_archive_sha256"]:
            raise ValueError("Original model archive changed")
        with zipfile.ZipFile(path) as z:
            for name, expected in json.loads(z.read("archive_manifest.json")).items():
                content = z.read(name)
                if digest(content) != expected:
                    raise ValueError("Original archived member changed")
                archived[str((Path(task["output_dir"]) / name).relative_to(dataset))] = len(content)
    for name, stat in plan["protected_files_stat_before_training"].items():
        if name in archived:
            assert archived[name] == stat[0], name
        else:
            actual = (dataset / name).stat()
            assert [actual.st_size, actual.st_mtime_ns] == stat, name
    result = {**metadata, "status": "six_matched_tasks_independently_verified", "training_terminal": raw,
              "source_sha256": digest(subprocess.check_output(["git", "show", source_commit + ":" + SELF_PATH])),
              "plan_sha256": sha(plan_path), "batch_result_sha256": sha(batch_path),
              "stage_verification_sha256": verification_hashes,
              "protected_original_files_verified": len(plan["protected_files_stat_before_training"]),
              "results": [{"model": p["model"], "max_start": p["max_start"], "selected_epoch": p["selected_epoch"],
                           "epochs_trained": p["epochs_trained"], "scores": p["scores"],
                           "training_budget": p["training_budget"], "metrics": r["metrics"]} for r, p in zip(records, proofs)],
              "scope": "Official selected results of the six registered tasks only; no new diagnosis, threshold selection or model training."}
    write(final, result, exclusive=True)
    write(state, {**metadata, "status": "completed", "final_sha256": sha(final)})
    print("SIX_MATCHED_TASKS_VERIFIED", sha(final), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_commit", required=True)
    args = parser.parse_args()
    repo = Path.cwd().resolve()
    assert repo == Path("/home/sci/work/BSTAN_isaac_factory")
    assert subprocess.check_output(["git", "branch", "--show-current"], text=True).strip() == "dev_xwt"
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip() == RUNTIME
    dataset = repo / "source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/output/bottleneck_dataset/experiments/factory_pdformer_134_v3"
    try:
        finish(dataset, args.source_commit)
    except Exception as error:
        print("MATCHED_FINISHER_STOPPED", type(error).__name__, str(error), flush=True)
        raise
