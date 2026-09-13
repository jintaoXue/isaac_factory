#!/usr/bin/env python3
"""Compare sixteen completed GRU32 views with cached near-parent views only."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

RUNTIME = "90a41a5b9c44e625f8c80083a8a950cd50f5eb13"
PARENT_SHA = "73dbbb0f81f43ac9422e1fb914247113a4936a9983694ea96648a291127b459c"
RELATIVE = "source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/tools/compare_baseline_gru_capacity.py"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def view_stats(result):
    """Report hits exclude timing misses; station-only matches do not."""
    row, = [x for x in result["thresholds"] if x["threshold"] == result["saved_report_threshold"]]
    n = row["n_true_upcoming"]
    hits = row["report_recall_upcoming"] * n
    assert abs(hits - round(hits)) < 1e-8
    assert round(hits) == row["n_matched_who_upcoming"] - row["upcoming_timing_misses"]
    assert n == result["groups"]["upcoming"]["count"]
    return dict(upcoming_AP=result["ranking"]["upcoming_vs_negative"]["tie_aware_average_precision"],
        upcoming_probability_median=result["groups"]["upcoming"]["will_q10_q50_q90"][1],
        upcoming_report_hits=round(hits), upcoming_true=n, upcoming_recall=row["report_recall_upcoming"],
        upcoming_station_hits=row["n_matched_who_upcoming"], upcoming_timing_misses=row["upcoming_timing_misses"],
        precision=row["report_precision"], recall=row["report_recall"], f1=row["report_f1"],
        saved_threshold=row["threshold"])


def parent_filename(model, seed, checkpoint, split):
    if checkpoint == "last" and model == "b5":
        return f"baseline_near_parent_b5s{seed}_last_{split}20260913.json"
    tag = "headprobe20260912" if checkpoint == "best" else "nearpreclast20260912"
    return f"{model}_seed{seed}_{split}_diagnostics_{tag}.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_commit", required=True)
    parser.add_argument("--candidate_sha256", required=True)
    args = parser.parse_args()
    repo = Path.cwd().resolve(); assert repo == Path("/home/sci/work/BSTAN_isaac_factory")
    d = repo / "source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/output/bottleneck_dataset/experiments/factory_pdformer_134_v3"
    parent_path = d / "baseline_readout_dropout_parent_comparison20260913.json"
    candidate_path = d / "baseline_dense_gru_capacity_metrics_20260913.json"
    output = d / "baseline_gru_capacity_parent_comparison20260913.json"; assert not output.exists()
    assert sha(parent_path) == PARENT_SHA and sha(candidate_path) == args.candidate_sha256
    parent, candidate = json.loads(parent_path.read_text()), json.loads(candidate_path.read_text())
    assert candidate["status"] == "four_training_runs_and_sixteen_frozen_diagnostics_completed_and_verified"
    assert candidate["source_commit"] == RUNTIME and not candidate["test_evaluated"] and not parent["test_evaluated"]
    assert not parent["parent_model_retrained"]
    parent_views = {x["file"]: x for x in parent["parent_diagnostics"]}; assert len(parent_views) == 16
    source = subprocess.check_output(["git", "show", args.source_commit + ":" + RELATIVE])

    def guard():
        assert subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip() == RUNTIME
        assert subprocess.check_output(["git", "branch", "--show-current"], text=True).strip() == "dev_xwt"
        assert not subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], text=True).strip()
        assert sha(parent_path) == PARENT_SHA and sha(candidate_path) == args.candidate_sha256
        for name, h in candidate["runtime_source_sha256"].items(): assert sha(repo / name) == h
        for name, v in candidate["dataset_files_stat"].items():
            s = (d / name).stat(); assert dict(size=s.st_size, mtime_ns=s.st_mtime_ns) == v

    guard(); rows = []; seen = set()
    for run in candidate["runs"]:
        model, seed = run["model"], run["seed"]
        assert model in ("b4", "b5") and seed in (42, 43)
        out = d / f"models/tuning/{model}_representation_v1/candidate_history/seed{seed}"
        for name, h in run["training"]["files_sha256"].items(): assert sha(out / name) == h
        for view in run["diagnostics"]:
            checkpoint, split = view["checkpoint"], view["split"]
            key = (model, seed, checkpoint, split); assert key not in seen; seen.add(key)
            original = parent_views[parent_filename(*key)]
            for item in (original, view):
                assert sha(d / item["file"]) == item["file_sha256"]
                assert json.loads((d / item["file"]).read_text()) == item["result"]
                result = item["result"]
                assert result["split"] == split and not result["test_evaluated"]
                assert result["dataset_manifest_sha256"] == candidate["dataset_manifest_sha256"]
                assert result["sample_count"] == (23859 if split == "train" else 5439)
                assert [result["groups"][g]["count"] for g in ("ongoing", "upcoming", "negative")] == (
                    [4191, 595, 297152] if split == "train" else [950, 145, 67331])
            before, after = view_stats(original["result"]), view_stats(view["result"])
            rows.append(dict(model=model, seed=seed, checkpoint=checkpoint, split=split,
                parent_file=original["file"], parent_file_sha256=original["file_sha256"],
                candidate_file=view["file"], candidate_file_sha256=view["file_sha256"],
                parent=before, candidate=after,
                candidate_minus_parent={k: after[k] - before[k] for k in ("upcoming_AP", "upcoming_report_hits", "upcoming_recall", "precision", "recall", "f1")}))
    assert seen == {(m, s, c, v) for m in ("b4", "b5") for s in (42, 43) for c in ("best", "last") for v in ("train", "validation")}
    guard()
    record = dict(status="sixteen_matched_cached_near_parent_comparisons_verified", runtime_commit=RUNTIME,
        source_commit=args.source_commit, source_sha256=hashlib.sha256(source).hexdigest(),
        parent_summary_sha256=PARENT_SHA, candidate_summary_sha256=args.candidate_sha256,
        comparisons=rows, model_forward=False, model_training=False, new_threshold_selection=False,
        parent_model_retrained=False, test_evaluated=False, goal_met=False)
    with output.open("x") as f: json.dump(record, f, indent=2); f.write("\n")
    for row in rows:
        if row["checkpoint"] == "best" and row["split"] == "validation":
            print("FORMAL", row["model"], row["seed"], "UP", row["parent"]["upcoming_report_hits"], "->", row["candidate"]["upcoming_report_hits"],
                "AP", row["parent"]["upcoming_AP"], "->", row["candidate"]["upcoming_AP"], "P", row["candidate"]["precision"], "F1", row["candidate"]["f1"], flush=True)
    print("GRU_CAPACITY_PARENT_COMPARISON_COMPLETE", output.stat().st_size, sha(output), flush=True)


if __name__ == "__main__": main()
