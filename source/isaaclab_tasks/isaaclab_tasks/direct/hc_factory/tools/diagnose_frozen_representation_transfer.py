#!/usr/bin/env python3
"""Analyze reusable frozen representations without running a model again."""

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np


SOURCE_PATH = "source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/tools/diagnose_frozen_representation_transfer.py"
EXPORT_COMMIT = "d79a09357fa14fc277d904a1b969ef4409dc7101"
EXPORT_SHA = "9ce7bf672414716f4eb92664240f419dc85c8af046886e3370a20c0170b1235b"
RUNTIME = "979c680fb4d1919ae760cdfb0038f69fb7cc6708"


def centroid_fit(x, labels):
    """Equal-prior class centers in coordinates standardized on the fitting set.

    No fitted regularization or validation labels. A variance below 1e-12 uses
    unit scaling; the raw X normalization and neural weights remain frozen.
    """
    x = np.asarray(x, dtype=np.float64); y = np.asarray(labels)
    if x.ndim != 2 or len(x) != len(y) or not np.isfinite(x).all() or set(y.tolist()) != {0, 1}:
        raise ValueError("Need finite fitting features and both classes")
    variance = x.var(0); scale2 = np.where(variance < 1e-12, 1., variance)
    positive, negative = x[y == 1].mean(0), x[y == 0].mean(0)
    weight = (positive - negative) / scale2
    offset = -float(((positive + negative) * .5) @ weight)
    return weight, offset


def centroid_scores(x_train, labels_train, episodes_train, x_query, excluded_episodes=None):
    if excluded_episodes is None:
        w, b = centroid_fit(x_train, labels_train)
        return np.asarray(x_query, dtype=np.float64) @ w + b
    result = np.empty(len(x_query), dtype=np.float64)
    if len(excluded_episodes) != len(x_query): raise ValueError("Query episode shape mismatch")
    for group in np.unique(excluded_episodes):
        fit, query = episodes_train != group, excluded_episodes == group
        w, b = centroid_fit(x_train[fit], labels_train[fit])
        result[query] = np.asarray(x_query[query], dtype=np.float64) @ w + b
    return result


def nearest_for_query(reference, query, labels, episodes, sample_ids, query_episode, query_sample, exclude_episode):
    """One physical-node reference pool; labels only define diagnostic neighbors."""
    squared = np.maximum(np.einsum("ij,ij->i", reference - query, reference - query), 0.)
    eligible = episodes != query_episode if exclude_episode else np.ones(len(reference), dtype=bool)
    nonself = ~((episodes == query_episode) & (sample_ids == query_sample))

    def nearest(mask):
        positions = np.flatnonzero(mask)
        if not len(positions): return None
        index = int(positions[np.argmin(squared[positions])])
        return dict(sample_index=int(sample_ids[index]), episode=int(episodes[index]),
                    label=int(labels[index]), distance=float(np.sqrt(squared[index])))

    return dict(nearest_training_nonself=nearest(nonself),
        nearest_eligible_positive=nearest(eligible & nonself & (labels == 1)),
        nearest_eligible_negative=nearest(eligible & nonself & (labels == 0)))


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""): h.update(b)
    return h.hexdigest()


def main():
    from factory_baselines.metrics import _binary_metrics

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_commit", required=True)
    parser.add_argument("--model", choices=("b4", "b5"), required=True)
    args = parser.parse_args()
    repo = Path.cwd().resolve(); assert repo == Path("/home/sci/work/BSTAN_isaac_factory")
    d = repo / "source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/output/bottleneck_dataset/experiments/factory_pdformer_134_v3"
    output = d / f"baseline_repr_transfer_{args.model}s42_last20260913.json"
    assert not output.exists()
    source = subprocess.check_output(["git", "show", args.source_commit + ":" + SOURCE_PATH])
    audit_path = d / "baseline_schedule_strata_final_verification20260913.json"
    assert sha(audit_path) == "c1837f569cb50023ec8647a473440ca406b96be61b360463f9a174db9c2ce340"
    audit = json.loads(audit_path.read_text())
    export_audit_path = d / "baseline_repr_exports_verification20260913.json"
    export_audit_hash = "d9971652a72fc24eb6388482e9c28382fcab226b2d0e76f8f7bd96f269e35f3d"
    assert sha(export_audit_path) == export_audit_hash
    export_audit = json.loads(export_audit_path.read_text())

    def guard():
        assert subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip() == RUNTIME
        assert subprocess.check_output(["git", "branch", "--show-current"], text=True).strip() == "dev_xwt"
        assert not subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], text=True).strip()
        for name, h in audit["current_model_files_sha256"].items(): assert sha(d / name) == h
        for name, value in audit["dataset_files_stat"].items():
            stat = (d / name).stat(); assert dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns) == value

    guard()
    with (d / "model_sample_index.csv").open(newline="") as f:
        samples = {int(r["sample_index"]): r for r in csv.DictReader(f) if r["split"] in ("train", "validation")}
    group_ids = sorted({r["group_id"] for r in samples.values()}); group_number = {g: i for i, g in enumerate(group_ids)}
    manifest = json.loads((d / "dataset_manifest.json").read_text())
    data, provenance = {}, []
    for split in ("train", "validation"):
        stem = f"baseline_repr_{args.model}s42_last_{split}20260913"
        path, summary = d / (stem + ".npz"), d / (stem + ".json")
        description = json.loads(summary.read_text()); assert sha(path) == description["archive_sha256"]
        verified = next(row for row in export_audit["archives"] if row["model"] == args.model and row["split"] == split)
        assert (path.name, sha(path), sha(summary)) == (verified["file"], verified["sha256"], verified["summary_sha256"])
        with np.load(path, allow_pickle=False) as stored:
            meta = json.loads(str(stored["metadata_json"].item())); assert meta == description["metadata"]
            assert meta["source_commit"] == EXPORT_COMMIT and meta["source_sha256"] == EXPORT_SHA
            assert (meta["model"], meta["seed"], meta["checkpoint"], meta["split"]) == (args.model, 42, "last", split)
            assert sha(d / meta["original_file"]) == meta["original_file_sha256"]
            mask = stored["label_kind"] != 1
            table = {key: stored[key][mask] for key in ("sample_index", "node_index", "label_kind", "probability", "raw_summary", "backbone", "event_input")}
        assert all(samples[int(index)]["split"] == split for index in table["sample_index"])
        table["episode"] = np.array([group_number[samples[int(i)]["group_id"]] for i in table["sample_index"]])
        table["label"] = (table["label_kind"] == 2).astype(np.int8)
        assert (int(table["label"].sum()), int((table["label"] == 0).sum())) == ((595, 297152) if split == "train" else (145, 67331))
        data[split] = table
        provenance.append(dict(file=path.name, bytes=path.stat().st_size, sha256=description["archive_sha256"], summary_file=summary.name, summary_sha256=sha(summary), checkpoint_sha256=meta["checkpoint_file_sha256"]))
    train, validation = data["train"], data["validation"]
    ranking, neighbors = [], []
    for view in ("raw_summary", "backbone", "event_input"):
        x = train[view]
        for evaluation, query, excluded in (("train_in_sample", train, None), ("train_reference_episode_excluded", train, train["episode"]), ("validation_train_only", validation, None)):
            scores = centroid_scores(x, train["label"], train["episode"], query[view], excluded)
            metrics = _binary_metrics(query["label"], scores)
            ranking.append(dict(view=view, evaluation=evaluation, average_precision=metrics["pr_auc"], roc_auc=metrics["roc_auc"],
                positive_count=metrics["positive_count"], negative_count=metrics["negative_count"],
                positive_score_q10_q50_q90=np.quantile(scores[query["label"] == 1], [.1, .5, .9]).tolist()))
            print("CENTROID", args.model, view, evaluation, metrics["pr_auc"], flush=True)
        # Distances use one label-independent all-train scale and preserve node ID.
        mean = np.asarray(x, dtype=np.float64).mean(0); variance = np.asarray(x, dtype=np.float64).var(0)
        scale = np.sqrt(np.where(variance < 1e-12, 1., variance))
        for split, query in data.items():
            by_node = defaultdict(list)
            for position in np.flatnonzero(query["label"] == 1): by_node[int(query["node_index"][position])].append(int(position))
            for node, positions in by_node.items():
                reference_mask = train["node_index"] == node
                reference = (np.asarray(x[reference_mask], dtype=np.float64) - mean) / scale
                for position in positions:
                    index, episode = int(query["sample_index"][position]), int(query["episode"][position])
                    result = nearest_for_query(reference, (query[view][position] - mean) / scale,
                        train["label"][reference_mask], train["episode"][reference_mask], train["sample_index"][reference_mask], episode, index, split == "train")
                    for name in ("nearest_eligible_positive", "nearest_eligible_negative"):
                        if result[name] is not None and split == "train": assert result[name]["episode"] != episode
                    neighbors.append(dict(view=view, split=split, sample_index=index, episode=episode,
                        group_id=group_ids[episode], node_index=node, resource_id=manifest["node_ids"][node],
                        original_probability=float(query["probability"][position]), **result))
            print("NEIGHBORS", args.model, view, split, len(by_node), flush=True)
    summaries = []
    for view in ("raw_summary", "backbone", "event_input"):
        for split in ("train", "validation"):
            rows = [r for r in neighbors if r["view"] == view and r["split"] == split]
            complete = [r for r in rows if r["nearest_eligible_positive"] is not None and r["nearest_eligible_negative"] is not None]
            margin = [r["nearest_eligible_negative"]["distance"] - r["nearest_eligible_positive"]["distance"] for r in complete]
            summaries.append(dict(view=view, split=split, upcoming_count=len(rows), with_positive_and_negative_reference=len(complete),
                nearest_nonself_same_episode=sum(r["nearest_training_nonself"] is not None and r["nearest_training_nonself"]["episode"] == r["episode"] for r in rows),
                cross_episode_positive_closer=sum(v > 0 for v in margin), equidistant=sum(v == 0 for v in margin),
                negative_minus_positive_distance_q10_q50_q90=np.quantile(margin, [.1, .5, .9]).tolist() if margin else None))
    for p in provenance: assert sha(d / p["file"]) == p["sha256"]
    guard()
    record = dict(status="fixed_centroid_and_neighbor_analysis_completed_without_model_forward", source_commit=args.source_commit,
        source_sha256=hashlib.sha256(source).hexdigest(), model=args.model, seed=42, checkpoint="last", runtime_commit=RUNTIME,
        export_verification_sha256=export_audit_hash,
        archives=provenance, centroid_ranking=ranking, nearest_neighbor_summary=summaries, upcoming_neighbors=neighbors,
        episode_ids=group_ids, variance_floor=1e-12,
        limitations=["Excluding an episode from diagnostic reference statistics does not remove it from the training of frozen neural weights.",
            "Raw summaries are lossy 84-dimensional summaries, not the entire observed 30-frame input.",
            "Centroid failure is not proof that a nonlinear head cannot use the representation.",
            "Nearest-neighbor distance is descriptive, not a causal attribution or new alarm filter.",
            "Window targets overlap; no independent-event confidence interval is asserted.",
            "B4 joint_onset and B5 vector_gat are distinct negative ablations, not a backbone-isolated comparison."],
        diagnostic_statistical_fit=True, model_training=False, model_forward=False, test_evaluated=False, goal_met=False)
    with output.open("x") as f: json.dump(record, f, indent=2); f.write("\n")
    print("REPRESENTATION_TRANSFER_COMPLETE", args.model, output.stat().st_size, sha(output), flush=True)
    print(json.dumps(summaries), flush=True)


if __name__ == "__main__": main()
