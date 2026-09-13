#!/usr/bin/env python3
"""Fixed physical-state rate probes with whole-training-episode exclusion.

This diagnoses a specific observed representation; it does not train B4/B5,
select checkpoints/thresholds, or establish an upper bound on their inputs.
"""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np


SOURCE_PATH = "source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/tools/diagnose_upcoming_state_transfer.py"
RUNTIME = "979c680fb4d1919ae760cdfb0038f69fb7cc6708"
MAIN_REFERENCE = "20c40e230aedee6aef2429d352413fbcf0fa571a"
MAIN_RULE_PATH = "source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/PDFormer/factory_bn/cause_cluster.py"
MANIFEST = "e3d7b2008ad7c5d0844c10a4c0670ff36c5ba961382706695689daf7a050244f"
VIEWS = ("node", "state", "node_state", "node_state_transition")


def physical_states(features):
    """Pointwise 60-second gates copied from the pinned main seed_cluster_ids.

    No temporal smoothing, future window, event labels, or fitted threshold.
    """
    x = np.asarray(features, dtype=np.float32)
    if x.shape[-1] != 27 or not np.isfinite(x).all():
        raise ValueError("Expected finite 27-channel observed features")
    out = np.full(x.shape[:-1], 6, dtype=np.int64)
    out[x[..., 4] < .8] = 0
    out[x[..., 7] > 0] = 4
    out[(x[..., 6] >= x[..., 7]) & (x[..., 6] > 24)] = 3
    out[(x[..., 0] >= 1) | (x[..., 1] >= 20) | (x[..., 16] >= .7)] = 5
    out[(x[..., 14] >= 20) | (x[..., 13] >= 20)] = 1
    out[x[..., 15] >= .25] = 2
    return out


def observed_states(x_last_two, mean, std, observed):
    raw = np.asarray(x_last_two, dtype=np.float32).copy()
    if raw.ndim != 3 or raw.shape[0] != 2 or raw.shape[-1] != 27:
        raise ValueError("Use exactly the two last observed windows")
    raw[..., :21] = raw[..., :21] * std + mean
    state = physical_states(raw)
    if observed.shape != state.shape:
        raise ValueError("Observation mask differs from state shape")
    return np.where(observed > .5, state, 7)


def keys_for(view, node, previous, state):
    if view == "node": return node
    if view == "state": return state
    if view == "node_state": return node * 8 + state
    if view == "node_state_transition": return node * 64 + previous * 8 + state
    raise ValueError(view)


def rate_probe(train_key, train_label, train_episode, query_key, query_episode=None):
    """One fixed pseudo-observation at training prevalence, excluding whole episodes.

    query_episode=None fits all train; otherwise every fitting statistic,
    including the fallback prevalence, excludes that training episode.
    No query label is accepted by this API.
    """
    k, y, ep, q = (np.asarray(v, dtype=np.int64) for v in (train_key, train_label, train_episode, query_key))
    if not len(k) or len(k) != len(y) or len(k) != len(ep) or not np.isin(y, [0, 1]).all():
        raise ValueError("Invalid training observations")
    if min(k.min(), q.min(), ep.min()) < 0: raise ValueError("Negative key/episode")
    width = int(max(k.max(), q.max())) + 1
    count, positive = np.bincount(k, minlength=width), np.bincount(k, weights=y, minlength=width)
    n, p = count[q].astype(float), positive[q].copy()
    total_n, total_p = np.full(len(q), len(y), dtype=float), np.full(len(q), y.sum(), dtype=float)
    if query_episode is not None:
        qe = np.asarray(query_episode, dtype=np.int64)
        if qe.shape != q.shape or not np.isin(qe, ep).all(): raise ValueError("Unknown excluded episode")
        for group in np.unique(qe):
            fit, select = ep == group, qe == group
            n[select] -= np.bincount(k[fit], minlength=width)[q[select]]
            p[select] -= np.bincount(k[fit], weights=y[fit], minlength=width)[q[select]]
            total_n[select] -= fit.sum(); total_p[select] -= y[fit].sum()
    if (total_n <= 0).any() or (n < 0).any() or (p < 0).any(): raise ValueError("Empty/invalid fitting population")
    scores = (p + total_p / total_n) / (n + 1)
    return scores, n.astype(np.int64), p.astype(np.int64)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""): h.update(b)
    return h.hexdigest()


def main():
    import torch
    from factory_baselines.dataset import FactoryBaselineTensorDataset
    from factory_baselines.metrics import _binary_metrics

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_commit", required=True)
    parser.add_argument("--dataset_dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = Path.cwd().resolve(); d = args.dataset_dir.resolve(); output = args.output.resolve()
    assert repo == Path("/home/sci/work/BSTAN_isaac_factory") and d.is_relative_to(repo)
    assert output.parent == d and d.is_dir() and not output.exists()
    audit_file = d / "baseline_schedule_strata_final_verification20260913.json"
    assert sha(audit_file) == "c1837f569cb50023ec8647a473440ca406b96be61b360463f9a174db9c2ce340"
    audit = json.loads(audit_file.read_text())

    def guard():
        assert subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip() == RUNTIME
        assert subprocess.check_output(["git", "branch", "--show-current"], text=True).strip() == "dev_xwt"
        assert not subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], text=True).strip()
        for name, h in audit["current_model_files_sha256"].items(): assert sha(d / name) == h
        for name, value in audit["dataset_files_stat"].items():
            stat = (d / name).stat(); assert dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns) == value

    guard(); assert sha(d / "dataset_manifest.json") == MANIFEST
    source = subprocess.check_output(["git", "show", args.source_commit + ":" + SOURCE_PATH])
    reference = subprocess.check_output(["git", "show", MAIN_REFERENCE + ":" + MAIN_RULE_PATH])
    namespace = {}; exec(compile(reference, MAIN_RULE_PATH, "exec"), namespace)
    manifest = json.loads((d / "dataset_manifest.json").read_text())
    norm = json.loads((d / "normalization.json").read_text())
    mean, std = (np.asarray(norm[k], dtype=np.float32) for k in ("feature_mean", "feature_std"))
    assert mean.shape == std.shape == (21,) and (std > 0).all()
    payload = torch.load(d / "dataset.pt", map_location="cpu", weights_only=True, mmap=True)
    with (d / "model_sample_index.csv").open(newline="") as f:
        samples = [r for r in csv.DictReader(f) if r["split"] in ("train", "validation")]
    groups = defaultdict(list)
    for row in samples: groups[row["group_id"]].append(row)
    train_groups = sorted(g for g, rows in groups.items() if rows[0]["split"] == "train")
    group_number = {g: i for i, g in enumerate(sorted(groups))}
    assert len(groups) == 168 and len(train_groups) == 138
    sources = {r["group_id"]: r for r in manifest["source_episodes"]}
    support = json.loads((d / "baseline_upcoming_schedule_support20260913.json").read_text())
    assert sha(d / "baseline_upcoming_schedule_support20260913.json") == "ebeead2dff053e3b8041ad563d3e95519c5a58b3b91c4006b0586cc6e3d86d02"
    expected_up = {(r["group_id"], r["resource_id"], r["onset_window_index"], r["start_index"]) for r in support["upcoming_targets"]}
    seen_up, seen_samples = set(), set()
    columns = defaultdict(list); differences = Counter(); label_counts = defaultdict(Counter)
    with np.load(d / "episodes.npz", allow_pickle=False) as bundle:
        ids = bundle["resource_ids"].tolist(); order = [ids.index(node) for node in manifest["node_ids"]]
        for group, rows in sorted(groups.items()):
            name = sources[group]["main_episode_name"]
            features = bundle[name + "_features"][:, order]
            windows = bundle[name + "_windows"]; positions = {int(w): i for i, w in enumerate(windows)}
            raw_state = physical_states(features)
            assert np.array_equal(raw_state, namespace["seed_cluster_ids"](features, window_size_s=60.))
            dataset = FactoryBaselineTensorDataset(payload, [int(r["sample_index"]) for r in rows])
            for row, sample in zip(rows, dataset):
                index = int(sample["sample_index"]); assert index == int(row["sample_index"]) and index not in seen_samples
                seen_samples.add(index)
                split = row["split"]; anchor = positions[int(row["anchor_window_index"])]
                assert anchor >= 1 and float(row["first_future_start_s"]) == (int(windows[anchor]) + 1) * 60
                valid = sample["occ_node_mask"].numpy() > .5
                will = sample["event_will"].numpy() > .5; start = sample["event_start"].numpy().astype(int)
                upcoming = valid & will & (start > 0); negative = valid & ~will; selected = upcoming | negative
                label_counts[split].update(upcoming=int(upcoming.sum()), negative=int(negative.sum()), ongoing=int((valid & will & (start == 0)).sum()), samples=1)
                observed = sample["observation_mask"][-2:].numpy() * sample["node_mask"].numpy()[None, :]
                inv = observed_states(sample["x"][-2:].numpy(), mean, std, observed)
                raw = raw_state[anchor-1:anchor+1]
                for kind, mask in [("upcoming", upcoming), ("negative", negative)]:
                    differences[split + "_" + kind + "_last_state_disagrees"] += int(((inv[-1] != raw[-1]) & mask).sum())
                    differences[split + "_" + kind + "_either_state_disagrees"] += int(((inv != raw).any(axis=0) & mask).sum())
                    differences[split + "_" + kind + "_missing_last_observation"] += int(((observed[-1] <= .5) & mask).sum())
                for node in np.flatnonzero(upcoming):
                    key = (group, manifest["node_ids"][node], int(windows[anchor]) + 1 + int(start[node]), int(start[node]))
                    assert key in expected_up and key not in seen_up; seen_up.add(key)
                nodes = np.flatnonzero(selected); count = len(nodes)
                for key, value in dict(label=upcoming[nodes].astype(int), node=nodes, previous=inv[0, nodes], state=inv[1, nodes], raw_previous=raw[0, nodes], raw_state=raw[1, nodes], episode=np.full(count, group_number[group]), validation=np.full(count, split == "validation")).items():
                    columns[key].append(value)
            print("STATE_EPISODE", group, "samples", len(rows), flush=True)
    assert seen_up == expected_up
    expected_counts = {"train": dict(upcoming=595, negative=297152, ongoing=4191, samples=23859), "validation": dict(upcoming=145, negative=67331, ongoing=950, samples=5439)}
    assert {k: dict(v) for k, v in label_counts.items()} == expected_counts
    assert seen_samples == {int(i) for split in ("train", "validation") for i in payload["split_indices"][split]}
    data = {k: np.concatenate(v) for k, v in columns.items()}; train = ~data["validation"].astype(bool); val = ~train
    results = []
    for representation, previous, current in (("inverted_baseline_X", "previous", "state"), ("raw_bundle_reference", "raw_previous", "raw_state")):
        for view in VIEWS:
            keys = keys_for(view, data["node"], data[previous], data[current])
            for mode, selected, excluded in (("train_in_sample", train, None), ("train_leave_episode_out", train, data["episode"][train]), ("validation_train_only", val, None)):
                scores, n, p = rate_probe(keys[train], data["label"][train], data["episode"][train], keys[selected], excluded)
                labels = data["label"][selected]; metrics = _binary_metrics(labels, scores)
                results.append(dict(representation=representation, view=view, evaluation=mode,
                    positive_count=int(labels.sum()), negative_count=int((labels == 0).sum()),
                    average_precision=metrics["pr_auc"], roc_auc=metrics["roc_auc"], prevalence=float(labels.mean()),
                    positive_without_fitting_cell_observations=int(((n == 0) & (labels > 0)).sum()),
                    positive_without_fitting_cell_positive=int(((p == 0) & (labels > 0)).sum()),
                    positive_score_q10_q50_q90=np.quantile(scores[labels > 0], [.1, .5, .9]).tolist()))
    guard()
    record = dict(status="fixed_state_rate_and_whole_episode_exclusion_diagnostic_completed", source_commit=args.source_commit,
        source_sha256=hashlib.sha256(source).hexdigest(), runtime_commit=RUNTIME,
        main_rule_reference_commit=MAIN_REFERENCE, main_rule_sha256=hashlib.sha256(reference).hexdigest(),
        source_audit_sha256=sha(audit_file), label_counts=expected_counts, raw_reference_vs_inverted_X=dict(differences), results=results,
        fixed_pseudo_observation_count=1, fitting="training-only empirical cell rates with training-prevalence prior; leave-episode-out excludes the entire episode including from the prior",
        scope="Fixed diagnostic statistical probes, no B4/B5 weight updates, checkpoint or threshold selection, future input, or test evaluation.",
        limitations=["A coarse state probe is not an input-information or architecture upper bound.", "Training leave-episode-out uses the existing all-training normalization, not newly refitted normalization per fold.", "Raw bundle reference may differ from actual baseline X after masking and float inversion; both views are reported separately.", "Window targets overlap; AP is descriptive and is not an independent-event confidence interval.", "The pinned main rule is not proof of the components used by the historical formal prefix8 checkpoint."],
        model_training=False, diagnostic_statistical_fit=True, test_evaluated=False, main_repository_modified=False, goal_met=False)
    with output.open("x") as f: json.dump(record, f, indent=2); f.write("\n")
    print("STATE_TRANSFER_COMPLETE", output.stat().st_size, sha(output), flush=True)
    for row in results: print(row["representation"], row["view"], row["evaluation"], "AP", row["average_precision"], "unsupported_up", row["positive_without_fitting_cell_positive"], flush=True)


if __name__ == "__main__": main()
