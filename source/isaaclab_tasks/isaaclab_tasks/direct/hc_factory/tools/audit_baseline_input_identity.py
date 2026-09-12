#!/usr/bin/env python3
"""Audit exact observed-input identity and conflicting train/validation targets."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import torch

from factory_baselines.dataset import FactoryBaselineTensorDataset, load_shared_dataset
from factory_baselines.precursor import attach_precursor
from factory_baselines.torch_trainer import _model_inputs


SPLITS = ("train", "validation")
FROZEN_MANIFEST = "e3d7b2008ad7c5d0844c10a4c0670ff36c5ba961382706695689daf7a050244f"
SOURCE_PATH = "source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/tools/audit_baseline_input_identity.py"
VIEWS = ("full_adapter", "event_context_disabled")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def observed_inputs(sample: dict[str, torch.Tensor], view: str = "full_adapter") -> dict[str, torch.Tensor]:
    """Use the actual model adapter, plus the observed state used by report decoding."""
    if view not in VIEWS:
        raise ValueError(f"Unknown observation view: {view}")
    inputs = {**_model_inputs(sample), "hist_last_hot": sample["hist_last_hot"]}
    if view == "event_context_disabled":
        # Both B4/B5 ignore target_node_mask. Without event_context_projection,
        # global features and job counts feed other heads, not event predictions.
        for key in ("target_node_mask", "global_features", "jobs_remaining", "jobs_total"):
            del inputs[key]
    return inputs


def input_fingerprint(sample: dict[str, torch.Tensor], view: str = "full_adapter") -> str:
    digest = hashlib.sha256(b"factory_baseline_observed_input_v2\0" + view.encode() + b"\0")
    for key, tensor in sorted(observed_inputs(sample, view).items()):
        value = tensor.detach().cpu().contiguous().numpy().copy()
        if not np.isfinite(value).all():
            raise ValueError(f"Non-finite observed input: {key}")
        value[value == 0] = 0  # +0 and -0 carry the same observed value.
        header = json.dumps([key, value.dtype.str, list(value.shape)], separators=(",", ":")).encode()
        data = value.tobytes(order="C")
        digest.update(len(header).to_bytes(8, "little"))
        digest.update(header)
        digest.update(len(data).to_bytes(8, "little"))
        digest.update(data)
    return digest.hexdigest()


def equal_observations(first: dict, second: dict, view: str = "full_adapter") -> bool:
    first, second = observed_inputs(first, view), observed_inputs(second, view)
    return first.keys() == second.keys() and all(
        first[key].dtype == second[key].dtype and torch.equal(first[key], second[key])
        for key in first
    )


def _summarize_groups(groups: dict, splits: tuple[str, ...]) -> dict:
    totals = Counter()
    examples = []
    for fingerprint, group in groups.items():
        members = [m for m in group["members"] if m["split"] in splits]
        if not members:
            continue
        counts = sum((group["counts"][s] for s in splits if s in group["counts"]),
                     np.zeros_like(next(iter(group["counts"].values()))))
        valid, positive, upcoming = counts
        negative, ongoing = valid - positive, positive - upcoming
        conflict = (positive > 0) & (negative > 0)
        up_conflict = (upcoming > 0) & (negative > 0)
        totals.update(samples=len(members), input_groups=1,
                      ongoing_targets=int(ongoing.sum()), upcoming_targets=int(upcoming.sum()),
                      negative_targets=int(negative.sum()),
                      positive_negative_conflict_node_groups=int(conflict.sum()),
                      positive_targets_with_negative_counterexample=int(positive[conflict].sum()),
                      upcoming_targets_with_negative_counterexample=int(upcoming[up_conflict].sum()),
                      positive_subtype_conflict_node_groups=int(((upcoming > 0) & (ongoing > 0)).sum()))
        if len(members) > 1:
            totals.update(repeated_input_groups=1, samples_in_repeated_input_groups=len(members),
                          cross_episode_repeated_groups=int(len({m["episode_group_id"] for m in members}) > 1))
        if conflict.any() and len(examples) < 24:
            examples.append(dict(input_sha256=fingerprint, members=members, nodes=[
                dict(node_index=int(node), valid=int(valid[node]), positive=int(positive[node]),
                     upcoming=int(upcoming[node]), negative=int(negative[node]))
                for node in np.flatnonzero(conflict)
            ]))
    for key in ("samples", "input_groups", "ongoing_targets", "upcoming_targets", "negative_targets",
                "positive_negative_conflict_node_groups", "positive_targets_with_negative_counterexample",
                "upcoming_targets_with_negative_counterexample", "positive_subtype_conflict_node_groups",
                "repeated_input_groups", "samples_in_repeated_input_groups", "cross_episode_repeated_groups"):
        totals.setdefault(key, 0)
    return {**dict(totals), "conflict_examples": examples,
            "conflict_example_group_limit": 24}


def audit_dataset(dataset, split_by_index: dict[int, str], view: str = "full_adapter") -> dict:
    """Exact groups over complete sample observations, retaining node identities.

    Dataset targets only count disagreements after hashing. A matching hash is
    additionally checked against its representative tensors, so a hash collision
    cannot be treated as evidence that observations are equal.
    """
    if set(split_by_index.values()) - set(SPLITS):
        raise ValueError("Only train and validation may be audited")
    if view not in VIEWS:
        raise ValueError(f"Unknown observation view: {view}")
    groups, seen = {}, set()
    index_digest = hashlib.sha256()
    schema = None
    for position in range(len(dataset)):
        sample = dataset[position]
        index = int(sample["sample_index"])
        if index not in split_by_index or index in seen:
            raise ValueError("Duplicate or unexpected sample index")
        seen.add(index)
        split = split_by_index[index]
        current_schema = {k: [str(v.dtype), list(v.shape)] for k, v in observed_inputs(sample, view).items()}
        if schema is not None and current_schema != schema:
            raise ValueError("Observed input schema changes across samples")
        schema = current_schema
        fingerprint = input_fingerprint(sample, view)
        index_digest.update(json.dumps([index, split, fingerprint], separators=(",", ":")).encode() + b"\n")
        valid = sample["occ_node_mask"].numpy() > .5
        will = sample["event_will"].numpy()
        start = sample["event_start"].numpy()
        if will.shape != valid.shape or start.shape != valid.shape or valid.ndim != 1:
            raise ValueError("Expected matching node target grids")
        if not np.isfinite(will[valid]).all() or not np.isin(will[valid], [0, 1]).all():
            raise ValueError("Expected binary event labels on valid nodes")
        positive = valid & (will > .5)
        if not np.isfinite(start[positive]).all() or (start[positive] < 0).any():
            raise ValueError("Invalid positive event start")
        counts = np.stack([valid, positive, positive & (start > 0)]).astype(np.int64)
        if fingerprint in groups:
            group = groups[fingerprint]
            if not equal_observations(sample, dataset[group["representative_position"]], view):
                raise ValueError("Hash collision: observed tensors differ")
        else:
            group = groups[fingerprint] = dict(representative_position=position, members=[], counts={})
        group["members"].append(dict(sample_index=index, split=split,
                                     episode_group_id=int(sample["sample_group_id"])))
        if split not in group["counts"]:
            group["counts"][split] = np.zeros_like(counts)
        group["counts"][split] += counts
    if seen != set(split_by_index):
        raise ValueError("Incomplete requested sample coverage")
    shared = Counter(input_groups=0, train_samples=0, validation_samples=0,
                     validation_upcoming_with_train_negative_counterexample=0,
                     validation_negative_with_train_upcoming_counterexample=0)
    for group in groups.values():
        if set(group["counts"]) != set(SPLITS):
            continue
        train, val = group["counts"]["train"], group["counts"]["validation"]
        shared.update(input_groups=1,
                      train_samples=sum(m["split"] == "train" for m in group["members"]),
                      validation_samples=sum(m["split"] == "validation" for m in group["members"]),
                      validation_upcoming_with_train_negative_counterexample=int(val[2][train[0] > train[1]].sum()),
                      validation_negative_with_train_upcoming_counterexample=int((val[0]-val[1])[train[2] > 0].sum()))
    return dict(observation_view=view, observed_input_schema=schema, sample_input_index_sha256=index_digest.hexdigest(),
                splits={name: _summarize_groups(groups, (name,)) for name in SPLITS},
                train_validation_union=_summarize_groups(groups, SPLITS), cross_split=dict(shared),
                hash_matches_verified_against_observed_tensors=True,
                note="Exact full-window observations, fixed node identities; not local-node similarity, "
                     "permutation symmetry or approximate ambiguity. Cross-split conflicts are supervision "
                     "diagnostics, not a bound on validation performance. Targets never enter the input hash.")


def audit(dataset_dir: Path) -> dict:
    manifest_path = dataset_dir / "dataset_manifest.json"
    if sha256(manifest_path) != FROZEN_MANIFEST:
        raise ValueError("This audit is registered for the unchanged 208-episode manifest")
    files = ["dataset_manifest.json", "dataset.pt", "model_sample_index.csv", "split_manifest.json", "normalization.json"]
    before = {name: ((dataset_dir/name).stat().st_size, (dataset_dir/name).stat().st_mtime_ns) for name in files}
    hashes = {name: sha256(dataset_dir/name) for name in files}
    payload, manifest = load_shared_dataset(dataset_dir)
    if "event_precursor" in payload:
        raise ValueError("Expected the unmodified shared tensor bundle before optional precursor attachment")
    mapping = {int(i): name for name in SPLITS for i in payload["split_indices"][name].tolist()}
    expected = sum(len(payload["split_indices"][name]) for name in SPLITS)
    if len(mapping) != expected:
        raise ValueError("Train/validation sample index overlap")
    indices = list(mapping)
    core_views = {}
    dataset = FactoryBaselineTensorDataset(payload, indices)
    for view in VIEWS:
        result = audit_dataset(dataset, mapping, view)
        core_views[view] = result
        print(view, "core:", json.dumps({k: v for k, v in result["train_validation_union"].items()
                                        if k != "conflict_examples"}), flush=True)
    modes = {}
    repeated_views = [view for view in VIEWS
                      if core_views[view]["train_validation_union"]["repeated_input_groups"] > 0]
    for mode in ("near", "near_far"):
        views = {view: dict(status="exact_duplicates_ruled_out_by_unique_core_subset",
                            reasoning="Adding precursor fields cannot merge distinct core observations")
                 for view in VIEWS if view not in repeated_views}
        modes[mode] = dict(precursor_built=bool(repeated_views), views=views)
        if repeated_views:
            augmented, contract = attach_precursor(payload, manifest, dataset_dir, mode, SPLITS)
            modes[mode]["input_feature_contract"] = contract
            for view in repeated_views:
                result = audit_dataset(FactoryBaselineTensorDataset(augmented, indices), mapping, view)
                views[view] = dict(status="full_input_identity_audited", result=result)
            print(mode, "full identity audit complete", flush=True)
    after = {name: ((dataset_dir/name).stat().st_size, (dataset_dir/name).stat().st_mtime_ns) for name in files}
    if before != after or sha256(manifest_path) != FROZEN_MANIFEST:
        raise ValueError("Dataset files changed during the audit")
    return dict(status="completed_exact_observed_input_identity_audit", dataset_files_sha256=hashes,
                source_dataset_version=manifest["dataset_version"], evaluation_contract=manifest["evaluation_contract"],
                splits_examined=list(SPLITS), shared_tensor_bundle_loaded=True, test_samples_examined=False,
                labels_used_only_for_post_hash_conflict_counts=True, core=core_views["full_adapter"],
                event_branch_core=core_views["event_context_disabled"], augmented_modes=modes,
                event_branch_scope="B4/B5 with event_context=false, including near/onset/far/timeattention arms. "
                                   "All graph history/adjacency/node_mask, optional precursor and decoded history hot. "
                                   "Excludes ignored target_node_mask and global/job fields used only by other heads. "
                                   "Does not collapse masked values, graph symmetries or learned representation equivalences.",
                dataset_file_size_mtime_unchanged=True, training_started=False, thresholds_selected=False,
                interpretation="Exact duplicates/conflicts only. Absence of duplicates does not establish "
                               "predictability, successful generalization or sufficient model capacity; "
                               "distinct histories can still be approximately ambiguous or lack future exogenous information.")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source_commit", help="Pinned Git source when executing via stdin")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not args.output.parent.is_dir():
        raise FileNotFoundError("Reuse an existing output directory")
    torch.set_num_threads(2)
    source = (subprocess.check_output(["git", "show", f"{args.source_commit}:{SOURCE_PATH}"])
              if args.source_commit else Path(__file__).read_bytes())
    result = audit(args.dataset_dir)
    import factory_baselines.dataset as dataset_module
    import factory_baselines.precursor as precursor_module
    import factory_baselines.torch_trainer as trainer_module
    import factory_baselines.b4_gcn_gru as b4_module
    import factory_baselines.b5_gat_gru as b5_module
    import factory_baselines.torch_heads as heads_module
    result.update(audit_source_commit=args.source_commit, audit_source_sha256=hashlib.sha256(source).hexdigest(),
                  runtime_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  runtime_source_sha256={module.__name__: sha256(Path(module.__file__))
                                         for module in (dataset_module, precursor_module, trainer_module,
                                                        b4_module, b5_module, heads_module)})
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(f"Output: {args.output}", flush=True)


if __name__ == "__main__":
    main()
