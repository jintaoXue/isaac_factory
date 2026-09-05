#!/usr/bin/env python3
"""Check exact node-swap symmetries, not an estimated global performance ceiling."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import torch

from factory_baselines.dataset import FactoryBaselineTensorDataset, load_shared_dataset


def indistinguishable_pair(x, adjacency, node_mask, first, second):
    permutation = torch.arange(adjacency.shape[-1])
    permutation[first], permutation[second] = second, first
    swapped = adjacency[:, permutation][:, :, permutation]
    graph_equal = (adjacency == swapped).all(dim=(1, 2))
    feature_equal = (x[:, :, first] == x[:, :, second]).all(dim=(1, 2))
    return graph_equal & feature_equal & node_mask[:, first] & node_mask[:, second]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation"), default="train")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(4)
    payload, manifest = load_shared_dataset(args.dataset_dir)
    indices = payload["split_indices"][args.split]
    dataset = FactoryBaselineTensorDataset(payload, indices)
    active = payload["occ_node_mask"][indices].bool()
    eligible = torch.nonzero(active.any(0)).flatten().tolist()
    pairs = list(itertools.combinations(eligible, 2))
    matches = {pair: [] for pair in pairs}
    for offset in range(0, len(indices), 128):
        subset = indices[offset:offset + 128]
        x, graph = payload["x"][subset], payload["adjacency"][subset]
        mask = active[offset:offset + len(subset)]
        for pair in pairs:
            found = indistinguishable_pair(x, graph, mask, *pair)
            matches[pair].extend((torch.nonzero(found).flatten() + offset).tolist())
    relevant = sorted({sample for group in matches.values() for sample in group})
    labels = {sample: dataset[sample]["event_will"] for sample in relevant}
    rows = []
    for (first, second), samples in matches.items():
        if not samples:
            continue
        rows.append({
            "nodes": [manifest["node_ids"][first], manifest["node_ids"][second]],
            "identical_history_and_graph_samples": len(samples),
            "different_event_labels": sum(
                int(labels[sample][first] != labels[sample][second]) for sample in samples
            ),
        })
    result = {
        "split": args.split, "test_evaluated": False, "sample_count": len(indices),
        "samples_with_exact_node_swap_symmetry": len(relevant), "pairs": rows,
        "interpretation": "Pair counts may overlap; do not sum as a global error lower bound. "
        "This checks exact transpositions only, not all graph symmetries or approximate ambiguity.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
