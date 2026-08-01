"""Build and validate fixed-shape BSTAN graph-sequence datasets."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch
from torch.utils.data import Dataset

from .graph_builder import build_static_graph
from .schema import (
    CONTINUOUS_FEATURES,
    DATASET_VERSION,
    GLOBAL_FEATURES,
    LABEL_VERSION,
    feature_is_applicable,
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _f(value: Any, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _json_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _i(value: Any, default: int = 0) -> int:
    if value in (None, ""):
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _group_id(run_id: str, env_id: int, episode_id: int) -> str:
    return f"{run_id}:env_{env_id:02d}:episode_{episode_id:02d}"


def _stable_group_number(group_id: str) -> int:
    return int(hashlib.sha1(group_id.encode("utf-8")).hexdigest()[:15], 16)


def _git_commit(repo_root: Path | None) -> str:
    if repo_root is None:
        return "unknown"
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _discover_groups(
    run_dirs: Iterable[Path], derived_dir_name: str
) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    for run_dir in (Path(path).resolve() for path in run_dirs):
        derived_root = run_dir / derived_dir_name
        for feature_path in sorted(
            derived_root.glob("episode_*/env_*/window_feature_table.csv")
        ):
            rel = feature_path.parent.relative_to(derived_root)
            raw_dir = run_dir / rel
            label_path = feature_path.parent / "bottleneck_label.csv"
            config_rows = _read_csv(raw_dir / "episode_config.csv")
            if not config_rows:
                raise ValueError(f"Missing episode config row: {raw_dir}")
            config = config_rows[0]
            run_id = config.get("run_id") or run_dir.name
            env_id = _i(config.get("env_id"))
            episode_id = _i(config.get("episode_id"))
            groups.append(
                {
                    "group_id": _group_id(run_id, env_id, episode_id),
                    "group_number": _stable_group_number(
                        _group_id(run_id, env_id, episode_id)
                    ),
                    "run_dir": run_dir,
                    "raw_dir": raw_dir,
                    "derived_dir": feature_path.parent,
                    "run_id": run_id,
                    "env_id": env_id,
                    "episode_id": episode_id,
                    "scenario_id": config.get("scenario_id") or "unknown",
                    "collector_version": config.get("collector_version") or "unknown",
                    "config": config,
                    "feature_rows": _read_csv(feature_path),
                    "label_rows": _read_csv(label_path),
                }
            )
    if not groups:
        raise ValueError(
            f"No Phase-B tables found under derived directory {derived_dir_name!r}"
        )
    group_ids = [group["group_id"] for group in groups]
    if len(group_ids) != len(set(group_ids)):
        raise ValueError("Duplicate episode group across source run directories")
    return groups


def _filter_rows(
    rows: list[dict[str, str]], window_size: float, stride: float
) -> list[dict[str, str]]:
    return [
        row
        for row in rows
        if math.isclose(_f(row.get("window_size_s")), window_size)
        and math.isclose(_f(row.get("stride_s")), stride)
    ]


def _productive_machine_prefixes(config: dict[str, Any]) -> set[str]:
    prefixes: set[str] = set()
    for product_steps in _json_dict(config.get("process_time_config")).values():
        if not isinstance(product_steps, dict):
            continue
        for step_config in product_steps.values():
            if isinstance(step_config, dict) and step_config.get("machine"):
                prefixes.add(str(step_config["machine"]))
    return prefixes


def _included_node_ids(group: dict[str, Any]) -> set[str]:
    productive_machines = _productive_machine_prefixes(group["config"])
    included = set()
    for row in group["feature_rows"]:
        node_id = row["resource_id"]
        if row.get("resource_type") != "machine" or any(
            node_id == prefix or node_id.startswith(f"{prefix}_ws")
            for prefix in productive_machines
        ):
            included.add(node_id)
    return included


def _build_node_catalog(
    groups: list[dict[str, Any]]
) -> tuple[list[str], dict[str, str]]:
    node_types: dict[str, str] = {}
    for group in groups:
        group["included_node_ids"] = _included_node_ids(group)
        for row in group["feature_rows"]:
            node_id = row["resource_id"]
            if node_id not in group["included_node_ids"]:
                continue
            resource_type = row.get("resource_type") or "unknown"
            previous = node_types.setdefault(node_id, resource_type)
            if previous != resource_type:
                raise ValueError(
                    f"Conflicting resource type for {node_id}: {previous} vs {resource_type}"
                )
    node_ids = sorted(node_types, key=lambda node_id: (node_types[node_id], node_id))
    return node_ids, node_types


def _target_split_sizes(n_groups: int) -> dict[str, int]:
    if n_groups < 3:
        raise ValueError(
            "At least 3 episode groups are required for train/validation/test"
        )
    validation = max(1, round(n_groups * 0.15))
    test = max(1, round(n_groups * 0.15))
    while validation + test >= n_groups:
        if validation >= test and validation > 1:
            validation -= 1
        elif test > 1:
            test -= 1
        else:
            break
    return {
        "train": n_groups - validation - test,
        "validation": validation,
        "test": test,
    }


def _split_groups(
    group_sample_indices: dict[str, list[int]],
    group_scenarios: dict[str, str],
    y_occurrence: torch.Tensor,
    seed: int,
) -> dict[str, list[str]]:
    sizes = _target_split_sizes(len(group_sample_indices))
    rng = random.Random(seed)
    group_ids = sorted(group_sample_indices)
    rng.shuffle(group_ids)
    positive_unordered = [
        group_id
        for group_id in group_ids
        if bool(y_occurrence[group_sample_indices[group_id]].bool().any())
    ]
    non_positive = [
        group_id for group_id in group_ids if group_id not in positive_unordered
    ]
    positive_by_scenario: dict[str, list[str]] = defaultdict(list)
    for group_id in positive_unordered:
        positive_by_scenario[group_scenarios[group_id]].append(group_id)
    positive = []
    while any(positive_by_scenario.values()):
        for scenario_id in sorted(positive_by_scenario):
            if positive_by_scenario[scenario_id]:
                positive.append(positive_by_scenario[scenario_id].pop())
    if len(positive) < 3:
        raise ValueError(
            "At least three episode groups with positive labels are required "
            "so every split has positives"
        )

    result = {
        "train": [positive[2]],
        "validation": [positive[0]],
        "test": [positive[1]],
    }
    remaining = positive[3:] + non_positive
    rng.shuffle(remaining)

    for split_name in ("validation", "test"):
        while len(result[split_name]) < sizes[split_name]:
            result[split_name].append(remaining.pop())
    result["train"].extend(remaining)
    if len(result["train"]) != sizes["train"]:
        raise RuntimeError("Episode split size calculation failed")
    return {name: sorted(values) for name, values in result.items()}


def _fit_normalization(
    x_continuous: torch.Tensor,
    applicability: torch.Tensor,
    node_mask: torch.Tensor,
    global_features: torch.Tensor,
    train_indices: list[int],
) -> dict[str, torch.Tensor]:
    train = torch.tensor(train_indices, dtype=torch.long)
    train_x = x_continuous[train]
    train_applicable = applicability[train] & node_mask[train, None, :, None]
    means = []
    stds = []
    for feature_index in range(train_x.shape[-1]):
        values = train_x[..., feature_index][train_applicable[..., feature_index]]
        if values.numel() == 0:
            means.append(torch.tensor(0.0))
            stds.append(torch.tensor(1.0))
            continue
        means.append(values.mean())
        std = values.std(unbiased=False)
        stds.append(std if std > 1e-8 else torch.tensor(1.0))

    train_global = global_features[train]
    global_mean = train_global.mean(dim=(0, 1))
    global_std = train_global.std(dim=(0, 1), unbiased=False)
    global_std = torch.where(global_std > 1e-8, global_std, torch.ones_like(global_std))
    return {
        "feature_mean": torch.stack(means),
        "feature_std": torch.stack(stds),
        "global_mean": global_mean,
        "global_std": global_std,
    }


def _validate_dataset(
    payload: dict[str, Any],
    split_indices: dict[str, list[int]],
    sample_rows: list[dict[str, Any]],
) -> None:
    x = payload["x"]
    sample_count, _, node_count, _ = x.shape
    if sample_count == 0:
        raise ValueError("Dataset contains no model samples")
    if payload["adjacency"].shape != (sample_count, node_count, node_count):
        raise ValueError("Adjacency shape does not match x")
    if payload["node_mask"].shape != (sample_count, node_count):
        raise ValueError("Node mask shape does not match x")
    if (
        not torch.isfinite(x).all()
        or not torch.isfinite(payload["global_features"]).all()
    ):
        raise ValueError("Dataset contains non-finite model inputs")

    positive_indices = payload["positive_mask"].nonzero(as_tuple=False).flatten()
    for sample_index in positive_indices.tolist():
        node_index = int(payload["y_node"][sample_index])
        if node_index < 0 or node_index >= node_count:
            raise ValueError("Positive sample target node is outside node catalog")
        if not bool(payload["node_mask"][sample_index, node_index]):
            raise ValueError("Positive sample target node is masked")

    all_indices: list[int] = []
    split_group_sets: dict[str, set[str]] = {}
    for split_name, indices in split_indices.items():
        if not indices:
            raise ValueError(f"{split_name} split is empty")
        all_indices.extend(indices)
        split_group_sets[split_name] = {
            sample_rows[index]["group_id"] for index in indices
        }
        labels = payload["y_occurrence"][indices]
        if not bool(labels.bool().any()) or not bool((labels == 0).any()):
            raise ValueError(
                f"{split_name} split must contain positive and negative samples"
            )
    if sorted(all_indices) != list(range(sample_count)):
        raise ValueError("Split indices do not cover each sample exactly once")
    split_names = list(split_group_sets)
    for index, first in enumerate(split_names):
        for second in split_names[index + 1 :]:
            if split_group_sets[first].intersection(split_group_sets[second]):
                raise ValueError("Episode group leakage across splits")


class BstanTensorDataset(Dataset):
    """Dictionary-style tensor dataset used by the Phase-D trainer."""

    TENSOR_KEYS = (
        "x",
        "adjacency",
        "node_mask",
        "global_features",
        "y_occurrence",
        "y_node",
        "y_type",
        "y_time_to_start",
        "y_duration",
        "y_severity",
        "positive_mask",
        "duration_mask",
        "sample_group_id",
    )

    def __init__(self, payload: dict[str, Any], indices: Iterable[int] | None = None):
        self.payload = payload
        self.indices = (
            list(indices) if indices is not None else list(range(len(payload["x"])))
        )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample_index = self.indices[index]
        return {key: self.payload[key][sample_index] for key in self.TENSOR_KEYS}


def build_bstan_dataset(
    run_dirs: Iterable[Path],
    out_dir: Path,
    derived_dir_name: str = "derived_phase_b_v1",
    window_size: float = 30.0,
    stride: float = 30.0,
    input_windows: int = 4,
    horizon: float = 120.0,
    seed: int = 42,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Build, validate, and persist the Phase-C graph-sequence dataset."""
    if input_windows <= 0:
        raise ValueError("input_windows must be positive")
    out_dir = Path(out_dir).resolve()
    groups = _discover_groups(run_dirs, derived_dir_name)
    for group in groups:
        group["feature_rows"] = _filter_rows(group["feature_rows"], window_size, stride)
        group["label_rows"] = _filter_rows(group["label_rows"], window_size, stride)
        if not group["feature_rows"] or not group["label_rows"]:
            raise ValueError(f"No matching Phase-B rows for {group['group_id']}")
        for label in group["label_rows"]:
            if label.get("label_version") != LABEL_VERSION:
                raise ValueError(
                    f"Unexpected label version for {group['group_id']}: "
                    f"{label.get('label_version')!r}"
                )
            if not math.isclose(_f(label.get("prediction_horizon")), horizon):
                raise ValueError(f"Prediction horizon mismatch for {group['group_id']}")

    node_ids, node_types = _build_node_catalog(groups)
    node_index = {node_id: index for index, node_id in enumerate(node_ids)}
    resource_types = sorted(set(node_types.values()))
    resource_type_index = {name: index for index, name in enumerate(resource_types)}

    continuous_samples: list[torch.Tensor] = []
    applicability_samples: list[torch.Tensor] = []
    type_samples: list[torch.Tensor] = []
    global_samples: list[torch.Tensor] = []
    adjacency_samples: list[torch.Tensor] = []
    node_masks: list[torch.Tensor] = []
    targets: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    edge_rows: list[dict[str, Any]] = []
    group_sample_indices: dict[str, list[int]] = defaultdict(list)

    for group in groups:
        by_window: dict[int, dict[str, dict[str, str]]] = defaultdict(dict)
        for row in group["feature_rows"]:
            if row["resource_id"] not in node_index:
                continue
            by_window[_i(row["window_index"])][row["resource_id"]] = row
        labels = {_i(row["window_index"]): row for row in group["label_rows"]}
        active_nodes = set(group["included_node_ids"])
        node_mask = torch.tensor(
            [node_id in active_nodes for node_id in node_ids], dtype=torch.bool
        )
        adjacency, group_edges = build_static_graph(
            node_ids, node_types, active_nodes, group["config"]
        )
        for edge in group_edges:
            edge_rows.append(
                {
                    "group_id": group["group_id"],
                    "scenario_id": group["scenario_id"],
                    **edge,
                }
            )

        window_indices = sorted(set(by_window).intersection(labels))
        for position in range(input_windows - 1, len(window_indices)):
            sequence_indices = window_indices[
                position - input_windows + 1 : position + 1
            ]
            if any(
                current != previous + 1
                for previous, current in zip(sequence_indices, sequence_indices[1:])
            ):
                continue
            anchor_index = sequence_indices[-1]
            label = labels[anchor_index]
            if _i(label.get("label_observed")) != 1:
                continue
            sequence_rows = [by_window[index] for index in sequence_indices]
            starts = [
                _f(next(iter(rows.values())).get("window_start_s"))
                for rows in sequence_rows
            ]
            if any(
                not math.isclose(current - previous, stride)
                for previous, current in zip(starts, starts[1:])
            ):
                continue

            x_continuous = torch.zeros(
                (input_windows, len(node_ids), len(CONTINUOUS_FEATURES)),
                dtype=torch.float32,
            )
            applicability = torch.zeros_like(x_continuous, dtype=torch.bool)
            x_type = torch.zeros(
                (input_windows, len(node_ids), len(resource_types)), dtype=torch.float32
            )
            x_global = torch.zeros(
                (input_windows, len(GLOBAL_FEATURES)), dtype=torch.float32
            )
            for time_index, rows in enumerate(sequence_rows):
                first_row = next(iter(rows.values()))
                x_global[time_index] = torch.tensor(
                    [_f(first_row.get(name)) for name in GLOBAL_FEATURES],
                    dtype=torch.float32,
                )
                for node_id, row in rows.items():
                    catalog_index = node_index[node_id]
                    resource_type = node_types[node_id]
                    for feature_index, feature_name in enumerate(CONTINUOUS_FEATURES):
                        applicable = feature_is_applicable(
                            feature_name, node_id, resource_type
                        )
                        applicability[
                            time_index, catalog_index, feature_index
                        ] = applicable
                        if applicable:
                            x_continuous[time_index, catalog_index, feature_index] = _f(
                                row.get(feature_name)
                            )
                    x_type[
                        time_index,
                        catalog_index,
                        resource_type_index[resource_type],
                    ] = 1.0

            occurrence = _i(label.get("will_bottleneck"))
            target_node_id = label.get("future_bottleneck_object_id") or ""
            target_type = label.get("future_bottleneck_type") or ""
            sample_index = len(continuous_samples)
            continuous_samples.append(x_continuous)
            applicability_samples.append(applicability)
            type_samples.append(x_type)
            global_samples.append(x_global)
            adjacency_samples.append(adjacency)
            node_masks.append(node_mask)
            targets.append(
                {
                    "occurrence": occurrence,
                    "node": node_index.get(target_node_id, -1),
                    "type": resource_type_index.get(target_type, -1),
                    "time_to_start": _f(label.get("time_to_start")),
                    "duration": _f(label.get("duration")),
                    "severity": _f(label.get("severity_weak")),
                    "duration_observed": _i(label.get("duration_observed")),
                    "group_number": group["group_number"],
                }
            )
            sample_rows.append(
                {
                    "sample_index": sample_index,
                    "group_id": group["group_id"],
                    "run_id": group["run_id"],
                    "env_id": group["env_id"],
                    "episode_id": group["episode_id"],
                    "scenario_id": group["scenario_id"],
                    "input_window_indices": json.dumps(sequence_indices),
                    "anchor_window_index": anchor_index,
                    "anchor_time_s": _f(label.get("anchor_time_s")),
                    "will_bottleneck": occurrence,
                    "target_node_id": target_node_id,
                    "target_type": target_type,
                }
            )
            group_sample_indices[group["group_id"]].append(sample_index)

    if not continuous_samples:
        raise ValueError("No observable four-window samples were generated")

    x_continuous = torch.stack(continuous_samples)
    applicability = torch.stack(applicability_samples)
    x_type = torch.stack(type_samples)
    global_features = torch.stack(global_samples)
    adjacency = torch.stack(adjacency_samples)
    node_mask = torch.stack(node_masks)
    y_occurrence = torch.tensor(
        [target["occurrence"] for target in targets], dtype=torch.float32
    )

    group_scenarios = {group["group_id"]: group["scenario_id"] for group in groups}
    split_groups = _split_groups(
        group_sample_indices, group_scenarios, y_occurrence, seed
    )
    split_indices = {
        split_name: sorted(
            index for group_id in group_ids for index in group_sample_indices[group_id]
        )
        for split_name, group_ids in split_groups.items()
    }
    normalization = _fit_normalization(
        x_continuous,
        applicability,
        node_mask,
        global_features,
        split_indices["train"],
    )
    normalized_continuous = (
        x_continuous - normalization["feature_mean"].view(1, 1, 1, -1)
    ) / normalization["feature_std"].view(1, 1, 1, -1)
    normalized_continuous = torch.where(
        applicability & node_mask[:, None, :, None],
        normalized_continuous,
        torch.zeros_like(normalized_continuous),
    )
    x_type = x_type * node_mask[:, None, :, None]
    normalized_global = (
        global_features - normalization["global_mean"].view(1, 1, -1)
    ) / normalization["global_std"].view(1, 1, -1)

    payload: dict[str, Any] = {
        "x": torch.cat((normalized_continuous, x_type), dim=-1),
        "adjacency": adjacency,
        "node_mask": node_mask,
        "global_features": normalized_global,
        "y_occurrence": y_occurrence,
        "y_node": torch.tensor(
            [target["node"] for target in targets], dtype=torch.int64
        ),
        "y_type": torch.tensor(
            [target["type"] for target in targets], dtype=torch.int64
        ),
        "y_time_to_start": torch.tensor(
            [target["time_to_start"] for target in targets], dtype=torch.float32
        ),
        "y_duration": torch.tensor(
            [target["duration"] for target in targets], dtype=torch.float32
        ),
        "y_severity": torch.tensor(
            [target["severity"] for target in targets], dtype=torch.float32
        ),
        "positive_mask": y_occurrence.bool(),
        "duration_mask": torch.tensor(
            [
                target["occurrence"] == 1 and target["duration_observed"] == 1
                for target in targets
            ],
            dtype=torch.bool,
        ),
        "sample_group_id": torch.tensor(
            [target["group_number"] for target in targets], dtype=torch.int64
        ),
        "split_indices": {
            name: torch.tensor(indices, dtype=torch.int64)
            for name, indices in split_indices.items()
        },
    }
    _validate_dataset(payload, split_indices, sample_rows)

    for row in sample_rows:
        row["split"] = next(
            name
            for name, indices in split_indices.items()
            if row["sample_index"] in indices
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_dir / "dataset.pt")
    node_rows = [
        {
            "node_index": index,
            "resource_id": node_id,
            "resource_type": node_types[node_id],
            "resource_type_index": resource_type_index[node_types[node_id]],
            "episode_presence_count": sum(
                node_id in group["included_node_ids"] for group in groups
            ),
        }
        for index, node_id in enumerate(node_ids)
    ]
    _write_csv(
        out_dir / "node_catalog.csv",
        node_rows,
        [
            "node_index",
            "resource_id",
            "resource_type",
            "resource_type_index",
            "episode_presence_count",
        ],
    )
    _write_csv(
        out_dir / "graph_edge_table.csv",
        edge_rows,
        [
            "group_id",
            "scenario_id",
            "source_node_id",
            "source_node_index",
            "target_node_id",
            "target_node_index",
            "edge_type",
        ],
    )
    _write_csv(
        out_dir / "model_sample_index.csv",
        sample_rows,
        [
            "sample_index",
            "split",
            "group_id",
            "run_id",
            "env_id",
            "episode_id",
            "scenario_id",
            "input_window_indices",
            "anchor_window_index",
            "anchor_time_s",
            "will_bottleneck",
            "target_node_id",
            "target_type",
        ],
    )
    split_json = {
        name: {"group_ids": split_groups[name], "sample_indices": split_indices[name]}
        for name in ("train", "validation", "test")
    }
    (out_dir / "split.json").write_text(
        json.dumps(split_json, indent=2) + "\n", encoding="utf-8"
    )
    normalization_json = {
        "fit_split": "train",
        "continuous_features": list(CONTINUOUS_FEATURES),
        "feature_mean": normalization["feature_mean"].tolist(),
        "feature_std": normalization["feature_std"].tolist(),
        "global_features": list(GLOBAL_FEATURES),
        "global_mean": normalization["global_mean"].tolist(),
        "global_std": normalization["global_std"].tolist(),
    }
    (out_dir / "normalization.json").write_text(
        json.dumps(normalization_json, indent=2) + "\n", encoding="utf-8"
    )

    collector_versions = sorted({group["collector_version"] for group in groups})
    feature_names = list(CONTINUOUS_FEATURES) + [
        f"resource_type__{resource_type}" for resource_type in resource_types
    ]
    manifest = {
        "dataset_version": DATASET_VERSION,
        "label_version": LABEL_VERSION,
        "source_run_directories": sorted({str(group["run_dir"]) for group in groups}),
        "derived_dir_name": derived_dir_name,
        "collector_versions": collector_versions,
        "window_size_s": window_size,
        "stride_s": stride,
        "input_windows": input_windows,
        "prediction_horizon_s": horizon,
        "feature_names": feature_names,
        "global_feature_names": list(GLOBAL_FEATURES),
        "node_ids": node_ids,
        "resource_types": resource_types,
        "edge_types": sorted({row["edge_type"] for row in edge_rows}),
        "sample_counts": {
            name: len(indices) for name, indices in split_indices.items()
        },
        "total_samples": len(sample_rows),
        "positive_samples": int(y_occurrence.sum().item()),
        "positive_rate": float(y_occurrence.mean().item()),
        "episode_counts": {
            name: len(group_ids) for name, group_ids in split_groups.items()
        },
        "scenario_counts": {
            name: dict(
                sorted(
                    {
                        scenario_id: sum(
                            group_scenarios[group_id] == scenario_id
                            for group_id in group_ids
                        )
                        for scenario_id in sorted(set(group_scenarios.values()))
                    }.items()
                )
            )
            for name, group_ids in split_groups.items()
        },
        "git_commit": _git_commit(repo_root),
        "seed": seed,
        "validation": "passed",
    }
    (out_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return {
        "payload": payload,
        "manifest": manifest,
        "split": split_json,
        "node_catalog": node_rows,
        "edge_rows": edge_rows,
        "sample_rows": sample_rows,
    }
