"""Build and validate fixed-shape B2-B5 graph-sequence datasets."""

from __future__ import annotations

import csv
import hashlib
import inspect
import json
import math
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from .graph_builder import build_static_graph
from factory_bn_shared.causes import ROOT_CAUSE_CLASSES, encode_root_cause
from factory_bn_shared.remain import (
    ensure_labor_saturated_feature,
    node_event_targets,
    occupancy_node_mask,
    ops_hot_mask,
    pack_remain_target,
)
from .schema import (
    COLLECTOR_VERSION,
    CONTINUOUS_FEATURES,
    DATASET_CONTRACT,
    DATASET_VERSION,
    GLOBAL_FEATURES,
    LABOR_FEATURE,
    LABEL_VERSION,
    PREDICTION_TARGET_VERSION,
    RESOURCE_TYPES,
    TARGET_NODE_CATEGORY,
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
            metadata_path = feature_path.parent / "shared_metadata.json"
            config_rows = _read_csv(raw_dir / "episode_config.csv")
            if not config_rows:
                raise ValueError(f"Missing episode config row: {raw_dir}")
            config = config_rows[0]
            if not metadata_path.is_file():
                raise FileNotFoundError(metadata_path)
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("derived_contract_version") != DATASET_CONTRACT:
                raise ValueError(
                    f"Unexpected derived contract in {metadata_path}: "
                    f"{metadata.get('derived_contract_version')!r}"
                )
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
                    "run_name": run_dir.name,
                    "raw_dir": raw_dir,
                    "derived_dir": feature_path.parent,
                    "run_id": run_id,
                    "env_id": env_id,
                    "episode_id": episode_id,
                    "scenario_id": metadata.get("scenario_id") or "unknown",
                    "collector_version": config.get("collector_version") or "unknown",
                    "derived_contract_version": metadata.get(
                        "derived_contract_version"
                    ),
                    "label_version": metadata.get("label_version"),
                    "raw_contract_version": metadata.get("raw_contract_version"),
                    "raw_episode_sha256": metadata.get("raw_episode_sha256") or "",
                    "config": config,
                    "feature_rows": _read_csv(feature_path),
                    "label_rows": _read_csv(label_path),
                    "job_kpi_rows": _read_csv(feature_path.parent / "job_kpi.csv"),
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
        and math.isclose(_f(row.get("stride_s"), window_size), stride)
    ]


def _included_node_ids(group: dict[str, Any]) -> set[str]:
    return {row["resource_id"] for row in group["feature_rows"]}


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


def _split_groups(
    group_sample_indices: dict[str, list[int]],
    group_run_names: dict[str, str],
    seed: int,
) -> dict[str, list[str]]:
    """Match factory_bn: shuffle and split whole episodes within each raw run."""
    rng = np.random.default_rng(seed)
    by_run: dict[str, list[str]] = defaultdict(list)
    for group_id in sorted(group_sample_indices):
        by_run[group_run_names[group_id]].append(group_id)
    result: dict[str, list[str]] = {"train": [], "validation": [], "test": []}
    for run_name in sorted(by_run):
        group = list(by_run[run_name])
        rng.shuffle(group)
        n = len(group)
        if n < 3:
            raise ValueError(
                f"Raw run {run_name!r} needs at least 3 accepted episodes, got {n}"
            )
        n_train = max(1, int(n * 0.70))
        n_validation = max(1, int(n * 0.15))
        if n_train + n_validation >= n:
            n_validation = max(1, n - n_train - 1)
        result["train"].extend(group[:n_train])
        result["validation"].extend(group[n_train : n_train + n_validation])
        result["test"].extend(group[n_train + n_validation :])
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
    if payload["observation_mask"].shape != x.shape[:3]:
        raise ValueError("Observation mask shape does not match x")
    if payload["target_node_mask"].shape != (sample_count, node_count):
        raise ValueError("Target node mask shape does not match x")
    if bool((payload["target_node_mask"] & ~payload["node_mask"]).any()):
        raise ValueError("Target node mask contains inactive graph nodes")
    if not bool(payload["target_node_mask"].any(dim=1).all()):
        raise ValueError("Every sample must have at least one occupancy target node")
    if (
        not torch.isfinite(x).all()
        or not torch.isfinite(payload["global_features"]).all()
    ):
        raise ValueError("Dataset contains non-finite model inputs")

    all_indices: list[int] = []
    split_group_sets: dict[str, set[str]] = {}
    for split_name, indices in split_indices.items():
        if not indices:
            raise ValueError(f"{split_name} split is empty")
        all_indices.extend(indices)
        split_group_sets[split_name] = {
            sample_rows[index]["group_id"] for index in indices
        }
    if sorted(all_indices) != list(range(sample_count)):
        raise ValueError("Split indices do not cover each sample exactly once")
    split_names = list(split_group_sets)
    for index, first in enumerate(split_names):
        for second in split_names[index + 1 :]:
            if split_group_sets[first].intersection(split_group_sets[second]):
                raise ValueError("Episode group leakage across splits")


class FactoryBaselineTensorDataset(Dataset):
    """Dictionary-style tensor dataset shared by B2-B5 baselines."""

    TENSOR_KEYS = (
        "x",
        "adjacency",
        "node_mask",
        "observation_mask",
        "target_node_mask",
        "occ_node_mask",
        "hist_last_hot",
        "global_features",
        "y_cause",
        "jobs_remaining",
        "jobs_total",
        "target_start_position",
        "target_remain_len",
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
        sample = {key: self.payload[key][sample_index] for key in self.TENSOR_KEYS}
        group_number = str(int(sample["sample_group_id"]))
        series = self.payload["remain_series"][group_number]
        y_score, y_hot, remain_mask, _ = pack_remain_target(
            series["score"].numpy(),
            series["hot"].numpy(),
            t=int(sample["target_start_position"]),
            done_ti=int(
                sample["target_start_position"] + sample["target_remain_len"]
            ),
            max_remain_windows=int(self.payload["max_remain_windows"]),
            occupancy_horizon_windows=int(self.payload["max_remain_windows"]),
        )
        event_will, event_start, event_duration = node_event_targets(
            y_hot,
            min_windows=int(self.payload["event_min_windows"]),
            remain_mask=remain_mask,
            occ_node_mask=sample["occ_node_mask"].numpy(),
        )
        sample["y_score"] = torch.from_numpy(y_score)
        sample["y_hot"] = torch.from_numpy(y_hot)
        sample["remain_mask"] = torch.from_numpy(remain_mask)
        sample["event_will"] = torch.from_numpy(event_will)
        sample["event_start"] = torch.from_numpy(event_start)
        sample["event_duration"] = torch.from_numpy(event_duration)
        sample["sample_index"] = torch.tensor(sample_index, dtype=torch.int64)
        return sample


def load_shared_dataset(dataset_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load and validate the single tensor contract consumed by B2-B5."""
    dataset_dir = Path(dataset_dir).resolve()
    manifest_path = dataset_dir / "dataset_manifest.json"
    dataset_path = dataset_dir / "dataset.pt"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    if not dataset_path.exists():
        raise FileNotFoundError(dataset_path)
    load_options: dict[str, Any] = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        load_options["weights_only"] = True
    payload = torch.load(dataset_path, **load_options)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_manifest = {
        "dataset_contract": DATASET_CONTRACT,
        "dataset_version": DATASET_VERSION,
        "label_version": LABEL_VERSION,
        "target_node_category": TARGET_NODE_CATEGORY,
        "prediction_target_version": PREDICTION_TARGET_VERSION,
    }
    for key, expected in expected_manifest.items():
        actual = manifest.get(key)
        if actual != expected:
            raise ValueError(f"Unexpected {key}: expected {expected!r}, got {actual!r}")
    required = {
        "x",
        "adjacency",
        "node_mask",
        "target_node_mask",
        "occ_node_mask",
        "hist_last_hot",
        "global_features",
        "remain_series",
        "max_remain_windows",
        "event_min_windows",
        "split_indices",
    }
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"dataset.pt is missing keys: {sorted(missing)}")
    return payload, manifest


def build_factory_baseline_dataset(
    run_dirs: Iterable[Path],
    out_dir: Path,
    derived_dir_name: str = "shared_bn_agg_unsupervised_v2",
    window_size: float = 60.0,
    stride: float = 60.0,
    input_windows: int = 30,
    horizon: float = 180.0,
    seed: int = 42,
    repo_root: Path | None = None,
    allowed_group_ids: set[str] | None = None,
    max_remain_windows: int = 15,
    hot_min_windows: int = 8,
    hot_gap_windows: int = 1,
) -> dict[str, Any]:
    """Build B2-B5 tensors using the main experiment's operational targets."""
    if input_windows <= 0:
        raise ValueError("input_windows must be positive")
    if max_remain_windows <= 0:
        raise ValueError("max_remain_windows must be positive")
    out_dir = Path(out_dir).resolve()
    groups = _discover_groups(run_dirs, derived_dir_name)
    if allowed_group_ids is not None:
        groups = [group for group in groups if group["group_id"] in allowed_group_ids]
        if not groups:
            raise ValueError(
                "No shared-derived episode groups matched the accepted allowlist"
            )
    for group in groups:
        if group["collector_version"] != COLLECTOR_VERSION:
            raise ValueError(
                f"Unexpected collector version for {group['group_id']}: "
                f"expected {COLLECTOR_VERSION!r}, "
                f"got {group['collector_version']!r}"
            )
        if group["derived_contract_version"] != DATASET_CONTRACT:
            raise ValueError(
                f"Unexpected derived contract for {group['group_id']}: "
                f"{group['derived_contract_version']!r}"
            )
        if group["label_version"] != LABEL_VERSION:
            raise ValueError(
                f"Unexpected label version for {group['group_id']}: "
                f"{group['label_version']!r}"
            )
        group["feature_rows"] = _filter_rows(group["feature_rows"], window_size, stride)
        group["label_rows"] = _filter_rows(group["label_rows"], window_size, stride)
        if not group["feature_rows"] or not group["label_rows"]:
            raise ValueError(f"No matching Phase-B rows for {group['group_id']}")
        for label in group["label_rows"]:
            if not math.isclose(_f(label.get("horizon_s")), horizon):
                raise ValueError(f"Prediction horizon mismatch for {group['group_id']}")

    node_ids, node_types = _build_node_catalog(groups)
    node_index = {node_id: index for index, node_id in enumerate(node_ids)}
    unknown_types = sorted(set(node_types.values()).difference(RESOURCE_TYPES))
    if unknown_types:
        raise ValueError(f"Unsupported resource types: {unknown_types}")
    resource_types = list(RESOURCE_TYPES)
    resource_type_index = {name: index for index, name in enumerate(resource_types)}

    continuous_samples: list[torch.Tensor] = []
    labor_samples: list[torch.Tensor] = []
    applicability_samples: list[torch.Tensor] = []
    type_samples: list[torch.Tensor] = []
    global_samples: list[torch.Tensor] = []
    adjacency_samples: list[torch.Tensor] = []
    node_masks: list[torch.Tensor] = []
    observation_masks: list[torch.Tensor] = []
    target_node_masks: list[torch.Tensor] = []
    targets: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    edge_rows: list[dict[str, Any]] = []
    group_sample_indices: dict[str, list[int]] = defaultdict(list)
    remain_series: dict[str, dict[str, torch.Tensor]] = {}

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

        window_indices = sorted(by_window)
        raw_features = torch.zeros(
            (len(window_indices), len(node_ids), len(CONTINUOUS_FEATURES)),
            dtype=torch.float32,
        )
        raw_scores = torch.zeros(
            (len(window_indices), len(node_ids), 1), dtype=torch.float32
        )
        raw_types = torch.zeros(
            (len(window_indices), len(node_ids), len(resource_types)),
            dtype=torch.float32,
        )
        for node_id in active_nodes:
            catalog_index = node_index[node_id]
            raw_types[
                :, catalog_index, resource_type_index[node_types[node_id]]
            ] = 1.0
        window_starts = []
        for time_index, window_index_value in enumerate(window_indices):
            rows = by_window[window_index_value]
            window_starts.append(_f(next(iter(rows.values())).get("window_start_s")))
            for node_id, row in rows.items():
                catalog_index = node_index[node_id]
                raw_features[time_index, catalog_index] = torch.tensor(
                    [_f(row.get(name)) for name in CONTINUOUS_FEATURES]
                )
                raw_scores[time_index, catalog_index, 0] = _f(
                    row.get("bottleneck_score_s")
                )
        canonical_features = ensure_labor_saturated_feature(
            torch.cat((raw_features, raw_types), dim=-1).numpy()
        )
        raw_hot = torch.from_numpy(
            ops_hot_mask(
                canonical_features,
                window_size_s=window_size,
                min_hot_windows=hot_min_windows,
                gap_windows=hot_gap_windows,
            )
        )
        target_node_mask = torch.from_numpy(
            occupancy_node_mask(canonical_features).astype(np.bool_)
        ) & node_mask
        remain_series[str(group["group_number"])] = {
            "score": raw_scores,
            "hot": raw_hot,
        }
        completion_times = sorted(
            _f(row.get("complete_s"), float("inf")) for row in group["job_kpi_rows"]
        )
        jobs_total = len(completion_times)
        jobs_remaining = [
            sum(complete_s > start_s for complete_s in completion_times)
            for start_s in window_starts
        ]
        done_position = next(
            (
                position
                for position, remaining in enumerate(jobs_remaining)
                if remaining <= 0
            ),
            len(window_indices),
        )
        for position in range(input_windows, len(window_indices)):
            sequence_indices = window_indices[position - input_windows : position]
            if any(
                current != previous + 1
                for previous, current in zip(sequence_indices, sequence_indices[1:])
            ):
                continue
            remain_len = done_position - position
            if remain_len <= 0:
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
            observation_mask = torch.zeros(
                (input_windows, len(node_ids)), dtype=torch.bool
            )
            x_labor = torch.from_numpy(
                canonical_features[position - input_windows : position, :, -1:].copy()
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
                    observation_mask[time_index, catalog_index] = True
                    for feature_index, feature_name in enumerate(CONTINUOUS_FEATURES):
                        applicable = observation_mask[
                            time_index, catalog_index
                        ] and feature_is_applicable(
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

            _y_score, y_hot_np, remain_mask_np, _ = pack_remain_target(
                raw_scores.numpy(),
                raw_hot.numpy(),
                t=position,
                done_ti=done_position,
                max_remain_windows=max_remain_windows,
                occupancy_horizon_windows=max_remain_windows,
            )
            event_will, event_start, _event_duration = node_event_targets(
                y_hot_np,
                min_windows=hot_min_windows,
                remain_mask=remain_mask_np,
                occ_node_mask=target_node_mask.numpy(),
            )
            event_nodes = np.flatnonzero(event_will > 0.5)
            occurrence = int(event_nodes.size > 0)
            if occurrence:
                event_node = min(event_nodes, key=lambda index: int(event_start[index]))
                target_node_id = node_ids[int(event_node)]
            else:
                target_node_id = ""
            target_type = node_types.get(target_node_id, "")
            anchor_index = sequence_indices[-1]
            label = labels.get(anchor_index, {})
            sample_index = len(continuous_samples)
            continuous_samples.append(x_continuous)
            labor_samples.append(x_labor)
            applicability_samples.append(applicability)
            type_samples.append(x_type)
            global_samples.append(x_global)
            adjacency_samples.append(adjacency)
            node_masks.append(node_mask)
            observation_masks.append(observation_mask)
            target_node_masks.append(target_node_mask)
            targets.append(
                {
                    "event_positive": occurrence,
                    "cause": encode_root_cause(label.get("root_cause_reason")),
                    "jobs_remaining": jobs_remaining[position],
                    "jobs_total": jobs_total,
                    "target_start_position": position,
                    "target_remain_len": remain_len,
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
                    "anchor_time_s": starts[-1],
                    "first_future_start_s": window_starts[position],
                    "event_will_any": occurrence,
                    "first_event_node_id": target_node_id,
                    "first_event_node_type": target_type,
                }
            )
            group_sample_indices[group["group_id"]].append(sample_index)

    if not continuous_samples:
        raise ValueError("No horizon-ready causal samples were generated")

    x_continuous = torch.stack(continuous_samples)
    x_labor = torch.stack(labor_samples)
    applicability = torch.stack(applicability_samples)
    x_type = torch.stack(type_samples)
    global_features = torch.stack(global_samples)
    adjacency = torch.stack(adjacency_samples)
    node_mask = torch.stack(node_masks)
    observation_mask = torch.stack(observation_masks)
    target_node_mask = torch.stack(target_node_masks)
    event_positive = torch.tensor(
        [target["event_positive"] for target in targets], dtype=torch.float32
    )

    group_scenarios = {group["group_id"]: group["scenario_id"] for group in groups}
    group_run_names = {group["group_id"]: group["run_name"] for group in groups}
    split_groups = _split_groups(group_sample_indices, group_run_names, seed)
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
        "x": torch.cat((normalized_continuous, x_type, x_labor), dim=-1),
        "adjacency": adjacency,
        "node_mask": node_mask,
        "observation_mask": observation_mask,
        "target_node_mask": target_node_mask,
        "occ_node_mask": target_node_mask.float(),
        "hist_last_hot": torch.stack(
            [
                remain_series[str(target["group_number"])]["hot"] [
                    target["target_start_position"] - 1
                ]
                for target in targets
            ]
        ),
        "global_features": normalized_global,
        "y_cause": torch.tensor(
            [target["cause"] for target in targets], dtype=torch.int64
        ),
        "jobs_remaining": torch.tensor(
            [target["jobs_remaining"] for target in targets], dtype=torch.float32
        ),
        "jobs_total": torch.tensor(
            [target["jobs_total"] for target in targets], dtype=torch.float32
        ),
        "target_start_position": torch.tensor(
            [target["target_start_position"] for target in targets], dtype=torch.int64
        ),
        "target_remain_len": torch.tensor(
            [target["target_remain_len"] for target in targets], dtype=torch.int64
        ),
        "remain_series": remain_series,
        "max_remain_windows": max_remain_windows,
        "event_min_windows": hot_min_windows,
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
            "edge_weight",
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
            "first_future_start_s",
            "event_will_any",
            "first_event_node_id",
            "first_event_node_type",
        ],
    )
    split_json = {
        name: {"group_ids": split_groups[name], "sample_indices": split_indices[name]}
        for name in ("train", "validation", "test")
    }
    (out_dir / "split_manifest.json").write_text(
        json.dumps(split_json, indent=2) + "\n", encoding="utf-8"
    )
    normalization_json = {
        "fit_split": "train",
        "continuous_features": list(CONTINUOUS_FEATURES),
        "binary_context_features": [LABOR_FEATURE],
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
    ] + [LABOR_FEATURE]
    manifest = {
        "dataset_version": DATASET_VERSION,
        "dataset_contract": DATASET_CONTRACT,
        "label_version": LABEL_VERSION,
        "prediction_target_version": PREDICTION_TARGET_VERSION,
        "target_node_category": TARGET_NODE_CATEGORY,
        "source_run_directories": sorted({str(group["run_dir"]) for group in groups}),
        "source_episodes": [
            {
                "group_id": group["group_id"],
                "scenario_id": group["scenario_id"],
                "raw_episode_sha256": group["raw_episode_sha256"],
            }
            for group in sorted(groups, key=lambda item: item["group_id"])
        ],
        "derived_dir_name": derived_dir_name,
        "collector_versions": collector_versions,
        "window_size_s": window_size,
        "stride_s": stride,
        "input_windows": input_windows,
        "prediction_horizon_s": horizon,
        "occupancy_horizon_windows": max_remain_windows,
        "occupancy_horizon_s": max_remain_windows * window_size,
        "remain_to_jobs_done": True,
        "max_remain_windows": max_remain_windows,
        "label_mode": "unsupervised_operational_occupancy",
        "hot_min_windows": hot_min_windows,
        "hot_gap_windows": hot_gap_windows,
        "event_min_windows": hot_min_windows,
        "cause_classes": list(ROOT_CAUSE_CLASSES),
        "feature_names": feature_names,
        "global_feature_names": list(GLOBAL_FEATURES),
        "node_ids": node_ids,
        "resource_types": resource_types,
        "edge_types": sorted({row["edge_type"] for row in edge_rows}),
        "sample_counts": {
            name: len(indices) for name, indices in split_indices.items()
        },
        "total_samples": len(sample_rows),
        "event_positive_samples": int(event_positive.sum().item()),
        "event_positive_rate": float(event_positive.mean().item()),
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
