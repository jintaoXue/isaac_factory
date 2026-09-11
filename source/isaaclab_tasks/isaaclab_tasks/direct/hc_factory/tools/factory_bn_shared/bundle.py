"""Bind frozen main-experiment cause labels to physical raw episode identities."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from .causes import ROOT_CAUSE_CLASSES


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main_episode_identities(meta: dict, names: list[str]) -> dict[str, str]:
    if len(meta["run_names"]) != len(meta["run_dirs"]):
        raise ValueError("Main bundle run_names/run_dirs lengths differ")
    runs = dict(zip(meta["run_names"], meta["run_dirs"], strict=True))
    if len(runs) != len(meta["run_names"]):
        raise ValueError("Main bundle contains duplicate run aliases")
    result = {}
    for name in names:
        alias, separator, episode = name.partition("__")
        if not separator or not episode.startswith("episode_"):
            raise ValueError(f"Unsupported main episode identity: {name}")
        # The main exporter reads env_00 only; never guess another env or run.
        path = Path(runs[alias]) / episode / "env_00" / "episode_config.csv"
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != 1:
            raise ValueError(f"{path}: expected exactly one config row")
        row = rows[0]
        run_id = row["run_id"]
        env_id, episode_id = int(row["env_id"]), int(row["episode_id"])
        if not run_id or env_id != 0 or episode_id != int(episode.removeprefix("episode_")):
            raise ValueError(f"Raw episode identity contradicts bundle name: {path}")
        result[name] = f"{run_id}:env_{env_id:02d}:episode_{episode_id:02d}"
    if len(set(result.values())) != len(result):
        raise ValueError("Multiple main bundle names refer to the same raw episode")
    return result


def load_frozen_cause_labels(
    bundle_dir: Path, group_ids: set[str], window_size: float
) -> tuple[dict[str, dict], dict]:
    """Read the shared A.3 target artifact, not model predictions or new CSV labels."""
    paths = {"meta": Path(bundle_dir) / "meta.json", "episodes": Path(bundle_dir) / "episodes.npz"}
    provenance = {
        "kind": "frozen_canonical_bundle",
        "anchor": "last_history_window",
        "files": {name: {"path": str(path.resolve()), "sha256": file_hash(path)}
                  for name, path in paths.items()},
    }
    meta = json.loads(paths["meta"].read_text(encoding="utf-8"))
    labels = {}
    with np.load(paths["episodes"], allow_pickle=False) as bundle:
        names = bundle["episode_names"].tolist()
        if len(names) != len(set(names)) or set(names) != set(meta["episodes"]):
            raise ValueError("Main bundle NPZ/meta episode inventories differ or contain duplicates")
        if (bundle["cause_classes"].tolist() != list(ROOT_CAUSE_CLASSES)
                or meta["cause_classes"] != list(ROOT_CAUSE_CLASSES)):
            raise ValueError("Main bundle cause class order does not match the shared contract")
        if (bundle["window_size_s"].shape != (1,)
                or not np.isclose(bundle["window_size_s"][0], window_size)
                or not np.isclose(meta["window_size_s"], window_size)):
            raise ValueError("Main bundle window size does not match the derived inputs")
        identities = main_episode_identities(meta, names)
        if set(identities.values()) != group_ids:
            raise ValueError("Frozen main bundle and accepted baseline episode cohorts differ")
        for name, group in identities.items():
            windows = bundle[name + "_windows"]
            starts = bundle[name + "_window_start_s"]
            cause = bundle[name + "_cause"]
            if (windows.ndim != 1 or not len(windows)
                    or starts.shape != windows.shape or cause.shape != windows.shape
                    or not np.issubdtype(windows.dtype, np.integer)
                    or not np.issubdtype(cause.dtype, np.integer)
                    or not np.isfinite(starts).all()
                    or np.any(np.diff(windows) <= 0) or np.any(np.diff(starts) <= 0)
                    or np.any(cause < -1) or np.any(cause >= len(ROOT_CAUSE_CLASSES))):
                raise ValueError(f"Invalid frozen cause/window arrays: {name}")
            labels[group] = {"name": name, "windows": windows, "starts": starts, "cause": cause}
    for name, path in paths.items():
        if file_hash(path) != provenance["files"][name]["sha256"]:
            raise ValueError(f"Main bundle changed while reading {path}")
    return labels, provenance


def align_frozen_causes(episode: dict, windows: list[int], starts: list[float]) -> np.ndarray:
    if (not np.array_equal(episode["windows"], windows)
            or episode["starts"].shape != (len(starts),)
            or not np.allclose(episode["starts"], starts, rtol=0, atol=1e-4)):
        raise ValueError(f"Frozen cause windows/anchors differ from inputs: {episode['name']}")
    return episode["cause"]
