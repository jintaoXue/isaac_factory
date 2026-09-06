"""Frozen labels are explicit inputs; inconsistent artifacts must fail closed."""

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "isaaclab_tasks/direct/hc_factory/tools"))
from factory_bn_shared.bundle import align_frozen_causes, file_hash, load_frozen_cause_labels
from factory_bn_shared.causes import ROOT_CAUSE_CLASSES


@pytest.fixture
def frozen(tmp_path):
    raw = tmp_path / "physical_run"
    config = raw / "episode_00/env_00/episode_config.csv"
    config.parent.mkdir(parents=True)
    with config.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["run_id", "env_id", "episode_id"])
        writer.writeheader()
        writer.writerow({"run_id": "physical_id", "env_id": 0, "episode_id": 0})
    name = "alias__episode_00"
    meta = {"run_names": ["alias"], "run_dirs": [str(raw)], "episodes": {name: {}},
            "cause_classes": list(ROOT_CAUSE_CLASSES), "window_size_s": 60.0}
    arrays = {"episode_names": np.asarray([name]), "cause_classes": np.asarray(ROOT_CAUSE_CLASSES),
              "window_size_s": np.asarray([60.0]), name + "_windows": np.arange(3),
              name + "_window_start_s": np.asarray([0., 60., 120.]),
              name + "_cause": np.asarray([-1, 2, 5])}
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    def write():
        (bundle / "meta.json").write_text(json.dumps(meta))
        np.savez(bundle / "episodes.npz", **arrays)
    write()
    return bundle, meta, arrays, write


def test_physical_identity_and_frozen_values(frozen):
    bundle, _, _, _ = frozen
    labels, provenance = load_frozen_cause_labels(bundle, {"physical_id:env_00:episode_00"}, 60)
    episode = labels["physical_id:env_00:episode_00"]
    np.testing.assert_array_equal(align_frozen_causes(episode, [0, 1, 2], [0, 60, 120]), [-1, 2, 5])
    assert episode["name"] == "alias__episode_00"
    assert provenance["files"]["episodes"]["sha256"] == file_hash(bundle / "episodes.npz")


@pytest.mark.parametrize("change", ["classes", "cohort", "size", "cause_range", "cause_float",
                                   "windows", "length", "nan_start", "meta_inventory"])
def test_inconsistent_bundle_rejected(frozen, change):
    bundle, meta, arrays, write = frozen
    groups = {"physical_id:env_00:episode_00"}
    if change == "classes":
        arrays["cause_classes"] = arrays["cause_classes"][::-1]
    elif change == "cohort":
        groups = {"different:env_00:episode_00"}
    elif change == "size":
        meta["window_size_s"] = 30
    elif change == "cause_range":
        arrays["alias__episode_00_cause"][1] = 99
    elif change == "cause_float":
        arrays["alias__episode_00_cause"] = np.asarray([-1., 2.5, 5.])
    elif change == "windows":
        arrays["alias__episode_00_windows"] = np.asarray([0, 0, 2])
    elif change == "length":
        arrays["alias__episode_00_cause"] = np.asarray([-1, 2])
    elif change == "nan_start":
        arrays["alias__episode_00_window_start_s"][1] = np.nan
    else:
        meta["episodes"] = {}
    write()
    with pytest.raises(ValueError):
        load_frozen_cause_labels(bundle, groups, 60)


@pytest.mark.parametrize("windows,starts", [([0, 1], [0, 60]), ([0, 1, 2], [0, 61, 120])])
def test_shifted_derived_inputs_rejected(frozen, windows, starts):
    bundle, _, _, _ = frozen
    labels, _ = load_frozen_cause_labels(bundle, {"physical_id:env_00:episode_00"}, 60)
    with pytest.raises(ValueError, match="anchors differ"):
        align_frozen_causes(labels["physical_id:env_00:episode_00"], windows, starts)


def test_missing_bundle_has_no_recomputed_label_fallback(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_frozen_cause_labels(tmp_path, {"physical_id:env_00:episode_00"}, 60)
