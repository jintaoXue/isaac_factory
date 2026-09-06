import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest
import numpy as np
import torch

TOOLS = Path(__file__).resolve().parents[1] / "isaaclab_tasks/direct/hc_factory/tools"
sys.path.insert(0, str(TOOLS))

from audit_baseline_episode_split import compare_splits, main_episode_identities


def test_equal_counts_do_not_prove_equal_split():
    baseline = {"train": ["a", "b"], "validation": ["c"], "test": ["d"]}
    main = {"train": ["a", "c"], "validation": ["b"], "test": ["d"]}
    report = compare_splits(baseline, main)
    assert not report["episode_split_match"]
    assert len(report["changed_split"]) == 2
    assert report["counts"]["train"] == {"baseline": 2, "main": 2, "intersection": 1}
    assert compare_splits(baseline, baseline)["episode_split_match"]


def test_cohort_mismatch_and_duplicate_rejected():
    baseline = {"train": ["a"], "validation": ["b"], "test": ["c"]}
    main = {"train": ["a"], "validation": ["b"], "test": ["d"]}
    report = compare_splits(baseline, main)
    assert report["baseline_only"] == ["c"]
    assert report["main_only"] == ["d"]
    with pytest.raises(ValueError, match="repeated"):
        compare_splits(baseline, {"train": ["a"], "validation": ["a"], "test": ["d"]})


def test_raw_identity_retains_alias_and_rejects_duplicate(tmp_path):
    real = tmp_path / "machine20"
    env = real / "episode_03/env_00"
    env.mkdir(parents=True)
    with (env / "episode_config.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["run_id", "episode_id", "env_id"])
        writer.writeheader()
        writer.writerow({"run_id": "collected_run", "episode_id": 3, "env_id": 0})
    meta = {"run_names": ["n10_machine1.0"], "run_dirs": [str(real)]}
    names = ["n10_machine1.0__episode_03"]
    assert main_episode_identities(meta, names) == {
        names[0]: "collected_run:env_00:episode_03"
    }
    meta["run_names"].append("machine20")
    meta["run_dirs"].append(str(real))
    with pytest.raises(ValueError, match="same raw episode"):
        main_episode_identities(meta, names + ["machine20__episode_03"])


def test_cli_reads_only_episode_inventory_and_rejects_overwrite(tmp_path):
    raw = tmp_path / "physical_run"
    bundle = tmp_path / "bundle"
    baseline = tmp_path / "baseline"
    bundle.mkdir()
    baseline.mkdir()
    names = [f"alias__episode_{index:02d}" for index in range(6)]
    for index in range(6):
        env = raw / f"episode_{index:02d}/env_00"
        env.mkdir(parents=True)
        with (env / "episode_config.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["run_id", "env_id", "episode_id"])
            writer.writeheader()
            writer.writerow({"run_id": "source", "env_id": 0, "episode_id": index})
    (bundle / "meta.json").write_text(json.dumps({
        "run_names": ["alias"], "run_dirs": [str(raw)], "episodes": dict.fromkeys(names, {})
    }))
    # No feature/label arrays: the audit must only access the episode inventory.
    np.savez(bundle / "episodes.npz", episode_names=np.asarray(names))
    checkpoint = tmp_path / "own_checkpoint.pt"
    torch.save({"config": {"seed": 42}}, checkpoint)
    groups = [f"source:env_00:episode_{index:02d}" for index in range(6)]
    np.random.default_rng(42).shuffle(groups)
    (baseline / "split_manifest.json").write_text(json.dumps({
        "train": {"group_ids": groups[:4]},
        "validation": {"group_ids": groups[4:5]},
        "test": {"group_ids": groups[5:]},
    }))
    (baseline / "dataset_manifest.json").write_text(json.dumps({
        "source_episodes": [{"group_id": group} for group in groups]
    }))
    output = tmp_path / "audit.json"
    command = [
        sys.executable, str(TOOLS / "audit_baseline_episode_split.py"),
        "--baseline_dir", str(baseline), "--main_bundle", str(bundle),
        "--main_checkpoint", str(checkpoint),
        "--pdformer_root", str(TOOLS.parent / "PDFormer"), "--output", str(output),
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    report = json.loads(output.read_text())
    assert report["episode_split_match"]
    assert not report["test_metrics_read"]
    assert report["counts"]["train"]["intersection"] == 4
    original = output.read_bytes()
    repeated = subprocess.run(command, capture_output=True, text=True)
    assert repeated.returncode != 0
    assert output.read_bytes() == original
