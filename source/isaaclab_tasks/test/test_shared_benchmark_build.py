"""Offline benchmark routing and overwrite guards, without an Isaac Sim dependency."""

import json
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "isaaclab_tasks/direct/hc_factory/tools"))
import build_shared_benchmark as builder


def test_nonempty_benchmark_is_not_overwritten(tmp_path, monkeypatch):
    out = tmp_path / "benchmark"
    out.mkdir()
    sentinel = out / "dataset.pt"
    sentinel.write_bytes(b"existing")
    monkeypatch.setattr(sys, "argv", ["build", "--run_dirs", str(tmp_path / "raw"),
                                      "--out_dir", str(out)])
    with pytest.raises(FileExistsError, match="empty benchmark"):
        builder.main()
    assert sentinel.read_bytes() == b"existing"


def test_same_run_names_are_rejected_before_writing(tmp_path, monkeypatch):
    out = tmp_path / "benchmark"
    monkeypatch.setattr(sys, "argv", ["build", "--run_dirs", str(tmp_path / "a/run"),
                                      str(tmp_path / "b/run"), "--out_dir", str(out)])
    with pytest.raises(ValueError, match="unique"):
        builder.main()
    assert not out.exists()


def test_offline_derived_stays_inside_benchmark(tmp_path, monkeypatch):
    raw = tmp_path / "raw_run"
    raw.mkdir()
    out = tmp_path / "benchmark"
    pairs = [(raw, raw / f"episode_{ep:02d}/env_00") for ep in range(3)]
    audits = [{"accepted": True, "run_id": "run", "env_id": 0, "episode_id": ep,
               "raw_contract_version": "tyx_raw_v0.3", "raw_episode_sha256": "fixture",
               "scenario_id": "fixture", "episode_end_s": 125.0} for ep in range(3)]
    monkeypatch.setattr(sys, "argv", ["build", "--run_dirs", str(raw), "--out_dir", str(out)])
    monkeypatch.setattr(builder, "discover_env_dirs", Mock(return_value=pairs))
    monkeypatch.setattr(builder, "audit_env_dir", Mock(side_effect=audits))
    monkeypatch.setattr(builder, "build_report", Mock(return_value={"status": "passed"}))

    def derive(**kwargs):
        kwargs["out_dir"].mkdir(parents=True)
        return {"output": str(kwargs["out_dir"])}

    process = Mock(side_effect=derive)
    build = Mock(return_value={"manifest": {key: None for key in (
        "dataset_contract", "dataset_version", "label_version", "total_samples",
        "event_positive_samples", "event_positive_rate", "sample_counts", "episode_counts",
    )}})
    monkeypatch.setattr(builder, "process_env_dir", process)
    monkeypatch.setattr(builder, "build_factory_baseline_dataset", build)
    builder.main()
    assert list(raw.iterdir()) == []
    assert process.call_count == 3
    for call in process.call_args_list:
        assert call.kwargs["closed_windows_only"] is False
        assert call.kwargs["out_dir"].is_relative_to(out / "derived" / raw.name)
        metadata = json.loads((call.kwargs["out_dir"] / "shared_metadata.json").read_text())
        assert metadata["closed_windows_only"] is False
    assert build.call_args.kwargs["derived_root"] == out / "derived"
    assert len(build.call_args.kwargs["allowed_group_ids"]) == 3
