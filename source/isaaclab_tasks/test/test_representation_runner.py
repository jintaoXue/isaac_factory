"""Check trial routing and audit gates without training or reading test metrics."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def runner(tmp_path):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    script = checkout / "batch_factory_baseline_representation.sh"
    shutil.copyfile(ROOT / script.name, script)
    subprocess.run(["git", "init", "-q", "-b", "dev_xwt", str(checkout)], check=True)
    subprocess.run(["git", "-C", str(checkout), "add", script.name], check=True)
    subprocess.run(["git", "-C", str(checkout), "-c", "user.name=Fixture",
                    "-c", "user.email=fixture@example.invalid", "commit", "-qm", "fixture"], check=True)
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "dataset.pt").write_bytes(b"runner routing fixture; not a tensor")
    manifest = dataset / "dataset_manifest.json"
    manifest.write_text("{}")
    split = dataset / "episode_split_audit.json"
    split.write_text(json.dumps({"episode_split_match": True, "provenance": {
        "baseline_manifest": {"sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()}
    }}))
    audit = dataset / "validation_contract_audit.json"
    audit.write_text(json.dumps({"comparison_match": True,
                                "split_audit_sha256": hashlib.sha256(split.read_bytes()).hexdigest()}))
    log = tmp_path / "calls.jsonl"
    python = tmp_path / "mock_python"
    python.write_text(f"#!{sys.executable}\n" + '''import json, os, sys
from pathlib import Path
if sys.argv[1] == "-":
    os.execv(sys.executable, [sys.executable, *sys.argv[1:]])
with Path(os.environ["CALL_LOG"]).open("a") as stream:
    stream.write(json.dumps(sys.argv[1:]) + "\\n")
''')
    python.chmod(0o755)
    env = {**os.environ, "DATASET_DIR": str(dataset), "PYTHON_BIN": str(python),
           "TUNE_SEEDS": "42 43", "CALL_LOG": str(log), "DEVICE": "cpu"}
    return script, dataset, env, log


@pytest.mark.parametrize("model", ["B4", "B5"])
def test_routes_all_candidates_without_test_evaluation(runner, model):
    script, dataset, env, log = runner
    process = subprocess.run(["bash", str(script), model], env=env, text=True, capture_output=True)
    assert process.returncode == 0, process.stderr
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(calls) == 9
    train_calls = calls[:-1]
    for index, args in enumerate(train_calls):
        assert "--validation_only" in args
        assert args[args.index("--dataset_dir") + 1] == str(dataset)
        assert args[args.index("--seed") + 1] == ("42" if index < 4 else "43")
        candidate = ["control", "identity", "history", "identity_history"][index % 4]
        assert args[args.index("--training_profile") + 1].endswith("_" + candidate)
        assert args[args.index("--node_embedding") + 1] == (
            "16" if candidate in {"identity", "identity_history"} else "0")
        assert args[args.index("--temporal_readout") + 1] == (
            "last_mean" if candidate in {"history", "identity_history"} else "last")
        expected_entry = "train_b4_gcn_gru.py" if model == "B4" else "train_b5_gat_gru.py"
        assert any(value.endswith(expected_entry) for value in args)
    assert calls[-1][0].endswith("select_baseline_tuning.py")
    repeated = subprocess.run(["bash", str(script), model], env=env, text=True, capture_output=True)
    assert repeated.returncode != 0
    assert len(log.read_text().splitlines()) == 9


@pytest.mark.parametrize("change", ["failed", "manifest", "split", "missing"])
def test_refuses_invalid_or_stale_audit_before_starting_trials(runner, change):
    script, dataset, env, log = runner
    audit_path = dataset / "validation_contract_audit.json"
    if change == "failed":
        audit = json.loads(audit_path.read_text())
        audit["comparison_match"] = False
        audit_path.write_text(json.dumps(audit))
    elif change == "manifest":
        (dataset / "dataset_manifest.json").write_text('{"changed": true}')
    elif change == "split":
        with (dataset / "episode_split_audit.json").open("a") as stream:
            stream.write("\n")
    else:
        audit_path.unlink()
    process = subprocess.run(["bash", str(script), "B4"], env=env, text=True, capture_output=True)
    assert process.returncode != 0
    assert not log.exists()
    assert not (dataset / "models").exists()
