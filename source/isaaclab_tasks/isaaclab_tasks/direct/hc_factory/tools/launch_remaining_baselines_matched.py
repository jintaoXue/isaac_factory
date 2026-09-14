#!/usr/bin/env python3
"""Guarded user-run launcher: check completed B4/B5, fast-forward, reuse dead tmux pane."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path("/home/sci/work/BSTAN_isaac_factory")
TOOLS = ROOT / "source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/tools"
DATASET = TOOLS.parent / "output/bottleneck_dataset/experiments/factory_pdformer_134_v3"
TAG = "remaining_matched20260914"
PYTHON = "/home/sci/repos/miniconda3/envs/env_isaaclab/bin/python"
PANES = ("baseline_dense_v6:0.0", "baseline_dense_diag:0.0")

def out(*args):
    return subprocess.check_output(args, text=True).strip()

def check_terminal(raw):
    if raw.split() != ["1", "0"]:
        raise ValueError("Existing tmux pane is not stopped with exit 0; inspect without killing it")

def validate_completion(final, state, raw):
    expected = [(m, cap) for cap in (5, 10, 15) for m in ("B4", "B5")]
    if (final["status"] != "six_matched_tasks_independently_verified"
            or [(r["model"], r["max_start"]) for r in final["results"]] != expected
            or final["test_evaluated"] or state["status"] != "completed"
            or state["final_sha256"] != hashlib.sha256(raw).hexdigest()):
        raise ValueError("B4/B5 has no matching successful final verification")

def launch(source, resume=False):
    if Path.cwd().resolve() != ROOT or out("git", "branch", "--show-current") != "dev_xwt":
        raise ValueError("Run only in /home/sci/work/BSTAN_isaac_factory on dev_xwt")
    if out("git", "status", "--porcelain", "--untracked-files=no"):
        raise ValueError("Tracked worktree changes exist; no update or training was attempted")
    for pane in PANES:
        check_terminal(out("tmux", "display-message", "-p", "-t", pane, "#{pane_dead} #{pane_dead_status}"))
    raw = (DATASET / "baseline_matched_protocol_training20260913_final_verification.json").read_bytes()
    validate_completion(json.loads(raw), json.loads(
        (DATASET / "baseline_matched_protocol_finish20260914_state.json").read_text()), raw)
    saved = out("tmux", "show-environment", "-g", "PYTHONPATH")
    if not saved.startswith("PYTHONPATH="):
        raise ValueError("Verified Python library path is unavailable; do not reinstall blindly")
    if not Path(PYTHON).is_file():
        raise ValueError("The registered env_isaaclab Python is missing")
    mode = "run" if resume else "all"
    if (DATASET / (TAG + "_results.json")).exists():
        raise FileExistsError("B2/B3 final results already exist; inspect them instead of retraining")
    plan_path = DATASET / (TAG + "_plan.json")
    if plan_path.exists() and not resume:
        raise FileExistsError("A registered B2/B3 queue already exists; inspect it before --resume")
    if resume:
        plan = json.loads(plan_path.read_text())
        if plan["source_commit"] != source:
            raise ValueError("Resume must keep the exact registered runtime")
        for task in plan["tasks"]:
            p = Path(task["record"])
            if p.exists() and json.loads(p.read_text())["status"] != "validation_completed":
                raise ValueError("A partial stage needs inspection; no automatic retraining")
    subprocess.run(["git", "merge-base", "--is-ancestor", "HEAD", source], check=True)
    # Both old panes and the formal result were checked before updating runtime.
    subprocess.run(["git", "merge", "--ff-only", source], check=True)
    env = ["env", "-u", "PYTHONHOME", "PYTHONDONTWRITEBYTECODE=1",
           "OMP_NUM_THREADS=2", "OPENBLAS_NUM_THREADS=2", "MKL_NUM_THREADS=2",
           "PYTHONPATH=" + str(TOOLS) + ":" + saved.split("=", 1)[1]]
    log = DATASET / (TAG + ("_resume.log" if resume else ".log"))
    with log.open("x", encoding="utf-8") as f:
        f.write("Registered runtime: " + source + "\n")
    cmd = "exec " + shlex.join(env + [PYTHON, "-B", "-u",
        str(TOOLS / "run_remaining_baselines_matched.py"), "--source_commit", source, "--mode", mode])
    cmd += " >> " + shlex.quote(str(log)) + " 2>&1"
    subprocess.run(["tmux", "respawn-pane", "-t", PANES[0], "-c", str(ROOT), cmd], check=True)
    print("B2/B3 六组队列已提交；先预检，通过后自动训练。")
    print("日志：", log)
    print("查看：tail -n 40 " + shlex.quote(str(log)))
    print("完成标志：B2_B3_SIX_TASKS_VERIFIED")
    print("若预检失败，保留输出并反馈；不要删记录、建目录或重复启动。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_commit", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    try:
        launch(args.source_commit, args.resume)
    except Exception as error:
        print("REMAINING_LAUNCH_STOPPED:", type(error).__name__, str(error), file=sys.stderr)
        raise SystemExit(1)
