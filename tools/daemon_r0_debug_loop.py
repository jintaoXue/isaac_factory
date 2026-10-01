#!/usr/bin/env python3
"""Daemonize and run the R0-debug closed loop (survives parent shell exit).

Usage:
  python tools/daemon_r0_debug_loop.py
  python tools/daemon_r0_debug_loop.py --attach   # watch existing evidence/train only until exit, then closed-loop
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOOP_DIR = ROOT / "outputs/train_monitor/r0_debug_loop"
WATCHDOG = ROOT / "tools/run_r0_debug_loop.sh"


def daemonize(logfile: Path) -> None:
    if os.fork() > 0:
        raise SystemExit(0)
    os.setsid()
    if os.fork() > 0:
        raise SystemExit(0)
    os.chdir(str(ROOT))
    fd = os.open(str(logfile), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    os.dup2(fd, 1)
    os.dup2(fd, 2)
    devnull = os.open(os.devnull, os.O_RDONLY)
    os.dup2(devnull, 0)
    for sig in (signal.SIGHUP,):
        signal.signal(sig, signal.SIG_IGN)


def attach_until_done() -> int:
    """Watch current evidence.path until status finished/failed; return 0 to let outer closed-loop continue."""
    ev_path = LOOP_DIR / "evidence.path"
    state_path = LOOP_DIR / "STATE.json"
    if not ev_path.exists():
        print("[attach] no evidence.path; skip attach", flush=True)
        return 0
    evidence = Path(ev_path.read_text().strip())
    print(f"[attach] watching {evidence}", flush=True)
    while True:
        if (LOOP_DIR / "STOP").exists():
            return 0
        status_file = evidence / "status.json"
        if status_file.exists():
            status = json.loads(status_file.read_text())
            if status.get("state") in ("finished", "failed"):
                print(f"[attach] train ended state={status.get('state')} signal={status.get('signal')}", flush=True)
                # Leave STATE active_run pointing at this evidence so closed_loop resume records the crash.
                if state_path.exists():
                    st = json.loads(state_path.read_text())
                    st["phase"] = "attach_seen_exit"
                    st["stop"] = False
                    # ensure relaunch will bump
                    st.setdefault("policy", {})["relaunch_count"] = max(
                        int(st.get("policy", {}).get("relaunch_count") or 0), 1
                    )
                    if status.get("signal") == "SIGSEGV":
                        st["policy"]["cuda_launch_blocking"] = True
                    state_path.write_text(json.dumps(st, indent=2) + "\n")
                return 0
        # heartbeat file for attach mode
        tick = {
            "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "kind": "attach_heartbeat",
            "evidence": str(evidence),
            "pid": os.getpid(),
        }
        # refresh last_tick lightly
        if state_path.exists():
            try:
                st = json.loads(state_path.read_text())
                st["last_tick"] = tick["at"]
                st["progress"] = {**(st.get("progress") or {}), **tick}
                # parse step from console
                console = evidence / "console.log"
                if console.exists():
                    import re

                    steps = re.findall(r"step=(\d+)\s+episode=(\d+)", console.read_text(errors="replace"))
                    if steps:
                        st["progress"]["step"] = int(steps[-1][0])
                        st["progress"]["episode"] = int(steps[-1][1])
                st.setdefault("ticks", []).append({"at": tick["at"], "kind": "attach_heartbeat",
                                                   "step": st.get("progress", {}).get("step"),
                                                   "episode": st.get("progress", {}).get("episode"),
                                                   "status": "running", "loop_pid": os.getpid()})
                if len(st["ticks"]) > 500:
                    st["ticks"] = st["ticks"][-300:]
                state_path.write_text(json.dumps(st, indent=2) + "\n")
            except Exception as exc:
                print(f"[attach] state update error: {exc}", flush=True)
        time.sleep(60)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attach", action="store_true", help="first watch existing train, then closed-loop")
    parser.add_argument("--foreground", action="store_true")
    args = parser.parse_args()
    LOOP_DIR.mkdir(parents=True, exist_ok=True)
    log = LOOP_DIR / "daemon.log"
    if not args.foreground:
        daemonize(log)
    (LOOP_DIR / "daemon.pid").write_text(str(os.getpid()))
    print(f"[daemon] pid={os.getpid()} attach={args.attach}", flush=True)

    env = os.environ.copy()
    env.setdefault("HC_R_RUN_TAG", "R0-debug")
    env.setdefault("HC_R_SEED", "42")
    env.setdefault("HC_MAX_HARD_EPISODES", "100")
    env.setdefault("HC_R_WANDB", "1")
    env.setdefault("HC_LOOP_MAX_RELAUNCH", "8")
    env.setdefault("HC_LOOP_RESUME", "1")
    env.setdefault("HC_PYTHON", sys.executable)

    if args.attach:
        attach_until_done()

    # Run durable bash watchdog (restarts closed_loop on unexpected death).
    os.execve(
        "/bin/bash",
        ["bash", str(WATCHDOG)],
        env,
    )


if __name__ == "__main__":
    raise SystemExit(main())
