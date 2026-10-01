#!/usr/bin/env python3
"""Closed-loop R0 train + crash diagnosis (SIGSEGV / deepcopy / device-path).

Launches ``run_r_series.py R0`` with W&B name derived from ``HC_R_RUN_TAG``
(default ``R0-debug`` → wandb ``R0-debug-N10``). On crash: classify stack,
apply known patches when possible, relaunch with ``-reN`` suffix.

Stop: touch ``outputs/train_monitor/r0_debug_loop/STOP`` or set STATE stop=true.

Usage:
  python tools/closed_loop_r0_debug.py
  HC_MAX_HARD_EPISODES=100 HC_R_SEED=42 python tools/closed_loop_r0_debug.py
"""
from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOOP_DIR = ROOT / "outputs/train_monitor/r0_debug_loop"
STATE_PATH = LOOP_DIR / "STATE.json"
STOP_PATH = LOOP_DIR / "STOP"
DIAG_DOC = ROOT / "docs/training_crash_diagnosis_2026-09-29.md"
TZ = timezone(timedelta(hours=8))
PY = os.environ.get("HC_PYTHON", sys.executable)

# Known surfaces from overnight diagnosis (2026-09-29/30).
KNOWN = {
    "deepcopy_typeerror": {
        "markers": ["unhashable type: 'dict'", "TpaInfoPool", "__deepcopy__"],
        "fix": "clone_env_subtree already in tpa_info_pool; relaunch",
        "blocking_next": False,
    },
    "segv_to_device": {
        "markers": ["hier_obs.py", "_to_device", "Fatal Python error"],
        "fix": "encode_* pre= + skip-if-on-device; already patched — relaunch; if repeats enable blocking",
        "blocking_next": True,
    },
    "segv_stack_ongoing": {
        "markers": ["_encode_ongoing", "torch.stack", "Fatal Python error"],
        "fix": "device-path / clone patches; relaunch with emb debug",
        "blocking_next": True,
    },
    "segv_embedding": {
        "markers": ["Embedding.forward", "sparse.py", "next_logistic_id", "Fatal Python error"],
        "fix": "emb_debug snapshot; not OOB — relaunch with sync encode",
        "blocking_next": True,
    },
}


def now() -> str:
    return datetime.now(TZ).isoformat(timespec="seconds")


def save_state(state: dict) -> None:
    LOOP_DIR.mkdir(parents=True, exist_ok=True)
    tmp = STATE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2) + "\n")
    tmp.replace(STATE_PATH)


def load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {}


def append_diag(note: str) -> None:
    if not DIAG_DOC.exists():
        return
    block = f"\n### Closed-loop tick ({now()})\n\n{note.strip()}\n"
    with DIAG_DOC.open("a") as f:
        f.write(block)


def parse_progress(console: str) -> dict:
    steps = re.findall(r"step=(\d+)\s+episode=(\d+)", console)
    out = {"step": None, "episode": None}
    if steps:
        out["step"], out["episode"] = map(int, steps[-1])
    return out


def classify_crash(console: str, status: dict) -> str:
    text = console[-80000:]
    if status.get("signal") == "SIGSEGV" or "Fatal Python error" in text or "SIGSEGV" in text:
        for name, spec in KNOWN.items():
            if name.startswith("segv") and all(m in text for m in spec["markers"][:2]):
                return name
        if "Embedding" in text or "sparse.py" in text:
            return "segv_embedding"
        if "_to_device" in text:
            return "segv_to_device"
        if "_encode_ongoing" in text or "torch.stack" in text:
            return "segv_stack_ongoing"
        return "segv_unknown"
    if "unhashable type: 'dict'" in text and ("deepcopy" in text or "TpaInfoPool" in text):
        return "deepcopy_typeerror"
    if status.get("state") == "failed":
        return "failed_other"
    return "unknown"


def ensure_patches() -> list[str]:
    """Sanity-check that overnight/morning patches are present; return missing."""
    missing = []
    checks = [
        (ROOT / "source/algo/hierarchical/hc_factory/tpa_info_pool.py", "clone_env_subtree"),
        (ROOT / "source/algo/hierarchical/hc_factory/hier_obs.py", "v if v.device == self.cuda_device"),
        (ROOT / "source/algo/hierarchical/hc_factory/decision_consistent.py", "encode_D(pre, ctx, pre=pre)"),
        (ROOT / "source/algo/hierarchical/hc_factory/hier_utils.py", "value.detach().cpu().clone()"),
    ]
    for path, needle in checks:
        if not path.exists() or needle not in path.read_text():
            missing.append(f"{path.name}:{needle[:40]}")
    return missing


def launch_train(tag: str, *, blocking: bool, emb_path: Path) -> tuple[subprocess.Popen, Path | None]:
    env = dict(os.environ)
    env.update(
        {
            "HC_R_RUN_TAG": tag,
            "HC_R_SEED": env.get("HC_R_SEED", "42"),
            "HC_MAX_HARD_EPISODES": env.get("HC_MAX_HARD_EPISODES", "100"),
            "HC_R_WANDB": env.get("HC_R_WANDB", "1"),
            "HC_EMB_DEBUG": "1",
            "HC_EMB_DEBUG_EVERY": env.get("HC_EMB_DEBUG_EVERY", "50"),
            "HC_EMB_DEBUG_NAMES": "next_logistic_id",
            "HC_EMB_DEBUG_PATH": str(emb_path),
            "PYTHONUNBUFFERED": "1",
            "PYTHONFAULTHANDLER": "1",
        }
    )
    if blocking:
        env["CUDA_LAUNCH_BLOCKING"] = "1"
        env["HC_CUDA_SYNC_ENCODE"] = "1"
    else:
        env.pop("CUDA_LAUNCH_BLOCKING", None)
        env["HC_CUDA_SYNC_ENCODE"] = "0"

    stdout_path = LOOP_DIR / f"{tag}_stdout.log"
    log = stdout_path.open("w")
    proc = subprocess.Popen(
        [PY, str(ROOT / "tools/run_r_series.py"), "R0", env.get("HC_R_DEVICE", "cuda:0")],
        cwd=ROOT,
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    (LOOP_DIR / "supervisor.train.pid").write_text(str(proc.pid))
    # Wait for evidence path line
    evidence = None
    for _ in range(120):
        time.sleep(0.5)
        text = stdout_path.read_text(errors="replace")
        m = re.search(r"evidence:\s+(\S+)", text)
        if m:
            evidence = Path(m.group(1))
            (LOOP_DIR / "evidence.path").write_text(str(evidence) + "\n")
            break
        if proc.poll() is not None:
            break
    return proc, evidence


def read_status(evidence: Path | None) -> dict:
    if not evidence:
        return {}
    sp = evidence / "status.json"
    if not sp.exists():
        return {}
    try:
        return json.loads(sp.read_text())
    except Exception:
        return {}


def read_console(evidence: Path | None) -> str:
    if not evidence:
        return ""
    cp = evidence / "console.log"
    if not cp.exists():
        return ""
    try:
        return cp.read_text(errors="replace")
    except Exception:
        return ""


def wait_finished(
    proc: subprocess.Popen,
    evidence: Path | None,
    state: dict,
    poll_s: float = 10.0,
    hb_s: float = 60.0,
):
    """Poll until train supervisor exits; write heartbeats. Returns final status dict."""
    last_step = None
    stall_since = None
    state["_last_hb"] = 0.0  # force immediate heartbeat
    while True:
        if STOP_PATH.exists() or state.get("stop"):
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
            state["phase"] = "stopped_by_user"
            state["stop"] = True
            save_state(state)
            try:
                proc.wait(timeout=30)
            except Exception:
                pass
            return read_status(evidence)

        rc = proc.poll()
        status = read_status(evidence)
        console = read_console(evidence)
        prog = parse_progress(console)

        # Also treat finalized failed status as exit even if poll lags.
        if rc is None and status.get("state") in ("finished", "failed"):
            rc = status.get("shell_exit_code", status.get("returncode", 1))

        # Stall detection (desk freeze often freezes logging without killing process)
        step = prog.get("step")
        if step is not None and step == last_step:
            if stall_since is None:
                stall_since = time.time()
            elif time.time() - stall_since > 600 and rc is None:
                tick = {
                    "at": now(),
                    "kind": "stall_warn",
                    "step": step,
                    "note": "no step progress >10min; possible desk freeze (gui+train GPU share)",
                }
                state.setdefault("ticks", []).append(tick)
                save_state(state)
                stall_since = time.time()  # re-arm
        else:
            stall_since = None
            last_step = step

        if time.time() - float(state.get("_last_hb") or 0) >= hb_s or rc is not None:
            state["_last_hb"] = time.time()
            tick = {
                "at": now(),
                "kind": "heartbeat" if rc is None else "exit",
                "tag": state.get("active_run", {}).get("tag"),
                "step": prog.get("step"),
                "episode": prog.get("episode"),
                "status": status.get("state"),
                "signal": status.get("signal"),
                "returncode": status.get("returncode"),
                "train_alive": rc is None,
                "segv": status.get("signal") == "SIGSEGV" or "Fatal Python error" in console[-20000:],
                "loop_pid": os.getpid(),
            }
            state.setdefault("ticks", []).append(tick)
            # keep ticks bounded
            if len(state["ticks"]) > 500:
                state["ticks"] = state["ticks"][-300:]
            state["progress"] = tick
            state["last_tick"] = tick["at"]
            save_state(state)
            print(json.dumps(tick), flush=True)

        if rc is not None:
            for _ in range(20):
                status = read_status(evidence)
                if status.get("state") in ("finished", "failed"):
                    break
                time.sleep(0.5)
            return status if status else {"state": "failed", "returncode": rc, "shell_exit_code": rc}

        time.sleep(poll_s)


def main() -> int:
    LOOP_DIR.mkdir(parents=True, exist_ok=True)
    max_relaunch = int(os.environ.get("HC_LOOP_MAX_RELAUNCH", "8"))
    base_tag = os.environ.get("HC_R_RUN_TAG", "R0-debug")

    missing = ensure_patches()
    prev = load_state() if STATE_PATH.exists() else {}
    resume = os.environ.get("HC_LOOP_RESUME", "1") != "0"
    relaunch = int(os.environ.get("HC_LOOP_START_RELAUNCH", "0"))
    blocking = os.environ.get("HC_LOOP_BLOCKING", "0") == "1"
    if resume and prev:
        relaunch = max(relaunch, int(prev.get("policy", {}).get("relaunch_count") or 0))
        blocking = blocking or bool(prev.get("policy", {}).get("cuda_launch_blocking"))
        # If previous active_run evidence shows a crash that was never recorded, count it.
        active = prev.get("active_run") or {}
        ev = active.get("evidence")
        if ev and Path(ev).joinpath("status.json").exists():
            st = json.loads(Path(ev).joinpath("status.json").read_text())
            if st.get("state") == "failed" and not any(
                h.get("evidence") == ev for h in (prev.get("history") or [])
            ):
                console = read_console(Path(ev))
                prog = parse_progress(console)
                kind = classify_crash(console, st)
                miss = {
                    "at": now(),
                    "tag": active.get("tag"),
                    "kind": kind,
                    "status": st,
                    "step": prog.get("step"),
                    "episode": prog.get("episode"),
                    "evidence": ev,
                    "note": "recovered after supervisor death (missed auto-relaunch)",
                }
                prev.setdefault("history", []).append(miss)
                prev["last_crash"] = miss
                prev.setdefault("findings", []).append(
                    f"MISSED: {active.get('tag')} {kind} @ step={prog.get('step')} (supervisor died)"
                )
                relaunch = max(relaunch, 1)
                if KNOWN.get(kind, {}).get("blocking_next"):
                    blocking = True
                append_diag(
                    f"- Recovered missed crash `{active.get('tag')}` `{kind}` "
                    f"step={prog.get('step')} evidence=`{ev}` (supervisor had died)."
                )

    state = {
        "goal": "Closed-loop R0 train (wandb R0-debug*) + auto-fix prior SIGSEGV/device-path crashes",
        "started_at": prev.get("started_at") or now(),
        "resumed_at": now() if prev else None,
        "phase": "running",
        "stop": False,
        "policy": {
            "variant": "R0",
            "wandb_tag_base": base_tag,
            "cuda_launch_blocking": blocking,
            "hc_emb_debug": True,
            "max_hard_episodes": int(os.environ.get("HC_MAX_HARD_EPISODES", "100")),
            "seed": int(os.environ.get("HC_R_SEED", "42")),
            "max_auto_relaunch": max_relaunch,
            "relaunch_count": relaunch,
            "heartbeat_sec": 60,
        },
        "patch_check_missing": missing,
        "ticks": list(prev.get("ticks") or [])[-100:],
        "findings": list(prev.get("findings") or []),
        "history": list(prev.get("history") or []),
        "last_crash": prev.get("last_crash"),
        "active_run": None,
    }
    if missing:
        state["findings"].append(f"WARNING missing patches: {missing}")
        print(f"[loop] WARNING missing patches: {missing}", flush=True)
    else:
        state["findings"].append(
            f"patches ok; continuous-field clone in _encode_ongoing; resume relaunch={relaunch} blocking={blocking}"
        )
    save_state(state)

    while True:
        if STOP_PATH.exists():
            state["phase"] = "stopped_by_user"
            state["stop"] = True
            save_state(state)
            return 0

        tag = base_tag if relaunch == 0 else f"{base_tag}-re{relaunch}"
        emb_path = LOOP_DIR / f"emb_debug_{tag}.log"
        print(f"[loop] launch tag={tag} blocking={blocking} wandb≈{tag}-N10", flush=True)
        proc, evidence = launch_train(tag, blocking=blocking, emb_path=emb_path)
        state["active_run"] = {
            "tag": tag,
            "pid": proc.pid,
            "evidence": str(evidence) if evidence else None,
            "blocking": blocking,
            "started_at": now(),
            "wandb_name_expected": f"{tag}-N10",
        }
        state["phase"] = "training"
        save_state(state)

        status = wait_finished(proc, evidence, state)
        console = read_console(evidence)
        prog = parse_progress(console)
        kind = classify_crash(console, status)

        result = {
            "at": now(),
            "tag": tag,
            "kind": kind,
            "status": status,
            "step": prog.get("step"),
            "episode": prog.get("episode"),
            "evidence": str(evidence) if evidence else None,
        }
        state["last_crash" if status.get("state") != "finished" else "last_success"] = result
        state.setdefault("history", []).append(result)
        save_state(state)

        if status.get("state") == "finished" and status.get("returncode", 1) == 0:
            state["phase"] = "success"
            state["stop"] = True
            state["findings"].append(f"{tag} finished OK step={prog.get('step')} ep={prog.get('episode')}")
            save_state(state)
            append_diag(f"- `{tag}` **finished OK** (step={prog.get('step')}, ep={prog.get('episode')}).")
            print(f"[loop] SUCCESS {tag}", flush=True)
            return 0

        # Crash path
        note = KNOWN.get(kind, {}).get("fix", "unknown crash — relaunch after logging")
        state["findings"].append(f"{tag}: {kind} @ step={prog.get('step')} — {note}")
        append_diag(
            f"- `{tag}` crash `{kind}` step={prog.get('step')} ep={prog.get('episode')} "
            f"signal={status.get('signal')} evidence=`{evidence}`\n"
            f"  Action: {note}"
        )
        print(f"[loop] CRASH {kind} tag={tag} step={prog.get('step')}", flush=True)

        if KNOWN.get(kind, {}).get("blocking_next"):
            blocking = True
            state["policy"]["cuda_launch_blocking"] = True

        relaunch += 1
        state["policy"]["relaunch_count"] = relaunch
        if relaunch > max_relaunch:
            state["phase"] = "stopped_relaunch_budget"
            state["stop"] = True
            save_state(state)
            print("[loop] relaunch budget exhausted", flush=True)
            return 1

        state["phase"] = "relaunching"
        state["active_run"] = None
        save_state(state)
        time.sleep(5)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        st = load_state()
        st["phase"] = "interrupted"
        st["stop"] = True
        save_state(st)
        raise
