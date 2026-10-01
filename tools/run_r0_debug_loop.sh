#!/usr/bin/env bash
# Durable outer watchdog for tools/closed_loop_r0_debug.py.
# Restarts the Python closed-loop if it dies unexpectedly (unless STOP or success).
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
LOOP_DIR="$ROOT/outputs/train_monitor/r0_debug_loop"
mkdir -p "$LOOP_DIR"
PY="${HC_PYTHON:-/home/xue/Repos/miniconda3/envs/isaaclab/bin/python}"
export HC_R_RUN_TAG="${HC_R_RUN_TAG:-R0-debug}"
export HC_R_SEED="${HC_R_SEED:-42}"
export HC_MAX_HARD_EPISODES="${HC_MAX_HARD_EPISODES:-100}"
export HC_R_WANDB="${HC_R_WANDB:-1}"
export HC_LOOP_MAX_RELAUNCH="${HC_LOOP_MAX_RELAUNCH:-8}"
export HC_PYTHON="$PY"

echo "$$" > "$LOOP_DIR/watchdog.pid"
echo "[watchdog] start $(date -Iseconds) pid=$$ tag=$HC_R_RUN_TAG" | tee -a "$LOOP_DIR/watchdog.log"

while true; do
  if [[ -f "$LOOP_DIR/STOP" ]]; then
    echo "[watchdog] STOP file present; exit" | tee -a "$LOOP_DIR/watchdog.log"
    exit 0
  fi
  # If STATE says success/stopped, do not respawn.
  if [[ -f "$LOOP_DIR/STATE.json" ]]; then
    phase=$(python3 -c "import json;print(json.load(open('$LOOP_DIR/STATE.json')).get('phase',''))" 2>/dev/null || true)
    stop=$(python3 -c "import json;print(json.load(open('$LOOP_DIR/STATE.json')).get('stop',False))" 2>/dev/null || true)
    if [[ "$phase" == "success" || "$stop" == "True" ]]; then
      echo "[watchdog] STATE phase=$phase stop=$stop; exit" | tee -a "$LOOP_DIR/watchdog.log"
      exit 0
    fi
  fi
  echo "[watchdog] launching closed_loop $(date -Iseconds)" | tee -a "$LOOP_DIR/watchdog.log"
  "$PY" "$ROOT/tools/closed_loop_r0_debug.py" >>"$LOOP_DIR/loop_stdout.log" 2>&1
  rc=$?
  echo "[watchdog] closed_loop exited rc=$rc at $(date -Iseconds)" | tee -a "$LOOP_DIR/watchdog.log"
  if [[ -f "$LOOP_DIR/STOP" ]]; then
    exit 0
  fi
  phase=$(python3 -c "import json;print(json.load(open('$LOOP_DIR/STATE.json')).get('phase',''))" 2>/dev/null || true)
  if [[ "$phase" == "success" || "$phase" == "stopped_by_user" || "$phase" == "stopped_relaunch_budget" ]]; then
    echo "[watchdog] terminal phase=$phase; exit" | tee -a "$LOOP_DIR/watchdog.log"
    exit "$rc"
  fi
  # Unexpected death of supervisor — respawn after short backoff.
  sleep 10
done
