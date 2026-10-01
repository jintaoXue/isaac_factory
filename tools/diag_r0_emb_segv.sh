#!/usr/bin/env bash
# Isolation launches for R0 Embedding SIGSEGV (see docs/training_crash_diagnosis_2026-09-29.md).
# Usage:
#   bash tools/diag_r0_emb_segv.sh cuda-blocking
#   bash tools/diag_r0_emb_segv.sh cpu
#   bash tools/diag_r0_emb_segv.sh no-wandb
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
CASE="${1:?case: cuda-blocking|cpu|no-wandb}"
export HC_MAX_HARD_EPISODES="${HC_MAX_HARD_EPISODES:-3}"
export HC_R_SEED="${HC_R_SEED:-42}"
export HC_EMB_DEBUG="${HC_EMB_DEBUG:-1}"
export HC_EMB_DEBUG_EVERY="${HC_EMB_DEBUG_EVERY:-50}"
export HC_EMB_DEBUG_NAMES="${HC_EMB_DEBUG_NAMES:-next_logistic_id}"
TAG="diag-emb-${CASE}-S${HC_R_SEED}"
export HC_R_RUN_TAG="${HC_R_RUN_TAG:-$TAG}"
export HC_EMB_DEBUG_PATH="${HC_EMB_DEBUG_PATH:-$ROOT/outputs/train_monitor/${TAG}_emb_debug.log}"
DEVICE="cuda:0"
case "$CASE" in
  cuda-blocking)
    export CUDA_LAUNCH_BLOCKING=1
    export HC_CUDA_SYNC_ENCODE=1
    export HC_R_WANDB="${HC_R_WANDB:-1}"
    ;;
  cpu)
    DEVICE="cpu"
    unset CUDA_LAUNCH_BLOCKING || true
    export HC_CUDA_SYNC_ENCODE=0
    export HC_R_WANDB="${HC_R_WANDB:-0}"
    ;;
  no-wandb)
    unset CUDA_LAUNCH_BLOCKING || true
    export HC_CUDA_SYNC_ENCODE=0
    export HC_R_WANDB=0
    ;;
  *)
    echo "unknown case: $CASE" >&2
    exit 2
    ;;
esac
echo "[diag] case=$CASE device=$DEVICE episodes=$HC_MAX_HARD_EPISODES emb_log=$HC_EMB_DEBUG_PATH"
echo "[diag] CUDA_LAUNCH_BLOCKING=${CUDA_LAUNCH_BLOCKING:-} HC_CUDA_SYNC_ENCODE=${HC_CUDA_SYNC_ENCODE:-} HC_R_WANDB=$HC_R_WANDB"
exec python ./tools/run_r_series.py R0 "$DEVICE"
