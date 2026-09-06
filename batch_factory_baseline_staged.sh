#!/usr/bin/env bash
# Preregistered weights-only continuation vs same-cap scratch training.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="${1:?Usage: bash batch_factory_baseline_staged.sh B4|B5}"
case "$MODEL" in B4|B5) ;; *) echo "Expected B4 or B5" >&2; exit 1;; esac
BASE="b${MODEL#B}"
TAG="${BENCHMARK_TAG:-factory_pdformer_134_v3}"
DATASET_DIR="${DATASET_DIR:-$ROOT/source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/output/bottleneck_dataset/experiments/$TAG}"
PARENT_DIR="${PARENT_DIR:-$DATASET_DIR/models/tuning/${BASE}_representation_v1}"
TUNING_TAG="${TUNING_TAG:-${BASE}_staged_training_v1}"
read -r -a SEEDS <<< "${TUNE_SEEDS:-42 43}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
exec "${PYTHON_BIN:-python}" -u \
  "$ROOT/source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/tools/run_staged_baseline.py" \
  --model "$MODEL" --dataset_dir "$DATASET_DIR" --parent_dir "$PARENT_DIR" \
  --output_dir "$DATASET_DIR/models/tuning/$TUNING_TAG" \
  --seeds "${SEEDS[@]}" --device "${DEVICE:-cuda:0}" --threads "$OMP_NUM_THREADS"
