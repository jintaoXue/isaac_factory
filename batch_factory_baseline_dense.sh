#!/usr/bin/env bash
set -euo pipefail

# Reuse the established benchmark and model directories; never create new ones.
REPO="$(git rev-parse --show-toplevel)"
[[ "$(git branch --show-current)" == dev_xwt && "$(basename "$REPO")" == BSTAN_isaac_factory ]] || {
  printf '%s\n' 'Run in the server BSTAN_isaac_factory repository on dev_xwt.' >&2
  exit 1
}
FACTORY="$REPO/source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory"
TOOLS="$FACTORY/tools"
DATASET_DIR="${DATASET_DIR:-$FACTORY/output/bottleneck_dataset/experiments/factory_pdformer_134_v3}"
RAW_ROOT="${RAW_ROOT:-/home/sci/work/BNPDFormer/_isaac_factory/source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/output/bottleneck_dataset}"
PYTHON="${PYTHON:-python}"
ARCHIVE_TAG="${ARCHIVE_TAG:?Set a unique ARCHIVE_TAG for files retained before this run}"
export PYTHONDONTWRITEBYTECODE=1

case "${1:-}" in
  audit|build)
    names=(human+log human+mach human+mat mach+log mach+mat mat+log
           human+log+mat mach+human+log mach+human+mat mach+log+mat four_dim norm20)
    additions=()
    for name in "${names[@]}"; do additions+=("$RAW_ROOT/$name"); done
    options=()
    [[ "$1" != build ]] || options+=(--apply)
    "$PYTHON" -u "$TOOLS/rebuild_dense_factory_benchmark.py" \
      --benchmark_dir "$DATASET_DIR" --archive_tag "$ARCHIVE_TAG" \
      --additional_run_dirs "${additions[@]}" "${options[@]}"
    ;;
  B4|B5|ALL)
    models=("$1")
    [[ "$1" != ALL ]] || models=(B4 B5)
    read -r -a seeds <<< "${TRAIN_SEEDS:-42 43}"
    # Check all destinations before the first archive or training call.
    for model in "${models[@]}"; do
      lower="$(printf '%s' "$model" | tr '[:upper:]' '[:lower:]')"
      for seed in "${seeds[@]}"; do
        [[ "$seed" =~ ^[0-9]+$ ]] || { printf '%s\n' 'Invalid seed' >&2; exit 1; }
        dir="$DATASET_DIR/models/tuning/${lower}_representation_v1/candidate_history/seed$seed"
        [[ -d "$dir" ]] || { printf 'Missing existing model directory: %s\n' "$dir" >&2; exit 1; }
      done
    done
    for model in "${models[@]}"; do
      lower="$(printf '%s' "$model" | tr '[:upper:]' '[:lower:]')"
      for seed in "${seeds[@]}"; do
        "$PYTHON" -u "$TOOLS/train_dense_baseline_control.py" \
          --model "$model" --dataset_dir "$DATASET_DIR" --seed "$seed" \
          --output_dir "$DATASET_DIR/models/tuning/${lower}_representation_v1/candidate_history/seed$seed" \
          --archive_tag "$ARCHIVE_TAG" --device "${DEVICE:-cuda:0}"
      done
    done
    ;;
  *) printf 'Usage: bash %s audit|build|B4|B5|ALL\n' "$0" >&2; exit 1 ;;
esac
