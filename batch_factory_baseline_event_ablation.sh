#!/usr/bin/env bash
# Controlled validation-only context x focal ablation. No test evaluation.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
[[ "$(git branch --show-current)" == dev_xwt ]] || { echo 'Expected dev_xwt'; exit 1; }
MODEL="${1:?Usage: bash batch_factory_baseline_event_ablation.sh B3|B4|B5}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda:0}"
TAG="${BENCHMARK_TAG:-factory_pdformer_134_v1}"
SEARCH="${TUNING_TAG:-${MODEL,,}_event_ablation_v1}"
DATASET_DIR="${DATASET_DIR:-$ROOT/source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/output/bottleneck_dataset/experiments/$TAG}"
TOOLS="$ROOT/source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/tools"
OUT="$DATASET_DIR/models/tuning/$SEARCH"
read -r -a SEEDS <<< "${TUNE_SEEDS:-42 43}"
read -r -a CANDIDATES <<< "${CANDIDATES:-context focal context_focal control}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
case "$MODEL" in
  B3) ENTRY=train_b3_lstm.py; CONFIG=(--batch_size 32 --learning_rate 0.0002 --weight_decay 0.01 --dropout 0.2);;
  B4) ENTRY=train_b4_gcn_gru.py; CONFIG=(--batch_size 24 --learning_rate 0.0003 --weight_decay 0.01 --dropout 0.2);;
  B5) ENTRY=train_b5_gat_gru.py; CONFIG=(--batch_size 16 --learning_rate 0.00015 --weight_decay 0.01 --dropout 0.2);;
  *) echo "Unknown model: $MODEL"; exit 1;;
esac
[[ -f "$DATASET_DIR/dataset.pt" ]] || { echo "Missing dataset: $DATASET_DIR"; exit 1; }
[[ ! -e "$OUT" ]] || { echo "Refusing to overwrite: $OUT"; exit 1; }
for candidate in "${CANDIDATES[@]}"; do
  case "$candidate" in context|focal|context_focal|control) ;; *) echo "Unknown candidate: $candidate"; exit 1;; esac
done
mkdir -p "$OUT"
git rev-parse HEAD > "$OUT/source_commit.txt"
for seed in "${SEEDS[@]}"; do
  for candidate in "${CANDIDATES[@]}"; do
    OPTIONS=()
    case "$candidate" in
      context) OPTIONS=(--event_context);;
      focal) OPTIONS=(--event_focal_gamma 2);;
      context_focal) OPTIONS=(--event_context --event_focal_gamma 2);;
    esac
    DEST="$OUT/candidate_$candidate/seed$seed"
    mkdir -p "$DEST"
    printf '\nModel=%s candidate=%s seed=%s validation only\n' "$MODEL" "$candidate" "$seed"
    "$PYTHON_BIN" -u "$TOOLS/$ENTRY" --dataset_dir "$DATASET_DIR" \
      --output_dir "$DEST" --training_profile "${SEARCH}_${candidate}" \
      --validation_only --seed "$seed" --device "$DEVICE" \
      --max_epochs "${MAX_EPOCHS:-60}" --min_epochs "${MIN_EPOCHS:-10}" \
      --patience "${PATIENCE:-10}" \
      --checkpoint_min_report_precision 0.80 --checkpoint_min_report_recall 0.35 \
      --report_threshold_sweep 0.55 0.60 0.62 0.65 0.68 0.70 0.72 0.75 0.78 0.80 0.82 0.85 \
      "${CONFIG[@]}" "${OPTIONS[@]}" 2>&1 | tee "$DEST/training.log"
  done
done
"$PYTHON_BIN" "$TOOLS/select_baseline_tuning.py" --tuning_dir "$OUT" --expected_seeds "${SEEDS[@]}"
