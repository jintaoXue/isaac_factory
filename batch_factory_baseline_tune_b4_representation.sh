#!/usr/bin/env bash
# Fixed-layout identity x history-readout ablation; validation-only selection.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
[[ "$(git branch --show-current)" == dev_xwt ]] || { echo 'Expected dev_xwt'; exit 1; }
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda:0}"
TAG="${BENCHMARK_TAG:-factory_pdformer_134_v1}"
SEARCH="${TUNING_TAG:-b4_representation_v1}"
DATASET_DIR="$ROOT/source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/output/bottleneck_dataset/experiments/$TAG"
TOOLS="$ROOT/source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/tools"
OUT="$DATASET_DIR/models/tuning/$SEARCH"
read -r -a SEEDS <<< "${TUNE_SEEDS:-42 43}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
[[ -f "$DATASET_DIR/dataset.pt" ]] || { echo "Missing dataset: $DATASET_DIR"; exit 1; }
[[ ! -e "$OUT" ]] || { echo "Refusing to overwrite: $OUT"; exit 1; }
mkdir -p "$OUT"
git rev-parse HEAD > "$OUT/source_commit.txt"
for seed in "${SEEDS[@]}"; do
  for candidate in identity history identity_history control; do
    OPTIONS=()
    case "$candidate" in
      identity) OPTIONS=(--node_embedding 16);;
      history) OPTIONS=(--temporal_readout last_mean);;
      identity_history) OPTIONS=(--node_embedding 16 --temporal_readout last_mean);;
    esac
    DEST="$OUT/candidate_$candidate/seed$seed"
    mkdir -p "$DEST"
    printf '\nB4 candidate=%s seed=%s validation only\n' "$candidate" "$seed"
    "$PYTHON_BIN" -u "$TOOLS/train_b4_gcn_gru.py" \
      --dataset_dir "$DATASET_DIR" --output_dir "$DEST" \
      --training_profile "${SEARCH}_${candidate}" --validation_only \
      --seed "$seed" --device "$DEVICE" --batch_size 24 \
      --gcn_hidden 64 --gru_hidden 128 --dropout 0.20 \
      --learning_rate 0.0003 --weight_decay 0.01 \
      --max_epochs "${MAX_EPOCHS:-60}" --min_epochs 10 --patience 10 \
      --event_focal_gamma 0 --lambda_event_will 2.5 \
      --event_will_upcoming_pos_weight 4 --event_will_ongoing_pos_weight 3 \
      --event_will_fp_weight 2 \
      --checkpoint_min_report_precision 0.80 --checkpoint_min_report_recall 0.35 \
      --report_threshold_sweep 0.55 0.60 0.62 0.65 0.68 0.70 0.72 0.75 0.78 0.80 0.82 0.85 \
      "${OPTIONS[@]}" 2>&1 | tee "$DEST/training.log"
  done
done
"$PYTHON_BIN" "$TOOLS/select_baseline_tuning.py" --tuning_dir "$OUT" --expected_seeds "${SEEDS[@]}"
