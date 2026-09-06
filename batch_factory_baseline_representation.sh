#!/usr/bin/env bash
# B4/B5 fixed-layout identity x history-readout ablation; validation-only selection.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
[[ "$(git branch --show-current)" == dev_xwt ]] || { echo 'Expected dev_xwt'; exit 1; }
MODEL="${1:?Usage: bash batch_factory_baseline_representation.sh B4|B5}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda:0}"
TAG="${BENCHMARK_TAG:-factory_pdformer_134_v3}"
SEARCH="${TUNING_TAG:-b${MODEL#B}_representation_v1}"
DATASET_DIR="${DATASET_DIR:-$ROOT/source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/output/bottleneck_dataset/experiments/$TAG}"
TOOLS="$ROOT/source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/tools"
OUT="$DATASET_DIR/models/tuning/$SEARCH"
read -r -a SEEDS <<< "${TUNE_SEEDS:-42 43}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
case "$MODEL" in
  B4) ENTRY=train_b4_gcn_gru.py; CONFIG=(--batch_size 24 --gcn_hidden 64 --learning_rate 0.0003 --min_epochs 10 --patience 10);;
  B5) ENTRY=train_b5_gat_gru.py; CONFIG=(--batch_size 16 --gat_hidden 64 --gat_heads 4 --learning_rate 0.00015 --min_epochs 15 --patience 20);;
  *) echo "Unknown model: $MODEL"; exit 1;;
esac
[[ -f "$DATASET_DIR/dataset.pt" ]] || { echo "Missing dataset: $DATASET_DIR"; exit 1; }
[[ ! -e "$OUT" ]] || { echo "Refusing to overwrite: $OUT"; exit 1; }
"$PYTHON_BIN" - "$DATASET_DIR" <<'PY'
import hashlib, json, sys
from pathlib import Path
root = Path(sys.argv[1])
split_path = root / "episode_split_audit.json"
split = json.loads(split_path.read_text())
audit = json.loads((root / "validation_contract_audit.json").read_text())
if not split["episode_split_match"] or not audit["comparison_match"]:
    raise SystemExit("Resolve the common-data audit before starting representation trials")
if hashlib.sha256(split_path.read_bytes()).hexdigest() != audit["split_audit_sha256"]:
    raise SystemExit("Episode split audit differs from the validated artifact")
manifest_hash = hashlib.sha256((root / "dataset_manifest.json").read_bytes()).hexdigest()
if manifest_hash != split["provenance"]["baseline_manifest"]["sha256"]:
    raise SystemExit("Dataset manifest changed after the common-data audit")
print("Common validation input/target audit matched; manifest:", manifest_hash)
PY
mkdir -p "$OUT"
git rev-parse HEAD > "$OUT/source_commit.txt"
for seed in "${SEEDS[@]}"; do
  for candidate in control identity history identity_history; do
    IDENTITY=0
    READOUT=last
    case "$candidate" in
      identity) IDENTITY=16;;
      history) READOUT=last_mean;;
      identity_history) IDENTITY=16; READOUT=last_mean;;
    esac
    DEST="$OUT/candidate_$candidate/seed$seed"
    mkdir -p "$DEST"
    printf '\n%s candidate=%s seed=%s validation only\n' "$MODEL" "$candidate" "$seed"
    "$PYTHON_BIN" -u "$TOOLS/$ENTRY" \
      --dataset_dir "$DATASET_DIR" --output_dir "$DEST" \
      --training_profile "${SEARCH}_${candidate}" --validation_only \
      --seed "$seed" --device "$DEVICE" --gru_hidden 128 --dropout 0.20 \
      --weight_decay 0.01 --max_epochs "${MAX_EPOCHS:-60}" \
      --event_focal_gamma 0 --lambda_event_will 2.5 \
      --event_will_upcoming_pos_weight 4 --event_will_ongoing_pos_weight 3 \
      --event_will_fp_weight 2 \
      --checkpoint_min_report_precision 0.80 --checkpoint_min_report_recall 0.35 \
      --report_threshold_sweep 0.55 0.60 0.62 0.65 0.68 0.70 0.72 0.75 0.78 0.80 0.82 0.85 \
      "${CONFIG[@]}" --node_embedding "$IDENTITY" --temporal_readout "$READOUT" \
      2>&1 | tee "$DEST/training.log"
  done
done
"$PYTHON_BIN" "$TOOLS/select_baseline_tuning.py" --tuning_dir "$OUT" --expected_seeds "${SEEDS[@]}"
