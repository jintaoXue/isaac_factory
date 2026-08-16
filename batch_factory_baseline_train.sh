#!/bin/bash
# Train B2-B5 against one shared dataset and split.
#
# Usage (repository root):
#   ./batch_factory_baseline_train.sh B5
#   ./batch_factory_baseline_train.sh ALL
#   RUN_MODE=smoke ./batch_factory_baseline_train.sh ALL
#
# Environment overrides:
#   BENCHMARK_TAG=shared_bstan_machine_human_v1
#   DEVICE=cuda:0 SEED=42 MAX_EPOCHS=100 PATIENCE=15 N_JOBS=8

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${ROOT}/source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/output/bottleneck_dataset"
TOOLS_DIR="${ROOT}/source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/tools"
BENCHMARK_TAG="${BENCHMARK_TAG:-shared_bstan_machine_human_v1}"
DATASET_DIR="${DATA_ROOT}/experiments/${BENCHMARK_TAG}"
MODEL_ROOT="${DATASET_DIR}/models"

DEVICE="${DEVICE:-cuda:0}"
SEED="${SEED:-42}"
MAX_EPOCHS="${MAX_EPOCHS:-100}"
PATIENCE="${PATIENCE:-15}"
N_JOBS="${N_JOBS:-8}"
RUN_MODE="${RUN_MODE:-formal}"

if [ $# -eq 0 ]; then
    echo "Usage: $0 <B2|B3|B4|B5|ALL> [B2|B3|B4|B5 ...]"
    echo "  RUN_MODE=smoke uses 5 XGBoost estimators and one PyTorch epoch."
    exit 1
fi

if [ "$RUN_MODE" = "smoke" ]; then
    MAX_EPOCHS=1
    PATIENCE=1
    XGB_ESTIMATORS=5
    OUTPUT_SUFFIX="_smoke"
elif [ "$RUN_MODE" = "formal" ]; then
    XGB_ESTIMATORS="${XGB_ESTIMATORS:-300}"
    OUTPUT_SUFFIX=""
else
    echo "RUN_MODE must be smoke or formal, got: ${RUN_MODE}" >&2
    exit 1
fi

MODELS=()
for arg in "$@"; do
    case "$arg" in
        ALL) MODELS+=(B2 B3 B4 B5) ;;
        B2|B3|B4|B5) MODELS+=("$arg") ;;
        *)
            echo "Unknown baseline: $arg" >&2
            exit 1
            ;;
    esac
done

# Deduplicate while preserving the requested order.
DEDUPED=()
SEEN=""
for model in "${MODELS[@]}"; do
    case " ${SEEN} " in
        *" ${model} "*) continue ;;
    esac
    SEEN="${SEEN} ${model}"
    DEDUPED+=("$model")
done
MODELS=("${DEDUPED[@]}")

for required in dataset.pt dataset_manifest.json split_manifest.json; do
    if [ ! -f "${DATASET_DIR}/${required}" ]; then
        echo "Missing ${DATASET_DIR}/${required}; run batch_factory_baseline_build.sh first." >&2
        exit 1
    fi
done

mkdir -p "$MODEL_ROOT"
echo "Dataset: ${DATASET_DIR}"
echo "Models: ${MODELS[*]}"
echo "Mode: ${RUN_MODE}  seed=${SEED}  device=${DEVICE}"

train_one() {
    local model=$1
    case "$model" in
        B2)
            python -c "import xgboost; print('xgboost', xgboost.__version__)"
            python "${TOOLS_DIR}/train_b2_xgboost.py" \
                --dataset_dir "$DATASET_DIR" \
                --output_dir "${MODEL_ROOT}/b2_xgboost_seed${SEED}${OUTPUT_SUFFIX}" \
                --seed "$SEED" \
                --n_estimators "$XGB_ESTIMATORS" \
                --n_jobs "$N_JOBS"
            ;;
        B3)
            python "${TOOLS_DIR}/train_b3_lstm.py" \
                --dataset_dir "$DATASET_DIR" \
                --output_dir "${MODEL_ROOT}/b3_lstm_seed${SEED}${OUTPUT_SUFFIX}" \
                --device "$DEVICE" \
                --seed "$SEED" \
                --max_epochs "$MAX_EPOCHS" \
                --patience "$PATIENCE"
            ;;
        B4)
            python "${TOOLS_DIR}/train_b4_gcn_gru.py" \
                --dataset_dir "$DATASET_DIR" \
                --output_dir "${MODEL_ROOT}/b4_gcn_gru_seed${SEED}${OUTPUT_SUFFIX}" \
                --device "$DEVICE" \
                --seed "$SEED" \
                --max_epochs "$MAX_EPOCHS" \
                --patience "$PATIENCE"
            ;;
        B5)
            python "${TOOLS_DIR}/train_b5_gat_gru.py" \
                --dataset_dir "$DATASET_DIR" \
                --output_dir "${MODEL_ROOT}/b5_gat_gru_seed${SEED}${OUTPUT_SUFFIX}" \
                --device "$DEVICE" \
                --seed "$SEED" \
                --max_epochs "$MAX_EPOCHS" \
                --patience "$PATIENCE"
            ;;
    esac
}

for model in "${MODELS[@]}"; do
    echo "========================================"
    echo "[$(date '+%F %T')] Training ${model}"
    echo "========================================"
    train_one "$model"
done

echo "[$(date '+%F %T')] Requested baselines completed."
echo "Outputs: ${MODEL_ROOT}"
