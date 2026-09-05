#!/bin/bash
# Train B2-B5 against one shared dataset and split.
#
# Usage (repository root):
#   ./batch_factory_baseline_train.sh B5
#   ./batch_factory_baseline_train.sh ALL
#   RUN_MODE=smoke ./batch_factory_baseline_train.sh ALL
#
# Environment overrides:
#   BENCHMARK_TAG=factory_main_aligned_v1 DEVICE=cuda:0 SEED=42 N_JOBS=8
#   B3_LEARNING_RATE=0.0003 B4_DROPOUT=0.2 B5_WEIGHT_DECAY=0.01

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${ROOT}/source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/output/bottleneck_dataset"
TOOLS_DIR="${ROOT}/source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/tools"
BENCHMARK_TAG="${BENCHMARK_TAG:-factory_main_aligned_v1}"
DATASET_DIR="${DATA_ROOT}/experiments/${BENCHMARK_TAG}"
MODEL_ROOT="${DATASET_DIR}/models"

DEVICE="${DEVICE:-cuda:0}"
SEED="${SEED:-42}"
TRAINING_PROFILE="${TRAINING_PROFILE:-baseline_fair_v2}"

# Model-specific optimization is fixed before test evaluation. The shared data,
# targets, metric protocol and validation-only threshold selection stay identical.
B2_XGB_ESTIMATORS="${B2_XGB_ESTIMATORS:-500}"
B2_MAX_DEPTH="${B2_MAX_DEPTH:-5}"
B2_LEARNING_RATE="${B2_LEARNING_RATE:-0.03}"
B2_MIN_CHILD_WEIGHT="${B2_MIN_CHILD_WEIGHT:-3.0}"
B2_REG_LAMBDA="${B2_REG_LAMBDA:-5.0}"

B3_BATCH_SIZE="${B3_BATCH_SIZE:-32}"
B3_MAX_EPOCHS="${B3_MAX_EPOCHS:-60}"
B3_PATIENCE="${B3_PATIENCE:-15}"
B3_MIN_EPOCHS="${B3_MIN_EPOCHS:-15}"
B3_LEARNING_RATE="${B3_LEARNING_RATE:-0.0003}"
B3_WEIGHT_DECAY="${B3_WEIGHT_DECAY:-0.001}"
B3_LR_MIN="${B3_LR_MIN:-0.000001}"
B3_LR_SCHEDULE="${B3_LR_SCHEDULE:-cosine}"
B3_LSTM_HIDDEN="${B3_LSTM_HIDDEN:-128}"
B3_NODE_HIDDEN="${B3_NODE_HIDDEN:-128}"
B3_NODE_EMBEDDING="${B3_NODE_EMBEDDING:-32}"
B3_DROPOUT="${B3_DROPOUT:-0.25}"

B4_BATCH_SIZE="${B4_BATCH_SIZE:-24}"
B4_MAX_EPOCHS="${B4_MAX_EPOCHS:-60}"
B4_PATIENCE="${B4_PATIENCE:-10}"
B4_MIN_EPOCHS="${B4_MIN_EPOCHS:-10}"
B4_LEARNING_RATE="${B4_LEARNING_RATE:-0.0003}"
B4_WEIGHT_DECAY="${B4_WEIGHT_DECAY:-0.01}"
B4_LR_MIN="${B4_LR_MIN:-0.000001}"
B4_LR_SCHEDULE="${B4_LR_SCHEDULE:-cosine}"
B4_GCN_HIDDEN="${B4_GCN_HIDDEN:-64}"
B4_GRU_HIDDEN="${B4_GRU_HIDDEN:-128}"
B4_DROPOUT="${B4_DROPOUT:-0.20}"

B5_BATCH_SIZE="${B5_BATCH_SIZE:-16}"
B5_MAX_EPOCHS="${B5_MAX_EPOCHS:-60}"
B5_PATIENCE="${B5_PATIENCE:-20}"
B5_MIN_EPOCHS="${B5_MIN_EPOCHS:-15}"
B5_LEARNING_RATE="${B5_LEARNING_RATE:-0.00015}"
B5_WEIGHT_DECAY="${B5_WEIGHT_DECAY:-0.01}"
B5_LR_MIN="${B5_LR_MIN:-0.000001}"
B5_LR_SCHEDULE="${B5_LR_SCHEDULE:-cosine}"
B5_GAT_HIDDEN="${B5_GAT_HIDDEN:-64}"
B5_GAT_HEADS="${B5_GAT_HEADS:-4}"
B5_GRU_HIDDEN="${B5_GRU_HIDDEN:-128}"
B5_DROPOUT="${B5_DROPOUT:-0.20}"

HOT_EVAL_THRESHOLD="${HOT_EVAL_THRESHOLD:-0.55}"
EVENT_REPORT_THRESHOLD="${EVENT_REPORT_THRESHOLD:-0.68}"
CHECKPOINT_MIN_REPORT_PRECISION="${CHECKPOINT_MIN_REPORT_PRECISION:-0.80}"
CHECKPOINT_MIN_REPORT_RECALL="${CHECKPOINT_MIN_REPORT_RECALL:-0.35}"
N_JOBS="${N_JOBS:-8}"
RUN_MODE="${RUN_MODE:-formal}"

if [ $# -eq 0 ]; then
    echo "Usage: $0 <B2|B3|B4|B5|ALL> [B2|B3|B4|B5 ...]"
    echo "  RUN_MODE=smoke uses 5 XGBoost estimators and one PyTorch epoch."
    exit 1
fi

if [ "$RUN_MODE" = "smoke" ]; then
    B2_XGB_ESTIMATORS=5
    B3_MAX_EPOCHS=1
    B3_PATIENCE=1
    B3_MIN_EPOCHS=1
    B4_MAX_EPOCHS=1
    B4_PATIENCE=1
    B4_MIN_EPOCHS=1
    B5_MAX_EPOCHS=1
    B5_PATIENCE=1
    B5_MIN_EPOCHS=1
    OUTPUT_SUFFIX="_smoke"
elif [ "$RUN_MODE" = "formal" ]; then
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
echo "Mode: ${RUN_MODE}  profile=${TRAINING_PROFILE}  seed=${SEED}  device=${DEVICE}"
echo "Evaluation: hot_threshold=${HOT_EVAL_THRESHOLD} report_threshold_default=${EVENT_REPORT_THRESHOLD} checkpoint_min_precision=${CHECKPOINT_MIN_REPORT_PRECISION} checkpoint_min_recall=${CHECKPOINT_MIN_REPORT_RECALL}"

train_one() {
    local model=$1
    case "$model" in
        B2)
            echo "B2 config: trees=${B2_XGB_ESTIMATORS} depth=${B2_MAX_DEPTH} lr=${B2_LEARNING_RATE} min_child_weight=${B2_MIN_CHILD_WEIGHT} reg_lambda=${B2_REG_LAMBDA}"
            python -c "import sklearn, xgboost; print('xgboost', xgboost.__version__, 'scikit-learn', sklearn.__version__)"
            python "${TOOLS_DIR}/train_b2_xgboost.py" \
                --dataset_dir "$DATASET_DIR" \
                --output_dir "${MODEL_ROOT}/b2_xgboost_seed${SEED}${OUTPUT_SUFFIX}" \
                --seed "$SEED" \
                --training_profile "$TRAINING_PROFILE" \
                --n_estimators "$B2_XGB_ESTIMATORS" \
                --max_depth "$B2_MAX_DEPTH" \
                --learning_rate "$B2_LEARNING_RATE" \
                --min_child_weight "$B2_MIN_CHILD_WEIGHT" \
                --reg_lambda "$B2_REG_LAMBDA" \
                --n_jobs "$N_JOBS" \
                --hot_eval_threshold "$HOT_EVAL_THRESHOLD" \
                --event_report_threshold "$EVENT_REPORT_THRESHOLD" \
                --report_threshold_min_precision "$CHECKPOINT_MIN_REPORT_PRECISION" \
                --checkpoint_min_report_recall "$CHECKPOINT_MIN_REPORT_RECALL"
            ;;
        B3)
            echo "B3 config: epochs=${B3_MAX_EPOCHS} min=${B3_MIN_EPOCHS} patience=${B3_PATIENCE} batch=${B3_BATCH_SIZE} lr=${B3_LEARNING_RATE} wd=${B3_WEIGHT_DECAY} dropout=${B3_DROPOUT}"
            python "${TOOLS_DIR}/train_b3_lstm.py" \
                --dataset_dir "$DATASET_DIR" \
                --output_dir "${MODEL_ROOT}/b3_lstm_seed${SEED}${OUTPUT_SUFFIX}" \
                --device "$DEVICE" \
                --seed "$SEED" \
                --training_profile "$TRAINING_PROFILE" \
                --batch_size "$B3_BATCH_SIZE" \
                --max_epochs "$B3_MAX_EPOCHS" \
                --patience "$B3_PATIENCE" \
                --min_epochs "$B3_MIN_EPOCHS" \
                --learning_rate "$B3_LEARNING_RATE" \
                --weight_decay "$B3_WEIGHT_DECAY" \
                --lr_min "$B3_LR_MIN" \
                --lr_schedule "$B3_LR_SCHEDULE" \
                --lstm_hidden "$B3_LSTM_HIDDEN" \
                --node_hidden "$B3_NODE_HIDDEN" \
                --node_embedding "$B3_NODE_EMBEDDING" \
                --dropout "$B3_DROPOUT" \
                --hot_eval_threshold "$HOT_EVAL_THRESHOLD" \
                --event_report_threshold "$EVENT_REPORT_THRESHOLD" \
                --checkpoint_min_report_precision "$CHECKPOINT_MIN_REPORT_PRECISION" \
                --checkpoint_min_report_recall "$CHECKPOINT_MIN_REPORT_RECALL"
            ;;
        B4)
            echo "B4 config: epochs=${B4_MAX_EPOCHS} min=${B4_MIN_EPOCHS} patience=${B4_PATIENCE} batch=${B4_BATCH_SIZE} lr=${B4_LEARNING_RATE} wd=${B4_WEIGHT_DECAY} dropout=${B4_DROPOUT}"
            python "${TOOLS_DIR}/train_b4_gcn_gru.py" \
                --dataset_dir "$DATASET_DIR" \
                --output_dir "${MODEL_ROOT}/b4_gcn_gru_seed${SEED}${OUTPUT_SUFFIX}" \
                --device "$DEVICE" \
                --seed "$SEED" \
                --training_profile "$TRAINING_PROFILE" \
                --batch_size "$B4_BATCH_SIZE" \
                --max_epochs "$B4_MAX_EPOCHS" \
                --patience "$B4_PATIENCE" \
                --min_epochs "$B4_MIN_EPOCHS" \
                --learning_rate "$B4_LEARNING_RATE" \
                --weight_decay "$B4_WEIGHT_DECAY" \
                --lr_min "$B4_LR_MIN" \
                --lr_schedule "$B4_LR_SCHEDULE" \
                --gcn_hidden "$B4_GCN_HIDDEN" \
                --gru_hidden "$B4_GRU_HIDDEN" \
                --dropout "$B4_DROPOUT" \
                --hot_eval_threshold "$HOT_EVAL_THRESHOLD" \
                --event_report_threshold "$EVENT_REPORT_THRESHOLD" \
                --checkpoint_min_report_precision "$CHECKPOINT_MIN_REPORT_PRECISION" \
                --checkpoint_min_report_recall "$CHECKPOINT_MIN_REPORT_RECALL"
            ;;
        B5)
            echo "B5 config: epochs=${B5_MAX_EPOCHS} min=${B5_MIN_EPOCHS} patience=${B5_PATIENCE} batch=${B5_BATCH_SIZE} lr=${B5_LEARNING_RATE} wd=${B5_WEIGHT_DECAY} dropout=${B5_DROPOUT}"
            python "${TOOLS_DIR}/train_b5_gat_gru.py" \
                --dataset_dir "$DATASET_DIR" \
                --output_dir "${MODEL_ROOT}/b5_gat_gru_seed${SEED}${OUTPUT_SUFFIX}" \
                --device "$DEVICE" \
                --seed "$SEED" \
                --training_profile "$TRAINING_PROFILE" \
                --batch_size "$B5_BATCH_SIZE" \
                --max_epochs "$B5_MAX_EPOCHS" \
                --patience "$B5_PATIENCE" \
                --min_epochs "$B5_MIN_EPOCHS" \
                --learning_rate "$B5_LEARNING_RATE" \
                --weight_decay "$B5_WEIGHT_DECAY" \
                --lr_min "$B5_LR_MIN" \
                --lr_schedule "$B5_LR_SCHEDULE" \
                --gat_hidden "$B5_GAT_HIDDEN" \
                --gat_heads "$B5_GAT_HEADS" \
                --gru_hidden "$B5_GRU_HIDDEN" \
                --dropout "$B5_DROPOUT" \
                --hot_eval_threshold "$HOT_EVAL_THRESHOLD" \
                --event_report_threshold "$EVENT_REPORT_THRESHOLD" \
                --checkpoint_min_report_precision "$CHECKPOINT_MIN_REPORT_PRECISION" \
                --checkpoint_min_report_recall "$CHECKPOINT_MIN_REPORT_RECALL"
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
