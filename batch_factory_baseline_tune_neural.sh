#!/bin/bash
# Validation-only, equal-budget search for one neural baseline at a time.

set -euo pipefail

if [ $# -ne 1 ]; then
    echo "Usage: $0 <B3|B4|B5>" >&2
    exit 1
fi

MODEL=$1
case "$MODEL" in
    B3)
        TOOL_NAME="train_b3_lstm.py"
        DEFAULT_TAG="b3_search_v1"
        ;;
    B4)
        TOOL_NAME="train_b4_gcn_gru.py"
        DEFAULT_TAG="b4_search_v1"
        ;;
    B5)
        TOOL_NAME="train_b5_gat_gru.py"
        DEFAULT_TAG="b5_search_v1"
        ;;
    *)
        echo "Expected B3, B4, or B5; received: ${MODEL}" >&2
        exit 1
        ;;
esac

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${ROOT}/source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/output/bottleneck_dataset"
TOOLS_DIR="${ROOT}/source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/tools"
BENCHMARK_TAG="${BENCHMARK_TAG:-factory_pdformer_134_v1}"
DATASET_DIR="${DATA_ROOT}/experiments/${BENCHMARK_TAG}"
TUNING_TAG="${TUNING_TAG:-$DEFAULT_TAG}"
TUNING_DIR="${DATASET_DIR}/models/tuning/${TUNING_TAG}"
TUNE_SEEDS="${TUNE_SEEDS:-42 43}"
DEVICE="${DEVICE:-cuda:0}"

for required in dataset.pt dataset_manifest.json split_manifest.json; do
    if [ ! -f "${DATASET_DIR}/${required}" ]; then
        echo "Missing ${DATASET_DIR}/${required}" >&2
        exit 1
    fi
done

if [ -e "$TUNING_DIR" ]; then
    echo "Tuning directory already exists: ${TUNING_DIR}" >&2
    echo "Set a new TUNING_TAG to start another immutable search." >&2
    exit 1
fi
mkdir -p "$TUNING_DIR"

REPORT_SWEEP=(
    0.55 0.60 0.62 0.65 0.68 0.70 0.72 0.75 0.78 0.80 0.82 0.85
)

run_candidate() {
    local name=$1
    shift
    for seed in $TUNE_SEEDS; do
        local output_dir="${TUNING_DIR}/candidate_${name}/seed${seed}"
        echo "============================================================"
        echo "${MODEL} candidate=${name} seed=${seed} validation-only"
        echo "============================================================"
        python "${TOOLS_DIR}/${TOOL_NAME}" \
            --dataset_dir "$DATASET_DIR" \
            --output_dir "$output_dir" \
            --training_profile "${TUNING_TAG}_${name}" \
            --validation_only \
            --device "$DEVICE" \
            --seed "$seed" \
            --report_threshold_sweep "${REPORT_SWEEP[@]}" \
            --checkpoint_min_report_precision 0.80 \
            --checkpoint_min_report_recall 0.35 \
            "$@"
    done
}

base_loss=(
    --lambda_event_will 2.5
    --event_will_pos_weight 3.0
    --event_will_fp_weight 2.0
    --event_will_upcoming_pos_weight 4.0
    --event_will_ongoing_pos_weight 3.0
)
balanced_loss=(
    --lambda_event_will 3.0
    --event_will_pos_weight 3.0
    --event_will_fp_weight 3.0
    --event_will_upcoming_pos_weight 6.0
    --event_will_ongoing_pos_weight 2.5
)
strict_loss=(
    --lambda_event_will 3.0
    --event_will_pos_weight 3.0
    --event_will_fp_weight 4.0
    --event_will_upcoming_pos_weight 7.0
    --event_will_ongoing_pos_weight 2.5
)

case "$MODEL" in
    B3)
        common=(--batch_size 32 --max_epochs 60 --min_epochs 12 --patience 12)
        run_candidate "c0_incumbent" "${common[@]}" \
            --learning_rate 0.0003 --weight_decay 0.001 \
            --lstm_hidden 128 --node_hidden 128 --node_embedding 32 --dropout 0.25 \
            "${base_loss[@]}"
        run_candidate "c1_balanced" "${common[@]}" \
            --learning_rate 0.0003 --weight_decay 0.001 \
            --lstm_hidden 128 --node_hidden 128 --node_embedding 32 --dropout 0.25 \
            "${balanced_loss[@]}"
        run_candidate "c2_strict" "${common[@]}" \
            --learning_rate 0.0003 --weight_decay 0.001 \
            --lstm_hidden 128 --node_hidden 128 --node_embedding 32 --dropout 0.25 \
            "${strict_loss[@]}"
        run_candidate "c3_compact96" "${common[@]}" \
            --learning_rate 0.0003 --weight_decay 0.005 \
            --lstm_hidden 96 --node_hidden 96 --node_embedding 16 --dropout 0.25 \
            "${balanced_loss[@]}"
        run_candidate "c4_compact64" "${common[@]}" \
            --learning_rate 0.0003 --weight_decay 0.01 \
            --lstm_hidden 64 --node_hidden 96 --node_embedding 16 --dropout 0.35 \
            "${balanced_loss[@]}"
        run_candidate "c5_two_layer" "${common[@]}" \
            --learning_rate 0.0002 --weight_decay 0.005 \
            --lstm_hidden 96 --lstm_layers 2 --node_hidden 96 \
            --node_embedding 16 --dropout 0.25 "${balanced_loss[@]}"
        run_candidate "c6_regularized" "${common[@]}" \
            --learning_rate 0.0002 --weight_decay 0.005 \
            --lstm_hidden 128 --node_hidden 128 --node_embedding 32 --dropout 0.35 \
            "${balanced_loss[@]}"
        run_candidate "c7_low_lr_strict" "${common[@]}" \
            --learning_rate 0.00015 --weight_decay 0.01 \
            --lstm_hidden 128 --node_hidden 128 --node_embedding 32 --dropout 0.25 \
            "${strict_loss[@]}"
        ;;
    B4)
        common=(--batch_size 24 --max_epochs 60 --min_epochs 10 --patience 10)
        run_candidate "c0_incumbent" "${common[@]}" \
            --learning_rate 0.0003 --weight_decay 0.01 \
            --gcn_hidden 64 --gru_hidden 128 --dropout 0.20 "${base_loss[@]}"
        run_candidate "c1_balanced" "${common[@]}" \
            --learning_rate 0.0003 --weight_decay 0.01 \
            --gcn_hidden 64 --gru_hidden 128 --dropout 0.20 "${balanced_loss[@]}"
        run_candidate "c2_strict" "${common[@]}" \
            --learning_rate 0.0003 --weight_decay 0.01 \
            --gcn_hidden 64 --gru_hidden 128 --dropout 0.20 "${strict_loss[@]}"
        run_candidate "c3_low_lr" "${common[@]}" \
            --learning_rate 0.0002 --weight_decay 0.01 \
            --gcn_hidden 64 --gru_hidden 128 --dropout 0.20 "${balanced_loss[@]}"
        run_candidate "c4_low_dropout" "${common[@]}" \
            --learning_rate 0.0003 --weight_decay 0.005 \
            --gcn_hidden 64 --gru_hidden 128 --dropout 0.10 "${balanced_loss[@]}"
        run_candidate "c5_high_dropout" "${common[@]}" \
            --learning_rate 0.0003 --weight_decay 0.02 \
            --gcn_hidden 64 --gru_hidden 128 --dropout 0.30 "${balanced_loss[@]}"
        run_candidate "c6_wide" "${common[@]}" \
            --learning_rate 0.0002 --weight_decay 0.01 \
            --gcn_hidden 96 --gru_hidden 160 --dropout 0.20 "${balanced_loss[@]}"
        run_candidate "c7_spatial96" "${common[@]}" \
            --learning_rate 0.0003 --weight_decay 0.01 \
            --gcn_hidden 96 --gru_hidden 128 --dropout 0.20 "${balanced_loss[@]}"
        ;;
    B5)
        common=(--batch_size 16 --max_epochs 60 --min_epochs 12 --patience 12)
        run_candidate "c0_stabilized" "${common[@]}" \
            --learning_rate 0.00015 --weight_decay 0.01 \
            --gat_hidden 64 --gat_heads 4 --gru_hidden 128 --dropout 0.20 \
            "${base_loss[@]}"
        run_candidate "c1_balanced" "${common[@]}" \
            --learning_rate 0.00015 --weight_decay 0.01 \
            --gat_hidden 64 --gat_heads 4 --gru_hidden 128 --dropout 0.20 \
            "${balanced_loss[@]}"
        run_candidate "c2_strict" "${common[@]}" \
            --learning_rate 0.00015 --weight_decay 0.01 \
            --gat_hidden 64 --gat_heads 4 --gru_hidden 128 --dropout 0.20 \
            "${strict_loss[@]}"
        run_candidate "c3_lr2" "${common[@]}" \
            --learning_rate 0.0002 --weight_decay 0.01 \
            --gat_hidden 64 --gat_heads 4 --gru_hidden 128 --dropout 0.20 \
            "${balanced_loss[@]}"
        run_candidate "c4_lr3" "${common[@]}" \
            --learning_rate 0.0003 --weight_decay 0.01 \
            --gat_hidden 64 --gat_heads 4 --gru_hidden 128 --dropout 0.20 \
            "${balanced_loss[@]}"
        run_candidate "c5_low_dropout" "${common[@]}" \
            --learning_rate 0.0002 --weight_decay 0.005 \
            --gat_hidden 64 --gat_heads 4 --gru_hidden 128 --dropout 0.10 \
            "${balanced_loss[@]}"
        run_candidate "c6_wide" "${common[@]}" \
            --learning_rate 0.0002 --weight_decay 0.01 \
            --gat_hidden 96 --gat_heads 4 --gru_hidden 160 --dropout 0.20 \
            "${balanced_loss[@]}"
        run_candidate "c7_two_heads" "${common[@]}" \
            --learning_rate 0.0002 --weight_decay 0.01 \
            --gat_hidden 64 --gat_heads 2 --gru_hidden 128 --dropout 0.20 \
            "${balanced_loss[@]}"
        ;;
esac

python "${TOOLS_DIR}/select_baseline_tuning.py" \
    --tuning_dir "$TUNING_DIR" \
    --expected_seeds $TUNE_SEEDS
