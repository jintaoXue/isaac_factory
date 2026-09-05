#!/bin/bash
# Validation-only, equal-seed B2 hyperparameter search. Test is never evaluated.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${ROOT}/source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/output/bottleneck_dataset"
TOOLS_DIR="${ROOT}/source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/tools"
BENCHMARK_TAG="${BENCHMARK_TAG:-factory_pdformer_134_v1}"
DATASET_DIR="${DATA_ROOT}/experiments/${BENCHMARK_TAG}"
TUNING_TAG="${TUNING_TAG:-b2_search_v2}"
TUNING_DIR="${DATASET_DIR}/models/tuning/${TUNING_TAG}"
TUNE_SEEDS="${TUNE_SEEDS:-42 43}"
N_JOBS="${N_JOBS:-8}"

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
    0.55 0.60 0.62 0.65 0.68 0.70 0.72 0.75 0.78
    0.80 0.82 0.85 0.88 0.90 0.92 0.95
)

run_candidate() {
    local name=$1
    local negative_ratio=$2
    local positive_weight=$3
    local estimators=$4
    local depth=$5
    local learning_rate=$6
    local min_child_weight=$7
    local reg_lambda=$8

    for seed in $TUNE_SEEDS; do
        local output_dir="${TUNING_DIR}/candidate_${name}/seed${seed}"
        echo "============================================================"
        echo "B2 candidate=${name} seed=${seed} validation-only"
        echo "negative_ratio=${negative_ratio} positive_weight=${positive_weight}"
        echo "trees=${estimators} depth=${depth} lr=${learning_rate}"
        echo "============================================================"
        python "${TOOLS_DIR}/train_b2_xgboost.py" \
            --dataset_dir "$DATASET_DIR" \
            --output_dir "$output_dir" \
            --training_profile "${TUNING_TAG}_${name}" \
            --validation_only \
            --seed "$seed" \
            --n_estimators "$estimators" \
            --max_depth "$depth" \
            --learning_rate "$learning_rate" \
            --min_child_weight "$min_child_weight" \
            --reg_lambda "$reg_lambda" \
            --negative_cell_ratio "$negative_ratio" \
            --hot_scale_pos_weight "$positive_weight" \
            --n_jobs "$N_JOBS" \
            --hot_eval_threshold 0.55 \
            --event_report_threshold 0.68 \
            --report_threshold_sweep "${REPORT_SWEEP[@]}" \
            --report_threshold_min_precision 0.80 \
            --checkpoint_min_report_recall 0.35
    done
}

# Search v1 showed that r4/w4 had the best mean F1 but high seed variance,
# while r4/w1 and every larger unweighted ratio collapsed recall. Refine the
# class emphasis between those endpoints and test two higher-coverage r8 runs.
run_candidate "c0_anchor_r4_w4" 4 4 500 5 0.03 3 5
run_candidate "c1_r4_w2" 4 2 500 5 0.03 3 5
run_candidate "c2_r4_w2_5" 4 2.5 500 5 0.03 3 5
run_candidate "c3_r4_w3" 4 3 500 5 0.03 3 5
run_candidate "c4_r4_w3_5" 4 3.5 500 5 0.03 3 5
run_candidate "c5_r4_w2_5_regularized" 4 2.5 700 4 0.03 5 10
run_candidate "c6_r8_w4" 8 4 500 5 0.03 3 5
run_candidate "c7_r8_w6" 8 6 500 5 0.03 3 5

python "${TOOLS_DIR}/select_baseline_tuning.py" \
    --tuning_dir "$TUNING_DIR" \
    --expected_seeds $TUNE_SEEDS
