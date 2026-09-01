#!/bin/bash
set -euo pipefail

# 2026 Journal Paper reproducible experiment entry point.
# Usage:
#   ./run_2026_journal_experiments.sh train [cuda:0]
#   HC_LOAD_DIR=... ./run_2026_journal_experiments.sh eval [cuda:0]
#   ./run_2026_journal_experiments.sh rule-n16 [cuda:0]   # server: K1+K10, N=16
#   ./run_2026_journal_experiments.sh rule-n10 [cuda:0]   # server: K1+K10, N=10
#   HC_LOAD_DIR=... ./run_2026_journal_experiments.sh hier-eval-n16 [cuda:0]
#   HC_LOAD_DIR=... ./run_2026_journal_experiments.sh hier-eval-n10 [cuda:0]
#
# Default protocol: 5 seeds (42–46) × 2 episodes = 10 episodes per run.
# Live eval writes under <test_exp>/eval/: episodes.jsonl, eval_summary_partial.json;
# metrics.jsonl + wandb MetricTest/* update each episode (~500-step heartbeat).

MODE="${1:-next}"
DEVICE="${2:-cuda:0}"
EVAL_STEPS="${HC_EVAL_STEPS:-2445000 2450000 3315000}"
BASELINE_SEEDS="${HC_BASELINE_SEEDS:-42 43 44 45 46}"
export HC_WANDB_TRAIN_PROJECT="${HC_WANDB_TRAIN_PROJECT:-HcFactory_TPA}"
export HC_WANDB_TEST_PROJECT="${HC_WANDB_TEST_PROJECT:-HcFactory_TPA_Eval}"
export HC_WANDB_BASELINE_PROJECT="${HC_WANDB_BASELINE_PROJECT:-${HC_WANDB_TEST_PROJECT}}"
export HC_TEST_SEEDS="${HC_TEST_SEEDS:-42,43,44,45,46}"
export HC_TEST_TIMES="${HC_TEST_TIMES:-2}"
export HC_RANDOM_EPISODES="${HC_RANDOM_EPISODES:-2}"
export HC_N16_RANDOM_EPISODES="${HC_N16_RANDOM_EPISODES:-2}"
export HC_RULE_EPISODES="${HC_RULE_EPISODES:-2}"

_episodes_per_seed() {
    echo "${HC_RULE_EPISODES}"
}

run_train() {
    echo "[journal] train run_test_27 on ${DEVICE}"
    ./batch_train.sh 27 "${DEVICE}"
}

run_hier_eval_for_n() {
    local n_products="$1"
    if [ -z "${HC_LOAD_DIR:-}" ]; then
        echo "错误: hier eval 需要 HC_LOAD_DIR=训练实验目录（目录内含 nn/）"
        exit 1
    fi
    export HC_TRAIN_N_PRODUCTS="${n_products}"
    for step in ${EVAL_STEPS}; do
        echo "[journal] hier eval N=${n_products} step=${step}, seeds=${HC_TEST_SEEDS}, repeats=${HC_TEST_TIMES}"
        HC_LOAD_STEP="${step}" ./batch_train.sh 29 "${DEVICE}"
    done
}

run_eval() {
    run_hier_eval_for_n 16
}

run_hier_eval_n16() {
    run_hier_eval_for_n 16
}

run_hier_eval_n10() {
    run_hier_eval_for_n 10
}

run_random_n10() {
    for seed in ${BASELINE_SEEDS}; do
        echo "[journal] Random N=10 seed=${seed}, $(_episodes_per_seed) episodes"
        HC_RUN_SEED="${seed}" ./batch_train.sh 26 "${DEVICE}"
    done
}

run_random_n16() {
    for seed in ${BASELINE_SEEDS}; do
        echo "[journal] Random N=16 seed=${seed}, ${HC_N16_RANDOM_EPISODES} episodes"
        HC_RUN_SEED="${seed}" ./batch_train.sh 32 "${DEVICE}"
    done
}

run_rule_k1_for_jobs() {
    local -a jobs=("$@")
    for seed in ${BASELINE_SEEDS}; do
        for job in "${jobs[@]}"; do
            echo "[journal] Rule job=${job} seed=${seed}, $(_episodes_per_seed) episodes"
            HC_RUN_SEED="${seed}" ./batch_train.sh "${job}" "${DEVICE}"
        done
    done
}

run_rule_k10_for_jobs() {
    local -a jobs=("$@")
    for seed in ${BASELINE_SEEDS}; do
        for job in "${jobs[@]}"; do
            echo "[journal] Rule job=${job} seed=${seed}, $(_episodes_per_seed) episodes"
            HC_RUN_SEED="${seed}" ./batch_train.sh "${job}" "${DEVICE}"
        done
    done
}

run_rule_n16() {
    echo "[journal] Rule N=16: K=1 (job 30) then K=10 (job 31)"
    run_rule_k1_for_jobs 30
    run_rule_k10_for_jobs 31
}

run_rule_n10() {
    echo "[journal] Rule N=10: K=1 (job 24) then K=10 (job 25)"
    run_rule_k1_for_jobs 24
    run_rule_k10_for_jobs 25
}

run_rule_k1() {
    run_rule_k1_for_jobs 30
}

run_rule_k10() {
    run_rule_k10_for_jobs 31
}

run_rule() {
    run_rule_n16
    run_rule_n10
}

run_baselines() {
    run_random_n10
    run_random_n16
    run_rule
}

case "${MODE}" in
    train) run_train ;;
    eval) run_eval ;;
    hier-eval-n16) run_hier_eval_n16 ;;
    hier-eval-n10) run_hier_eval_n10 ;;
    random) run_random_n10; run_random_n16 ;;
    random-n10) run_random_n10 ;;
    random-n16) run_random_n16 ;;
    rule-n16) run_rule_n16 ;;
    rule-n10) run_rule_n10 ;;
    rule-k1) run_rule_k1 ;;
    rule-k10) run_rule_k10 ;;
    rule) run_rule ;;
    baselines) run_baselines ;;
    next)
        run_eval
        run_baselines
        ;;
    all)
        echo "all 分两阶段运行：先 train；训练结束后将 HC_LOAD_DIR 指向新目录，再运行 eval/baselines。"
        run_train
        ;;
    *)
        echo "用法: $0 <train|eval|hier-eval-n16|hier-eval-n10|rule-n16|rule-n10|random|rule|baselines|next|all> [cuda:N]"
        exit 1
        ;;
esac
