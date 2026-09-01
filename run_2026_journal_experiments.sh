#!/bin/bash
set -euo pipefail

# 2026 Journal Paper reproducible experiment entry point.
# Usage:
#   ./run_2026_journal_experiments.sh train [cuda:0]
#   HC_LOAD_DIR=... ./run_2026_journal_experiments.sh eval [cuda:0]
#   ./run_2026_journal_experiments.sh random [cuda:0]
#   ./run_2026_journal_experiments.sh rule-k1 [cuda:0]
#   ./run_2026_journal_experiments.sh rule-k10 [cuda:0]
#   ./run_2026_journal_experiments.sh baselines [cuda:0]
#   HC_LOAD_DIR=... ./run_2026_journal_experiments.sh next [cuda:0]
#
# eval defaults to the three checkpoints identified from run_test_27:
# best-window neighborhood 2445000/2450000 and final 3315000.
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
export HC_TEST_TIMES="${HC_TEST_TIMES:-4}"
export HC_RANDOM_EPISODES="${HC_RANDOM_EPISODES:-4}"
export HC_N16_RANDOM_EPISODES="${HC_N16_RANDOM_EPISODES:-4}"
export HC_RULE_EPISODES="${HC_RULE_EPISODES:-4}"

run_train() {
    echo "[journal] train run_test_27 on ${DEVICE}"
    ./batch_train.sh 27 "${DEVICE}"
}

run_eval() {
    if [ -z "${HC_LOAD_DIR:-}" ]; then
        echo "错误: eval 需要 HC_LOAD_DIR=训练实验目录（目录内含 nn/）"
        exit 1
    fi
    for step in ${EVAL_STEPS}; do
        echo "[journal] eval checkpoint step=${step}, seeds=${HC_TEST_SEEDS}, repeats=${HC_TEST_TIMES}"
        HC_LOAD_STEP="${step}" ./batch_train.sh 29 "${DEVICE}"
    done
}

run_random() {
    for seed in ${BASELINE_SEEDS}; do
        echo "[journal] Random seed=${seed}, 4 episodes per N setting"
        HC_RUN_SEED="${seed}" ./batch_train.sh 26 32 "${DEVICE}"
    done
}

run_rule_k1() {
    for seed in ${BASELINE_SEEDS}; do
        echo "[journal] Rule K=1 seed=${seed}, 4 episodes"
        HC_RUN_SEED="${seed}" ./batch_train.sh 30 "${DEVICE}"
    done
}

run_rule_k10() {
    for seed in ${BASELINE_SEEDS}; do
        echo "[journal] Rule K=10 seed=${seed}, 4 episodes"
        HC_RUN_SEED="${seed}" ./batch_train.sh 31 "${DEVICE}"
    done
}

run_rule() {
    run_rule_k1
    run_rule_k10
}

run_baselines() {
    run_random
    run_rule
}

case "${MODE}" in
    train) run_train ;;
    eval) run_eval ;;
    random) run_random ;;
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
        echo "用法: $0 <train|eval|random|rule-k1|rule-k10|rule|baselines|next|all> [cuda:N]"
        exit 1
        ;;
esac
