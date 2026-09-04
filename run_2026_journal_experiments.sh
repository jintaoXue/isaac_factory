#!/bin/bash
set -euo pipefail

# Hier4TPA journal entry — versions T0 / T1 / T1R / T1RH (+ baselines / eval).
# See docs/experiment_protocol.md §1c.
#
# Usage:
#   ./run_2026_journal_experiments.sh T0|T1|T1R|T1RH [cuda:0]
#   HC_LOAD_DIR=... ./run_2026_journal_experiments.sh eval-T1 [cuda:0]
#   ./run_2026_journal_experiments.sh baselines [cuda:0]
#   HC_LOAD_DIR=... HC_LOAD_STEP=... ./run_2026_journal_experiments.sh hier-eval [cuda:0]

MODE="${1:-}"
DEVICE="${2:-cuda:0}"

export HC_WANDB_TRAIN_PROJECT="${HC_WANDB_TRAIN_PROJECT:-HcFactory_TPA}"
export HC_WANDB_TEST_PROJECT="${HC_WANDB_TEST_PROJECT:-HcFactory_TPA_Eval}"
export HC_WANDB_BASELINE_PROJECT="${HC_WANDB_BASELINE_PROJECT:-${HC_WANDB_TEST_PROJECT}}"
export HC_TEST_SEEDS="${HC_TEST_SEEDS:-42,43,44,45,46}"
export HC_TEST_TIMES="${HC_TEST_TIMES:-2}"
export HC_CATALOG_TAG="${HC_CATALOG_TAG:-T1_random_ep20}"

EVAL_STEPS="${HC_EVAL_STEPS:-}"

usage() {
    cat <<EOF
用法: $0 <mode> [cuda:N]

训练（主推版本）:
  T0      hard train
  T1      explore → ORU + hard
  T1R     ORU + PER + Dueling（复用 catalog）
  T1RH    T1R + hierarchical credit + B-score

评测:
  eval-T0 | eval-T1 | eval-T1R | eval-T1RH
          需 HC_LOAD_DIR；可选 HC_LOAD_STEP / HC_EVAL_STEPS
  hier-eval / hier-eval-n16 / hier-eval-n10
          通用评测（HC_EVAL_VARIANT 默认 eval）

基线:
  baselines | rule-n10 | rule-n16 | random-n10 | random-n16 | random | rule

其它:
  train   同 T1（兼容旧入口）
EOF
}

run_hier_eval_for_n() {
    local n_products="$1"
    if [ -z "${HC_LOAD_DIR:-}" ]; then
        echo "错误: hier eval 需要 HC_LOAD_DIR=训练实验目录（目录内含 nn/）"
        exit 1
    fi
    export HC_TRAIN_N_PRODUCTS="${n_products}"
    if [ -z "${EVAL_STEPS}" ]; then
        echo "[journal] hier eval N=${n_products} step=latest, variant=${HC_EVAL_VARIANT:-eval}"
        ./batch_train.sh 29 "${DEVICE}"
        return
    fi
    for step in ${EVAL_STEPS}; do
        echo "[journal] hier eval N=${n_products} step=${step}, variant=${HC_EVAL_VARIANT:-eval}"
        HC_LOAD_STEP="${step}" ./batch_train.sh 29 "${DEVICE}"
    done
}

run_hier_eval() {
    echo "[journal] hier eval N16→N10 variant=${HC_EVAL_VARIANT:-eval} load=${HC_LOAD_DIR:-}"
    run_hier_eval_for_n 16
    run_hier_eval_for_n 10
}

run_eval_variant() {
    local variant="$1"
    export HC_EVAL_VARIANT="${variant}"
    if [ -z "${HC_LOAD_DIR:-}" ]; then
        echo "错误: eval-${variant} 需要 HC_LOAD_DIR"
        exit 1
    fi
    run_hier_eval
}

case "${MODE}" in
    ""|-h|--help|help) usage; exit 0 ;;
    T0) ./batch_train.sh T0 "${DEVICE}" ;;
    T1|train) ./batch_train.sh T1 "${DEVICE}" ;;
    T1R) ./batch_train.sh T1R "${DEVICE}" ;;
    T1RH) ./batch_train.sh T1RH "${DEVICE}" ;;
    eval-T0) run_eval_variant T0 ;;
    eval-T1) run_eval_variant T1 ;;
    eval-T1R) run_eval_variant T1R ;;
    eval-T1RH) run_eval_variant T1RH ;;
    hier-eval) run_hier_eval ;;
    hier-eval-n16) run_hier_eval_for_n 16 ;;
    hier-eval-n10) run_hier_eval_for_n 10 ;;
    # Legacy aliases (old curr/hard eval entry points)
    hier-eval-hard)
        export HC_LOAD_DIR="${HC_LOAD_DIR:-${HC_HARD_LOAD_DIR:-}}"
        export HC_EVAL_VARIANT=T0
        EVAL_STEPS="${HC_EVAL_STEPS:-${HC_HARD_EVAL_STEP:-}}"
        run_hier_eval
        ;;
    hier-eval-curr)
        export HC_LOAD_DIR="${HC_LOAD_DIR:-${HC_CURR_LOAD_DIR:-}}"
        export HC_EVAL_VARIANT=curr
        EVAL_STEPS="${HC_EVAL_STEPS:-${HC_CURR_EVAL_STEP:-}}"
        run_hier_eval
        ;;
    random-n10) ./batch_train.sh 26 "${DEVICE}" ;;
    random-n16) ./batch_train.sh 32 "${DEVICE}" ;;
    random) ./batch_train.sh 26 32 "${DEVICE}" ;;
    rule-n10) ./batch_train.sh 24 25 "${DEVICE}" ;;
    rule-n16) ./batch_train.sh 30 31 "${DEVICE}" ;;
    rule) ./batch_train.sh 24 25 30 31 "${DEVICE}" ;;
    baselines) ./batch_train.sh E "${DEVICE}" ;;
    *)
        usage
        exit 1
        ;;
esac
