#!/bin/bash
set -euo pipefail

# Hier4TPA journal entry — E0 eval + legacy T0/T1/T1R/T1RH.
# See docs/experiment_protocol.md (E0–E5); old board: docs/experiment_protocol_old.md.
#
# Usage:
#   ./run_2026_journal_experiments.sh E0 [cuda:0] [--dry-run]
#   ./run_2026_journal_experiments.sh E1 [cuda:0] [--dry-run]
#   ./run_2026_journal_experiments.sh T0|T1|T1R|T1RH [cuda:0]
#   ./run_2026_journal_experiments.sh TEACHER [cuda:0]
#   HC_LOAD_DIR=... ./run_2026_journal_experiments.sh eval-T1 [cuda:0]
#   ./run_2026_journal_experiments.sh baselines [cuda:0]
#   HC_LOAD_DIR=... HC_LOAD_STEP=... ./run_2026_journal_experiments.sh hier-eval [cuda:0]

MODE="${1:-}"
DEVICE="${2:-cuda:0}"

export HC_WANDB_TRAIN_PROJECT="${HC_WANDB_TRAIN_PROJECT:-HcFactory_TPA}"
export HC_WANDB_TEST_PROJECT="${HC_WANDB_TEST_PROJECT:-HcFactory_TPA_Eval}"
export HC_WANDB_BASELINE_PROJECT="${HC_WANDB_BASELINE_PROJECT:-${HC_WANDB_TEST_PROJECT}}"
export HC_TEST_SEEDS="${HC_TEST_SEEDS:-43,44,45,46,47,48,49,50,51,52}"
export HC_TEST_TIMES="${HC_TEST_TIMES:-1}"
export HC_CATALOG_TAG="${HC_CATALOG_TAG:-T1_random_ep20}"
# E1–E5 微调默认 30；T0/T1 hard 见 batch_train HC_MAX_HARD_EPISODES=100
export HC_MAX_TRAIN_EPISODES="${HC_MAX_TRAIN_EPISODES:-30}"
export HC_MAX_HARD_EPISODES="${HC_MAX_HARD_EPISODES:-100}"

EVAL_STEPS="${HC_EVAL_STEPS:-}"

usage() {
    cat <<EOF
用法: $0 <mode> [cuda:N]

训练（主推版本）:
  T0      hard train
  T1      explore → ORU + hard
  T1R     ORU + PER + Dueling（复用 catalog）
  T1RH    T1R + hierarchical credit + B-score
  E1      T0 权重热启动微调（step1290000，低 lr / ε≈0.05，S42，默认 ${HC_MAX_TRAIN_EPISODES:-30} ep）
  TEACHER 冻结 T0 教师采库（E2，ε=0，默认 50 ep，seed 42）

评测:
  E0 [cuda:N] [--dry-run]
          固定 step1290000，N10/K10/T40000，评测 seed 43–52 各 1 局，epsilon=0
          权重：logs/rl_games/HcFactory/hier_2026-08-27_23-17-41
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

run_e0_eval() {
    # Fixed E0 protocol; do not inherit legacy experiment settings.
    local repo_root load_dir head dry_run="${3:-}"
    repo_root=$(cd -- "$(dirname -- "$0")" && pwd)
    cd "${repo_root}"
    load_dir="${repo_root}/logs/rl_games/HcFactory/hier_2026-08-27_23-17-41"
    if [[ ! "${DEVICE}" =~ ^cuda:[0-9]+$ && "${DEVICE}" != cpu ]]; then
        echo "错误: 设备需为 cuda:N 或 cpu" >&2
        return 1
    fi
    if [[ -n "${dry_run}" && "${dry_run}" != --dry-run ]] || (( $# > 3 )); then
        echo "用法: $0 E0 [cuda:N] [--dry-run]" >&2
        return 1
    fi
    # Account configuration only; never print credentials.
    if [[ "${dry_run}" != --dry-run ]]; then
        local local_env="${HC_WANDB_LOCAL_ENV:-.wandb_local.env}"
        if [[ -f "${local_env}" ]]; then
            set -a
            source "${local_env}"
            set +a
        fi
        if [[ -n "${HC_WANDB_API_KEY:-}" ]]; then
            export WANDB_API_KEY="${HC_WANDB_API_KEY}"
        fi
    fi
    export WANDB_ENTITY="${HC_WANDB_ENTITY:-${WANDB_ENTITY:-rl-driving}}"
    export WANDB_MODE="${HC_WANDB_MODE:-${WANDB_MODE:-online}}"
    # train.py reads this variable independently of Hydra.
    export HC_WARMSTART=""
    local -a cmd=(
        python train.py --task HRTPaHC-v1 --algo hier
        --device "${DEVICE}" --num_envs 1 --headless --seed 42
        --test --test_times "${HC_TEST_TIMES}" --test_seeds "${HC_TEST_SEEDS}"
        --test_epsilon 0 --train_n_products 10 --max_parallel_cd_dispatch 10
        --load_dir "${load_dir}" --load_step 1290000
        --wandb_activate --wandb_project HcFactory_TPA_Eval
        --wandb_name Hier4TPA-E0-N10-S42-step1290000-eval
        --ftg_thresh_phy 0.95
        --seed 42
        agent.params.config.t_max_anchor=64000
        agent.params.config.max_episodic_steps=40000
        agent.params.config.parallel_producing_limit=10
        agent.params.config.c_forbid_none_mode=always
        agent.params.config.curriculum=false
        agent.params.config.explore=false
        agent.params.config.explore_catalog=false
        agent.params.config.catalog_collect=false
        agent.params.config.oru=false
        agent.params.config.prioritized_replay=false
        agent.params.config.dueling_dqn=false
        agent.params.config.noisy_net=false
        agent.params.config.hierarchical_credit=false
        agent.params.config.b_score_rl=false
        agent.params.config.env_rule_based_exploration=false
        'agent.params.config.warmstart=""'
        'agent.params.config.load_name=""'
    )
    # Horizon: 64000 * 10 / 16 = 40000. Keep anchor=64000.
    # Eval seeds 43–52 × HC_TEST_TIMES (default 1). Train convention remains S42.
    echo "[E0] N=10 K=10 dispatch=10 T=40000 epsilon=0; eval_seeds=43..52 x${HC_TEST_TIMES}; step=1290000"
    echo "[E0] load_dir=${load_dir}; project=HcFactory_TPA_Eval"
    if [[ "${dry_run}" == --dry-run ]]; then
        printf '%q ' "${cmd[@]}"
        printf '\n'
        return 0
    fi
    "${cmd[@]}"
}

run_e1_train() {
    # E1: T0 warmstart finetune (protocol T0+W). Same N/K/T as E0; train seed 42.
    local repo_root load_dir dry_run="${3:-}"
    repo_root=$(cd -- "$(dirname -- "$0")" && pwd)
    cd "${repo_root}"
    load_dir="${repo_root}/logs/rl_games/HcFactory/hier_2026-08-27_23-17-41"
    if [[ ! "${DEVICE}" =~ ^cuda:[0-9]+$ && "${DEVICE}" != cpu ]]; then
        echo "错误: 设备需为 cuda:N 或 cpu" >&2
        return 1
    fi
    if [[ -n "${dry_run}" && "${dry_run}" != --dry-run ]] || (( $# > 3 )); then
        echo "用法: $0 E1 [cuda:N] [--dry-run]" >&2
        return 1
    fi
    if [[ "${dry_run}" != --dry-run ]]; then
        local local_env="${HC_WANDB_LOCAL_ENV:-.wandb_local.env}"
        if [[ -f "${local_env}" ]]; then
            set -a
            source "${local_env}"
            set +a
        fi
        if [[ -n "${HC_WANDB_API_KEY:-}" ]]; then
            export WANDB_API_KEY="${HC_WANDB_API_KEY}"
        fi
    fi
    export WANDB_ENTITY="${HC_WANDB_ENTITY:-${WANDB_ENTITY:-rl-driving}}"
    export WANDB_MODE="${HC_WANDB_MODE:-${WANDB_MODE:-online}}"
    if [[ ! -d "${load_dir}" ]]; then
        echo "错误: 缺少教师权重目录: ${load_dir}" >&2
        return 1
    fi
    local -a cmd=(
        python train.py --task HRTPaHC-v1 --algo hier
        --device "${DEVICE}" --num_envs 1 --headless --seed 42
        --train_n_products 10 --max_parallel_cd_dispatch 10
        --max_sim_episodes "${HC_MAX_TRAIN_EPISODES}"
        --load_dir "${load_dir}" --load_step 1290000
        --wandb_activate --wandb_project HcFactory_TPA
        --wandb_name Hier4TPA-E1-N10-S42
        --algo_variant E1
        --ftg_thresh_phy 0.95
        agent.params.config.t_max_anchor=64000
        agent.params.config.max_episodic_steps=40000
        agent.params.config.parallel_producing_limit=10
        agent.params.config.c_forbid_none_mode=always
        agent.params.config.curriculum=false
        agent.params.config.explore=false
        agent.params.config.explore_catalog=false
        agent.params.config.catalog_collect=false
        agent.params.config.oru=false
        agent.params.config.prioritized_replay=false
        agent.params.config.dueling_dqn=false
        agent.params.config.noisy_net=false
        agent.params.config.hierarchical_credit=false
        agent.params.config.b_score_rl=false
        agent.params.config.env_rule_based_exploration=false
        agent.params.config.learning_rate=2.0e-5
        agent.params.config.encoder_learning_rate=1.0e-5
        agent.params.config.late_learning_rate=2.0e-5
        agent.params.config.late_encoder_learning_rate=1.0e-5
        agent.params.config.epsilon_start=0.05
        agent.params.config.epsilon_end=0.05
        agent.params.config.epsilon_decay_steps=1
        'agent.params.config.warmstart=""'
        'agent.params.config.load_name=""'
    )
    echo "[E1] warmstart step=1290000; N=10 K=10 T=40000; lr_q=2e-5 lr_enc=1e-5 eps=0.05; seed=42"
    echo "[E1] max_sim_episodes=${HC_MAX_TRAIN_EPISODES}; load_dir=${load_dir}; project=HcFactory_TPA; wandb=Hier4TPA-E1-N10-S42"
    if [[ "${dry_run}" == --dry-run ]]; then
        printf '%q ' "${cmd[@]}"
        printf '\n'
        return 0
    fi
    "${cmd[@]}"
}

case "${MODE}" in
    E0) run_e0_eval "$@" ;;
    E1) run_e1_train "$@" ;;
    ""|-h|--help|help) usage; exit 0 ;;
    T0) ./batch_train.sh T0 "${DEVICE}" ;;
    T1|train) ./batch_train.sh T1 "${DEVICE}" ;;
    T1R) ./batch_train.sh T1R "${DEVICE}" ;;
    T1RH) ./batch_train.sh T1RH "${DEVICE}" ;;
    TEACHER|teacher|E2-collect) ./batch_train.sh TEACHER "${DEVICE}" ;;
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
