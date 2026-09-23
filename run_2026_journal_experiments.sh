#!/bin/bash
set -euo pipefail

# Hier4TPA journal entry — E0–E6 + ablations; see docs/experiment_protocol.md.
# Usage:
#   ./run_2026_journal_experiments.sh E0 [cuda:0] [--dry-run]
#   ./run_2026_journal_experiments.sh E1|…|E5|E5-no-oru|E6|E6-no-oru [cuda:0] [--dry-run]
#   ./run_2026_journal_experiments.sh TEACHER [cuda:0]
#   ./run_2026_journal_experiments.sh T0|T1|T1R|T1RH [cuda:0]
#   HC_LOAD_DIR=... ./run_2026_journal_experiments.sh eval-T1 [cuda:0]
#   ./run_2026_journal_experiments.sh baselines [cuda:0]
#   HC_LOAD_DIR=... HC_LOAD_STEP=... ./run_2026_journal_experiments.sh hier-eval [cuda:0]
#   ./run_2026_journal_experiments.sh eval-5090 [cuda:0] [--dry-run]
#   ./run_2026_journal_experiments.sh eval-E5|eval-desk|eval-desk-rev [cuda:0] [--dry-run]
# Stub: E6-no-guide|E6-no-hier|E6-no-ar|E2-random-data|…

MODE="${1:-}"
DEVICE="${2:-cuda:0}"

export HC_WANDB_TRAIN_PROJECT="${HC_WANDB_TRAIN_PROJECT:-HcFactory_TPA}"
export HC_WANDB_TEST_PROJECT="${HC_WANDB_TEST_PROJECT:-HcFactory_TPA_Eval}"
export HC_WANDB_BASELINE_PROJECT="${HC_WANDB_BASELINE_PROJECT:-${HC_WANDB_TEST_PROJECT}}"
export HC_TEST_SEEDS="${HC_TEST_SEEDS:-43,44,45,46,47,48,49,50,51,52}"
export HC_TEST_TIMES="${HC_TEST_TIMES:-1}"
export HC_CATALOG_TAG="${HC_CATALOG_TAG:-T1_random_ep20}"
# E1–E6 微调默认 60；T0/T1 hard 见 batch_train HC_MAX_HARD_EPISODES=100
export HC_MAX_TRAIN_EPISODES="${HC_MAX_TRAIN_EPISODES:-60}"
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
  E1      T0 权重热启动微调（step1290000，低 lr / ε≈0.05，S42，默认 ${HC_MAX_TRAIN_EPISODES:-60} ep）
  E1.5    E1＋仅信用缩放（A×2.0 B×1.5，无 b_score；历史污染跑已改名为此）
  E2      E1＋教师 offline_replay ORU（25% 教师混合；需先 TEACHER 采库）
  E2.5    E2＋仅信用缩放（无 b_score；历史污染跑已改名为此）
  E3      E2＋教师引导在线探索（ε 分支教师/随机混合，教师比例衰减）
  E3-no-oru  E1＋教师探索、不开 ORU（拆开 data vs guide）
  E3.5    E3＋仅信用缩放（无 b_score；历史污染跑已改名为此）
  E4      E3＋层级学习（B-score RL＋A/B 信用缩放）
  E4-no-oru  E4 去掉 ORU（保留教师探索＋层级学习）
  E5      E3＋自回归（分层 ε＋步内候选采样；无 H/b_score）
  E5-no-oru  E5 去掉 ORU（保留教师探索＋AR）
  E6      E5＋E4 完整方法（AR＋层级学习）
  E6-no-oru  E6 去掉 ORU（保留教师探索＋AR＋层级学习）
  TEACHER 冻结 T0 教师采库（ε=0，默认 50 ep，seed 42）

规划 stub（会提示未实现）:
  E6-no-guide | E6-no-hier | E6-no-ar
  E6-plus-replay | E6-plus-curriculum | E6-plus-staged | E2-random-data

评测:
  E0 [cuda:N] [--dry-run]
          固定 step1290000，N10/K10/T40000，评测 seed 43–52 各 1 局，epsilon=0
          权重：logs/rl_games/HcFactory/hier_2026-08-27_23-17-41
  eval-T0 | eval-T1 | eval-T1R | eval-T1RH
          需 HC_LOAD_DIR；可选 HC_LOAD_STEP / HC_EVAL_STEPS
  hier-eval / hier-eval-n16 / hier-eval-n10
          通用评测（HC_EVAL_VARIANT 默认 eval）
  eval-5090 [cuda:N] [--dry-run]
          一条龙评测 5090 本地权重：E1→E2→E2.5→E6（训练最优 step；默认跳过已完成的 E0）
          见 docs/eval_checkpoint_selection.md；HC_EVAL_INCLUDE_E0=1 可加跑 E0
          默认 HC_EVAL_SEED_CHUNK=5（每 5 个 seed 重启进程，避免第 10 局前崩溃）
  eval-E5 [cuda:N] [--dry-run]
          评测从工位同步过来的 E5 训练最优 ckpt（step 750000）
  eval-desk [cuda:N] [--dry-run]
          评测工位 E*：E4 / E5-no-oru(peak+late) / E5 / E3.5 / E3-no-oru / E1.5 / E3
  eval-desk-rev [cuda:N] [--dry-run]
          同上倒序（本机与 5090 对开）
  eval-E5-no-oru-late [cuda:N] [--dry-run]
          仅评 E5-no-oru 后半程最优 step 870000

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
    if [ -z "${EVAL_STEPS:-}" ]; then
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

run_e1_5_train() {
    # E1.5: E1 + credit scales only (no b_score_rl). Matches historical YAML-leak E1.
    local repo_root load_dir dry_run="${3:-}"
    repo_root=$(cd -- "$(dirname -- "$0")" && pwd)
    cd "${repo_root}"
    load_dir="${repo_root}/logs/rl_games/HcFactory/hier_2026-08-27_23-17-41"
    if [[ ! "${DEVICE}" =~ ^cuda:[0-9]+$ && "${DEVICE}" != cpu ]]; then
        echo "错误: 设备需为 cuda:N 或 cpu" >&2
        return 1
    fi
    if [[ -n "${dry_run}" && "${dry_run}" != --dry-run ]] || (( $# > 3 )); then
        echo "用法: $0 E1.5 [cuda:N] [--dry-run]" >&2
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
    # Prefer hydra overrides so credit_scale_* and b_score_rl stay explicit.
    # --hierarchical_credit alone no longer forces b_score_rl (orthogonal flags).
    local -a cmd=(
        python train.py --task HRTPaHC-v1 --algo hier
        --device "${DEVICE}" --num_envs 1 --headless --seed 42
        --train_n_products 10 --max_parallel_cd_dispatch 10
        --max_sim_episodes "${HC_MAX_TRAIN_EPISODES}"
        --load_dir "${load_dir}" --load_step 1290000
        --wandb_activate --wandb_project HcFactory_TPA
        --wandb_name Hier4TPA-E1.5-N10-S42
        --algo_variant E1.5
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
        agent.params.config.hierarchical_credit=true
        agent.params.config.b_score_rl=false
        agent.params.config.credit_scale_A=2.0
        agent.params.config.credit_scale_B=1.5
        agent.params.config.credit_scale_CD=1.0
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
    echo "[E1.5] E1 + credit A×2.0 B×1.5 (no b_score); seed=42"
    echo "[E1.5] max_sim_episodes=${HC_MAX_TRAIN_EPISODES}; load_dir=${load_dir}; wandb=Hier4TPA-E1.5-N10-S42"
    if [[ "${dry_run}" == --dry-run ]]; then
        printf '%q ' "${cmd[@]}"
        printf '\n'
        return 0
    fi
    "${cmd[@]}"
}

run_e2_train() {
    # E2: E1 warmstart + teacher offline_replay ORU (protocol T2+W).
    local repo_root load_dir catalog_root dry_run="${3:-}"
    local teacher_eps="${HC_TEACHER_EPISODES:-50}"
    repo_root=$(cd -- "$(dirname -- "$0")" && pwd)
    cd "${repo_root}"
    load_dir="${repo_root}/logs/rl_games/HcFactory/hier_2026-08-27_23-17-41"
    # Prefer explicit override; else canonical E2 teacher path; else tolerate
    # mis-tagged teacher dumps under T1_random_ep20 (journal default leak).
    if [[ -n "${HC_EXPLORE_CATALOG_DIR:-}" ]]; then
        catalog_root="${HC_EXPLORE_CATALOG_DIR}"
    else
        catalog_root="${repo_root}/env_checkpoints/policy_explore/N10_T40000__E2_teacher_ep${teacher_eps}"
        if [[ ! -d "${catalog_root}/offline_replay" ]]; then
            local alt="${repo_root}/env_checkpoints/random_explore/N10_T40000__T1_random_ep20"
            if [[ -d "${alt}/offline_replay" ]]; then
                echo "[E2] WARN: canonical teacher catalog missing; using ${alt}" >&2
                catalog_root="${alt}"
            fi
        fi
    fi
    if [[ ! "${DEVICE}" =~ ^cuda:[0-9]+$ && "${DEVICE}" != cpu ]]; then
        echo "错误: 设备需为 cuda:N 或 cpu" >&2
        return 1
    fi
    if [[ -n "${dry_run}" && "${dry_run}" != --dry-run ]] || (( $# > 3 )); then
        echo "用法: $0 E2 [cuda:N] [--dry-run]" >&2
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
    if [[ ! -d "${catalog_root}/offline_replay" ]]; then
        echo "错误: 缺少教师 offline_replay: ${catalog_root}/offline_replay" >&2
        echo "请先跑: ./run_2026_journal_experiments.sh TEACHER ${DEVICE}" >&2
        echo "或设置 HC_EXPLORE_CATALOG_DIR=... 指向采库根目录" >&2
        return 1
    fi
    local -a cmd=(
        python train.py --task HRTPaHC-v1 --algo hier
        --device "${DEVICE}" --num_envs 1 --headless --seed 42
        --train_n_products 10 --max_parallel_cd_dispatch 10
        --max_sim_episodes "${HC_MAX_TRAIN_EPISODES}"
        --load_dir "${load_dir}" --load_step 1290000
        --oru
        --wandb_activate --wandb_project HcFactory_TPA
        --wandb_name Hier4TPA-E2-N10-S42
        --algo_variant E2
        --ftg_thresh_phy 0.95
        agent.params.config.t_max_anchor=64000
        agent.params.config.max_episodic_steps=40000
        agent.params.config.parallel_producing_limit=10
        agent.params.config.c_forbid_none_mode=always
        agent.params.config.curriculum=false
        agent.params.config.explore=false
        agent.params.config.explore_catalog=false
        agent.params.config.catalog_collect=false
        agent.params.config.oru=true
        "agent.params.config.explore_catalog_dir=${catalog_root}"
        agent.params.config.oru_mix_start=0.25
        agent.params.config.oru_warmup_updates=0
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
    echo "[E2] E1 warmstart + ORU; catalog=${catalog_root}; mix_start=0.25; seed=42"
    echo "[E2] max_sim_episodes=${HC_MAX_TRAIN_EPISODES}; load_dir=${load_dir}; project=HcFactory_TPA; wandb=Hier4TPA-E2-N10-S42"
    if [[ "${dry_run}" == --dry-run ]]; then
        printf '%q ' "${cmd[@]}"
        printf '\n'
        return 0
    fi
    "${cmd[@]}"
}

run_e2_5_train() {
    # E2.5: E2 + credit scales only (no b_score_rl). Matches historical YAML-leak E2.
    local repo_root load_dir catalog_root dry_run="${3:-}"
    local teacher_eps="${HC_TEACHER_EPISODES:-50}"
    repo_root=$(cd -- "$(dirname -- "$0")" && pwd)
    cd "${repo_root}"
    load_dir="${repo_root}/logs/rl_games/HcFactory/hier_2026-08-27_23-17-41"
    if [[ -n "${HC_EXPLORE_CATALOG_DIR:-}" ]]; then
        catalog_root="${HC_EXPLORE_CATALOG_DIR}"
    else
        catalog_root="${repo_root}/env_checkpoints/policy_explore/N10_T40000__E2_teacher_ep${teacher_eps}"
        if [[ ! -d "${catalog_root}/offline_replay" ]]; then
            local alt="${repo_root}/env_checkpoints/random_explore/N10_T40000__T1_random_ep20"
            if [[ -d "${alt}/offline_replay" ]]; then
                echo "[E2.5] WARN: canonical teacher catalog missing; using ${alt}" >&2
                catalog_root="${alt}"
            fi
        fi
    fi
    if [[ ! "${DEVICE}" =~ ^cuda:[0-9]+$ && "${DEVICE}" != cpu ]]; then
        echo "错误: 设备需为 cuda:N 或 cpu" >&2
        return 1
    fi
    if [[ -n "${dry_run}" && "${dry_run}" != --dry-run ]] || (( $# > 3 )); then
        echo "用法: $0 E2.5 [cuda:N] [--dry-run]" >&2
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
    if [[ ! -d "${catalog_root}/offline_replay" ]]; then
        echo "错误: 缺少教师 offline_replay: ${catalog_root}/offline_replay" >&2
        echo "请先跑: ./run_2026_journal_experiments.sh TEACHER ${DEVICE}" >&2
        echo "或设置 HC_EXPLORE_CATALOG_DIR=... 指向采库根目录" >&2
        return 1
    fi
    local -a cmd=(
        python train.py --task HRTPaHC-v1 --algo hier
        --device "${DEVICE}" --num_envs 1 --headless --seed 42
        --train_n_products 10 --max_parallel_cd_dispatch 10
        --max_sim_episodes "${HC_MAX_TRAIN_EPISODES}"
        --load_dir "${load_dir}" --load_step 1290000
        --oru
        --wandb_activate --wandb_project HcFactory_TPA
        --wandb_name Hier4TPA-E2.5-N10-S42
        --algo_variant E2.5
        --ftg_thresh_phy 0.95
        agent.params.config.t_max_anchor=64000
        agent.params.config.max_episodic_steps=40000
        agent.params.config.parallel_producing_limit=10
        agent.params.config.c_forbid_none_mode=always
        agent.params.config.curriculum=false
        agent.params.config.explore=false
        agent.params.config.explore_catalog=false
        agent.params.config.catalog_collect=false
        agent.params.config.oru=true
        agent.params.config.hierarchical_credit=true
        agent.params.config.b_score_rl=false
        agent.params.config.credit_scale_A=2.0
        agent.params.config.credit_scale_B=1.5
        agent.params.config.credit_scale_CD=1.0
        "agent.params.config.explore_catalog_dir=${catalog_root}"
        agent.params.config.oru_mix_start=0.25
        agent.params.config.oru_warmup_updates=0
        agent.params.config.prioritized_replay=false
        agent.params.config.dueling_dqn=false
        agent.params.config.noisy_net=false
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
    echo "[E2.5] E2 + credit A×2.0 B×1.5 (no b_score); catalog=${catalog_root}; seed=42"
    echo "[E2.5] max_sim_episodes=${HC_MAX_TRAIN_EPISODES}; load_dir=${load_dir}; wandb=Hier4TPA-E2.5-N10-S42"
    if [[ "${dry_run}" == --dry-run ]]; then
        printf '%q ' "${cmd[@]}"
        printf '\n'
        return 0
    fi
    "${cmd[@]}"
}

run_e3_train() {
    # E3: E2 + teacher-guided online exploration (+E).
    local repo_root load_dir catalog_root dry_run="${3:-}"
    local teacher_eps="${HC_TEACHER_EPISODES:-50}"
    repo_root=$(cd -- "$(dirname -- "$0")" && pwd)
    cd "${repo_root}"
    load_dir="${repo_root}/logs/rl_games/HcFactory/hier_2026-08-27_23-17-41"
    if [[ -n "${HC_EXPLORE_CATALOG_DIR:-}" ]]; then
        catalog_root="${HC_EXPLORE_CATALOG_DIR}"
    else
        catalog_root="${repo_root}/env_checkpoints/policy_explore/N10_T40000__E2_teacher_ep${teacher_eps}"
        if [[ ! -d "${catalog_root}/offline_replay" ]]; then
            local alt="${repo_root}/env_checkpoints/random_explore/N10_T40000__T1_random_ep20"
            if [[ -d "${alt}/offline_replay" ]]; then
                echo "[E3] WARN: canonical teacher catalog missing; using ${alt}" >&2
                catalog_root="${alt}"
            fi
        fi
    fi
    if [[ ! "${DEVICE}" =~ ^cuda:[0-9]+$ && "${DEVICE}" != cpu ]]; then
        echo "错误: 设备需为 cuda:N 或 cpu" >&2
        return 1
    fi
    if [[ -n "${dry_run}" && "${dry_run}" != --dry-run ]] || (( $# > 3 )); then
        echo "用法: $0 E3 [cuda:N] [--dry-run]" >&2
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
    if [[ ! -d "${catalog_root}/offline_replay" ]]; then
        echo "错误: 缺少教师 offline_replay: ${catalog_root}/offline_replay" >&2
        echo "请先跑: ./run_2026_journal_experiments.sh TEACHER ${DEVICE}" >&2
        echo "或设置 HC_EXPLORE_CATALOG_DIR=... 指向采库根目录" >&2
        return 1
    fi
    local -a cmd=(
        python train.py --task HRTPaHC-v1 --algo hier
        --device "${DEVICE}" --num_envs 1 --headless --seed 42
        --train_n_products 10 --max_parallel_cd_dispatch 10
        --max_sim_episodes "${HC_MAX_TRAIN_EPISODES}"
        --load_dir "${load_dir}" --load_step 1290000
        --oru
        --teacher_explore
        --wandb_activate --wandb_project HcFactory_TPA
        --wandb_name Hier4TPA-E3-N10-S42
        --algo_variant E3
        --ftg_thresh_phy 0.95
        agent.params.config.t_max_anchor=64000
        agent.params.config.max_episodic_steps=40000
        agent.params.config.parallel_producing_limit=10
        agent.params.config.c_forbid_none_mode=always
        agent.params.config.curriculum=false
        agent.params.config.explore=false
        agent.params.config.explore_catalog=false
        agent.params.config.catalog_collect=false
        agent.params.config.oru=true
        agent.params.config.teacher_explore=true
        "agent.params.config.explore_catalog_dir=${catalog_root}"
        agent.params.config.oru_mix_start=0.25
        agent.params.config.oru_warmup_updates=0
        agent.params.config.teacher_explore_ratio_start=1.0
        agent.params.config.teacher_explore_ratio_end=0.0
        agent.params.config.teacher_explore_decay_env_steps=300000
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
    echo "[E3] E2 + teacher_explore; catalog=${catalog_root}; teacher_ratio 1→0 / 300k env steps; seed=42"
    echo "[E3] max_sim_episodes=${HC_MAX_TRAIN_EPISODES}; load_dir=${load_dir}; project=HcFactory_TPA; wandb=Hier4TPA-E3-N10-S42"
    if [[ "${dry_run}" == --dry-run ]]; then
        printf '%q ' "${cmd[@]}"
        printf '\n'
        return 0
    fi
    "${cmd[@]}"
}

run_e3_5_train() {
    # E3.5: E3 + credit scales only (no b_score_rl). Matches historical YAML-leak E3.
    local repo_root load_dir catalog_root dry_run="${3:-}"
    local teacher_eps="${HC_TEACHER_EPISODES:-50}"
    repo_root=$(cd -- "$(dirname -- "$0")" && pwd)
    cd "${repo_root}"
    load_dir="${repo_root}/logs/rl_games/HcFactory/hier_2026-08-27_23-17-41"
    if [[ -n "${HC_EXPLORE_CATALOG_DIR:-}" ]]; then
        catalog_root="${HC_EXPLORE_CATALOG_DIR}"
    else
        catalog_root="${repo_root}/env_checkpoints/policy_explore/N10_T40000__E2_teacher_ep${teacher_eps}"
        if [[ ! -d "${catalog_root}/offline_replay" ]]; then
            local alt="${repo_root}/env_checkpoints/random_explore/N10_T40000__T1_random_ep20"
            if [[ -d "${alt}/offline_replay" ]]; then
                echo "[E3.5] WARN: canonical teacher catalog missing; using ${alt}" >&2
                catalog_root="${alt}"
            fi
        fi
    fi
    if [[ ! "${DEVICE}" =~ ^cuda:[0-9]+$ && "${DEVICE}" != cpu ]]; then
        echo "错误: 设备需为 cuda:N 或 cpu" >&2
        return 1
    fi
    if [[ -n "${dry_run}" && "${dry_run}" != --dry-run ]] || (( $# > 3 )); then
        echo "用法: $0 E3.5 [cuda:N] [--dry-run]" >&2
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
    if [[ ! -d "${catalog_root}/offline_replay" ]]; then
        echo "错误: 缺少教师 offline_replay: ${catalog_root}/offline_replay" >&2
        echo "请先跑: ./run_2026_journal_experiments.sh TEACHER ${DEVICE}" >&2
        echo "或设置 HC_EXPLORE_CATALOG_DIR=... 指向采库根目录" >&2
        return 1
    fi
    local -a cmd=(
        python train.py --task HRTPaHC-v1 --algo hier
        --device "${DEVICE}" --num_envs 1 --headless --seed 42
        --train_n_products 10 --max_parallel_cd_dispatch 10
        --max_sim_episodes "${HC_MAX_TRAIN_EPISODES}"
        --load_dir "${load_dir}" --load_step 1290000
        --oru
        --teacher_explore
        --wandb_activate --wandb_project HcFactory_TPA
        --wandb_name Hier4TPA-E3.5-N10-S42
        --algo_variant E3.5
        --ftg_thresh_phy 0.95
        agent.params.config.t_max_anchor=64000
        agent.params.config.max_episodic_steps=40000
        agent.params.config.parallel_producing_limit=10
        agent.params.config.c_forbid_none_mode=always
        agent.params.config.curriculum=false
        agent.params.config.explore=false
        agent.params.config.explore_catalog=false
        agent.params.config.catalog_collect=false
        agent.params.config.oru=true
        agent.params.config.teacher_explore=true
        agent.params.config.hierarchical_credit=true
        agent.params.config.b_score_rl=false
        agent.params.config.credit_scale_A=2.0
        agent.params.config.credit_scale_B=1.5
        agent.params.config.credit_scale_CD=1.0
        "agent.params.config.explore_catalog_dir=${catalog_root}"
        agent.params.config.oru_mix_start=0.25
        agent.params.config.oru_warmup_updates=0
        agent.params.config.teacher_explore_ratio_start=1.0
        agent.params.config.teacher_explore_ratio_end=0.0
        agent.params.config.teacher_explore_decay_env_steps=300000
        agent.params.config.prioritized_replay=false
        agent.params.config.dueling_dqn=false
        agent.params.config.noisy_net=false
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
    echo "[E3.5] E3 + credit A×2.0 B×1.5 (no b_score); catalog=${catalog_root}; seed=42"
    echo "[E3.5] max_sim_episodes=${HC_MAX_TRAIN_EPISODES}; load_dir=${load_dir}; wandb=Hier4TPA-E3.5-N10-S42"
    if [[ "${dry_run}" == --dry-run ]]; then
        printf '%q ' "${cmd[@]}"
        printf '\n'
        return 0
    fi
    "${cmd[@]}"
}

run_e4_train() {
    # E4: E3 + hierarchical credit (A/B scales) + B-score RL.
    local repo_root load_dir catalog_root dry_run="${3:-}"
    local teacher_eps="${HC_TEACHER_EPISODES:-50}"
    repo_root=$(cd -- "$(dirname -- "$0")" && pwd)
    cd "${repo_root}"
    load_dir="${repo_root}/logs/rl_games/HcFactory/hier_2026-08-27_23-17-41"
    if [[ -n "${HC_EXPLORE_CATALOG_DIR:-}" ]]; then
        catalog_root="${HC_EXPLORE_CATALOG_DIR}"
    else
        catalog_root="${repo_root}/env_checkpoints/policy_explore/N10_T40000__E2_teacher_ep${teacher_eps}"
        if [[ ! -d "${catalog_root}/offline_replay" ]]; then
            local alt="${repo_root}/env_checkpoints/random_explore/N10_T40000__T1_random_ep20"
            if [[ -d "${alt}/offline_replay" ]]; then
                echo "[E4] WARN: canonical teacher catalog missing; using ${alt}" >&2
                catalog_root="${alt}"
            fi
        fi
    fi
    if [[ ! "${DEVICE}" =~ ^cuda:[0-9]+$ && "${DEVICE}" != cpu ]]; then
        echo "错误: 设备需为 cuda:N 或 cpu" >&2
        return 1
    fi
    if [[ -n "${dry_run}" && "${dry_run}" != --dry-run ]] || (( $# > 3 )); then
        echo "用法: $0 E4 [cuda:N] [--dry-run]" >&2
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
    if [[ ! -d "${catalog_root}/offline_replay" ]]; then
        echo "错误: 缺少教师 offline_replay: ${catalog_root}/offline_replay" >&2
        echo "请先跑: ./run_2026_journal_experiments.sh TEACHER ${DEVICE}" >&2
        echo "或设置 HC_EXPLORE_CATALOG_DIR=... 指向采库根目录" >&2
        return 1
    fi
    local -a cmd=(
        python train.py --task HRTPaHC-v1 --algo hier
        --device "${DEVICE}" --num_envs 1 --headless --seed 42
        --train_n_products 10 --max_parallel_cd_dispatch 10
        --max_sim_episodes "${HC_MAX_TRAIN_EPISODES}"
        --load_dir "${load_dir}" --load_step 1290000
        --oru
        --teacher_explore
        --hierarchical_credit
        --b_score_rl
        --wandb_activate --wandb_project HcFactory_TPA
        --wandb_name Hier4TPA-E4-N10-S42
        --algo_variant E4
        --ftg_thresh_phy 0.95
        agent.params.config.t_max_anchor=64000
        agent.params.config.max_episodic_steps=40000
        agent.params.config.parallel_producing_limit=10
        agent.params.config.c_forbid_none_mode=always
        agent.params.config.curriculum=false
        agent.params.config.explore=false
        agent.params.config.explore_catalog=false
        agent.params.config.catalog_collect=false
        agent.params.config.oru=true
        agent.params.config.teacher_explore=true
        agent.params.config.hierarchical_credit=true
        agent.params.config.b_score_rl=true
        agent.params.config.credit_scale_A=2.0
        agent.params.config.credit_scale_B=1.5
        agent.params.config.credit_scale_CD=1.0
        "agent.params.config.explore_catalog_dir=${catalog_root}"
        agent.params.config.oru_mix_start=0.25
        agent.params.config.oru_warmup_updates=0
        agent.params.config.teacher_explore_ratio_start=1.0
        agent.params.config.teacher_explore_ratio_end=0.0
        agent.params.config.teacher_explore_decay_env_steps=300000
        agent.params.config.prioritized_replay=false
        agent.params.config.dueling_dqn=false
        agent.params.config.noisy_net=false
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
    echo "[E4] E3 + hier_credit + b_score_rl; A×2.0 B×1.5; catalog=${catalog_root}; seed=42"
    echo "[E4] max_sim_episodes=${HC_MAX_TRAIN_EPISODES}; load_dir=${load_dir}; project=HcFactory_TPA; wandb=Hier4TPA-E4-N10-S42"
    if [[ "${dry_run}" == --dry-run ]]; then
        printf '%q ' "${cmd[@]}"
        printf '\n'
        return 0
    fi
    "${cmd[@]}"
}

run_e3_no_oru_train() {
    # E3-no-oru: E1 + teacher_explore, no ORU (ablation: guide without data).
    local repo_root load_dir dry_run="${3:-}"
    repo_root=$(cd -- "$(dirname -- "$0")" && pwd)
    cd "${repo_root}"
    load_dir="${repo_root}/logs/rl_games/HcFactory/hier_2026-08-27_23-17-41"
    if [[ ! "${DEVICE}" =~ ^cuda:[0-9]+$ && "${DEVICE}" != cpu ]]; then
        echo "错误: 设备需为 cuda:N 或 cpu" >&2
        return 1
    fi
    if [[ -n "${dry_run}" && "${dry_run}" != --dry-run ]] || (( $# > 3 )); then
        echo "用法: $0 E3-no-oru [cuda:N] [--dry-run]" >&2
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
        --teacher_explore
        --wandb_activate --wandb_project HcFactory_TPA
        --wandb_name Hier4TPA-E3-no-oru-N10-S42
        --algo_variant E3-no-oru
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
        agent.params.config.teacher_explore=true
        agent.params.config.teacher_explore_ratio_start=1.0
        agent.params.config.teacher_explore_ratio_end=0.0
        agent.params.config.teacher_explore_decay_env_steps=300000
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
    echo "[E3-no-oru] E1 + teacher_explore; oru=false (no catalog); seed=42"
    echo "[E3-no-oru] max_sim_episodes=${HC_MAX_TRAIN_EPISODES}; load_dir=${load_dir}; wandb=Hier4TPA-E3-no-oru-N10-S42"
    if [[ "${dry_run}" == --dry-run ]]; then
        printf '%q ' "${cmd[@]}"
        printf '\n'
        return 0
    fi
    "${cmd[@]}"
}

run_e4_no_oru_train() {
    # E4-no-oru: E4 without ORU (keep teacher_explore + hierarchical credit + b_score).
    local repo_root load_dir dry_run="${3:-}"
    repo_root=$(cd -- "$(dirname -- "$0")" && pwd)
    cd "${repo_root}"
    load_dir="${repo_root}/logs/rl_games/HcFactory/hier_2026-08-27_23-17-41"
    if [[ ! "${DEVICE}" =~ ^cuda:[0-9]+$ && "${DEVICE}" != cpu ]]; then
        echo "错误: 设备需为 cuda:N 或 cpu" >&2
        return 1
    fi
    if [[ -n "${dry_run}" && "${dry_run}" != --dry-run ]] || (( $# > 3 )); then
        echo "用法: $0 E4-no-oru [cuda:N] [--dry-run]" >&2
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
        --teacher_explore
        --hierarchical_credit
        --b_score_rl
        --wandb_activate --wandb_project HcFactory_TPA
        --wandb_name Hier4TPA-E4-no-oru-N10-S42
        --algo_variant E4-no-oru
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
        agent.params.config.teacher_explore=true
        agent.params.config.hierarchical_credit=true
        agent.params.config.b_score_rl=true
        agent.params.config.credit_scale_A=2.0
        agent.params.config.credit_scale_B=1.5
        agent.params.config.credit_scale_CD=1.0
        agent.params.config.teacher_explore_ratio_start=1.0
        agent.params.config.teacher_explore_ratio_end=0.0
        agent.params.config.teacher_explore_decay_env_steps=300000
        agent.params.config.prioritized_replay=false
        agent.params.config.dueling_dqn=false
        agent.params.config.noisy_net=false
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
    echo "[E4-no-oru] E4 without ORU; teacher_explore+hier_credit+b_score; seed=42"
    echo "[E4-no-oru] max_sim_episodes=${HC_MAX_TRAIN_EPISODES}; load_dir=${load_dir}; wandb=Hier4TPA-E4-no-oru-N10-S42"
    if [[ "${dry_run}" == --dry-run ]]; then
        printf '%q ' "${cmd[@]}"
        printf '\n'
        return 0
    fi
    "${cmd[@]}"
}

run_e5_train() {
    # E5: E3 + autoregressive only (no hierarchical credit / b_score).
    local repo_root load_dir catalog_root dry_run="${3:-}"
    local teacher_eps="${HC_TEACHER_EPISODES:-50}"
    repo_root=$(cd -- "$(dirname -- "$0")" && pwd)
    cd "${repo_root}"
    load_dir="${repo_root}/logs/rl_games/HcFactory/hier_2026-08-27_23-17-41"
    if [[ -n "${HC_EXPLORE_CATALOG_DIR:-}" ]]; then
        catalog_root="${HC_EXPLORE_CATALOG_DIR}"
    else
        catalog_root="${repo_root}/env_checkpoints/policy_explore/N10_T40000__E2_teacher_ep${teacher_eps}"
        if [[ ! -d "${catalog_root}/offline_replay" ]]; then
            local alt="${repo_root}/env_checkpoints/random_explore/N10_T40000__T1_random_ep20"
            if [[ -d "${alt}/offline_replay" ]]; then
                echo "[E5] WARN: canonical teacher catalog missing; using ${alt}" >&2
                catalog_root="${alt}"
            fi
        fi
    fi
    if [[ ! "${DEVICE}" =~ ^cuda:[0-9]+$ && "${DEVICE}" != cpu ]]; then
        echo "错误: 设备需为 cuda:N 或 cpu" >&2
        return 1
    fi
    if [[ -n "${dry_run}" && "${dry_run}" != --dry-run ]] || (( $# > 3 )); then
        echo "用法: $0 E5 [cuda:N] [--dry-run]" >&2
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
    if [[ ! -d "${catalog_root}/offline_replay" ]]; then
        echo "错误: 缺少教师 offline_replay: ${catalog_root}/offline_replay" >&2
        echo "请先跑: ./run_2026_journal_experiments.sh TEACHER ${DEVICE}" >&2
        echo "或设置 HC_EXPLORE_CATALOG_DIR=... 指向采库根目录" >&2
        return 1
    fi
    local -a cmd=(
        python train.py --task HRTPaHC-v1 --algo hier
        --device "${DEVICE}" --num_envs 1 --headless --seed 42
        --train_n_products 10 --max_parallel_cd_dispatch 10
        --max_sim_episodes "${HC_MAX_TRAIN_EPISODES}"
        --load_dir "${load_dir}" --load_step 1290000
        --oru
        --teacher_explore
        --autoregressive
        --wandb_activate --wandb_project HcFactory_TPA
        --wandb_name Hier4TPA-E5-N10-S42
        --algo_variant E5
        --ftg_thresh_phy 0.95
        agent.params.config.t_max_anchor=64000
        agent.params.config.max_episodic_steps=40000
        agent.params.config.parallel_producing_limit=10
        agent.params.config.c_forbid_none_mode=always
        agent.params.config.curriculum=false
        agent.params.config.explore=false
        agent.params.config.explore_catalog=false
        agent.params.config.catalog_collect=false
        agent.params.config.oru=true
        agent.params.config.teacher_explore=true
        agent.params.config.autoregressive=true
        "agent.params.config.explore_catalog_dir=${catalog_root}"
        agent.params.config.oru_mix_start=0.25
        agent.params.config.oru_warmup_updates=0
        agent.params.config.teacher_explore_ratio_start=1.0
        agent.params.config.teacher_explore_ratio_end=0.0
        agent.params.config.teacher_explore_decay_env_steps=300000
        agent.params.config.ar_n_candidates=4
        agent.params.config.ar_softmax_temperature=1.0
        agent.params.config.ar_eps_scale_A=0.5
        agent.params.config.ar_eps_scale_B=0.5
        agent.params.config.ar_eps_scale_C=1.0
        agent.params.config.ar_eps_scale_D=1.0
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
    echo "[E5] E3 + autoregressive (layered ε + cand=4); no hier_credit/b_score; catalog=${catalog_root}; seed=42"
    echo "[E5] max_sim_episodes=${HC_MAX_TRAIN_EPISODES}; load_dir=${load_dir}; wandb=Hier4TPA-E5-N10-S42"
    if [[ "${dry_run}" == --dry-run ]]; then
        printf '%q ' "${cmd[@]}"
        printf '\n'
        return 0
    fi
    "${cmd[@]}"
}

run_e6_train() {
    # E6: E5 + E4 = E3 + AR + hierarchical credit + b_score (full method).
    local repo_root load_dir catalog_root dry_run="${3:-}"
    local teacher_eps="${HC_TEACHER_EPISODES:-50}"
    repo_root=$(cd -- "$(dirname -- "$0")" && pwd)
    cd "${repo_root}"
    load_dir="${repo_root}/logs/rl_games/HcFactory/hier_2026-08-27_23-17-41"
    if [[ -n "${HC_EXPLORE_CATALOG_DIR:-}" ]]; then
        catalog_root="${HC_EXPLORE_CATALOG_DIR}"
    else
        catalog_root="${repo_root}/env_checkpoints/policy_explore/N10_T40000__E2_teacher_ep${teacher_eps}"
        if [[ ! -d "${catalog_root}/offline_replay" ]]; then
            local alt="${repo_root}/env_checkpoints/random_explore/N10_T40000__T1_random_ep20"
            if [[ -d "${alt}/offline_replay" ]]; then
                echo "[E6] WARN: canonical teacher catalog missing; using ${alt}" >&2
                catalog_root="${alt}"
            fi
        fi
    fi
    if [[ ! "${DEVICE}" =~ ^cuda:[0-9]+$ && "${DEVICE}" != cpu ]]; then
        echo "错误: 设备需为 cuda:N 或 cpu" >&2
        return 1
    fi
    if [[ -n "${dry_run}" && "${dry_run}" != --dry-run ]] || (( $# > 3 )); then
        echo "用法: $0 E6 [cuda:N] [--dry-run]" >&2
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
    if [[ ! -d "${catalog_root}/offline_replay" ]]; then
        echo "错误: 缺少教师 offline_replay: ${catalog_root}/offline_replay" >&2
        echo "请先跑: ./run_2026_journal_experiments.sh TEACHER ${DEVICE}" >&2
        echo "或设置 HC_EXPLORE_CATALOG_DIR=... 指向采库根目录" >&2
        return 1
    fi
    local -a cmd=(
        python train.py --task HRTPaHC-v1 --algo hier
        --device "${DEVICE}" --num_envs 1 --headless --seed 42
        --train_n_products 10 --max_parallel_cd_dispatch 10
        --max_sim_episodes "${HC_MAX_TRAIN_EPISODES}"
        --load_dir "${load_dir}" --load_step 1290000
        --oru
        --teacher_explore
        --autoregressive
        --hierarchical_credit
        --b_score_rl
        --wandb_activate --wandb_project HcFactory_TPA
        --wandb_name Hier4TPA-E6-N10-S42
        --algo_variant E6
        --ftg_thresh_phy 0.95
        agent.params.config.t_max_anchor=64000
        agent.params.config.max_episodic_steps=40000
        agent.params.config.parallel_producing_limit=10
        agent.params.config.c_forbid_none_mode=always
        agent.params.config.curriculum=false
        agent.params.config.explore=false
        agent.params.config.explore_catalog=false
        agent.params.config.catalog_collect=false
        agent.params.config.oru=true
        agent.params.config.teacher_explore=true
        agent.params.config.autoregressive=true
        agent.params.config.hierarchical_credit=true
        agent.params.config.b_score_rl=true
        agent.params.config.credit_scale_A=2.0
        agent.params.config.credit_scale_B=1.5
        agent.params.config.credit_scale_CD=1.0
        "agent.params.config.explore_catalog_dir=${catalog_root}"
        agent.params.config.oru_mix_start=0.25
        agent.params.config.oru_warmup_updates=0
        agent.params.config.teacher_explore_ratio_start=1.0
        agent.params.config.teacher_explore_ratio_end=0.0
        agent.params.config.teacher_explore_decay_env_steps=300000
        agent.params.config.ar_n_candidates=4
        agent.params.config.ar_softmax_temperature=1.0
        agent.params.config.ar_eps_scale_A=0.5
        agent.params.config.ar_eps_scale_B=0.5
        agent.params.config.ar_eps_scale_C=1.0
        agent.params.config.ar_eps_scale_D=1.0
        agent.params.config.prioritized_replay=false
        agent.params.config.dueling_dqn=false
        agent.params.config.noisy_net=false
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
    echo "[E6] E5+E4 full method (AR+hier_credit+b_score); catalog=${catalog_root}; seed=42"
    echo "[E6] max_sim_episodes=${HC_MAX_TRAIN_EPISODES}; load_dir=${load_dir}; wandb=Hier4TPA-E6-N10-S42"
    if [[ "${dry_run}" == --dry-run ]]; then
        printf '%q ' "${cmd[@]}"
        printf '\n'
        return 0
    fi
    "${cmd[@]}"
}

run_e5_no_oru_train() {
    # E5-no-oru: E5 without ORU (AR + teacher_explore; no catalog).
    local repo_root load_dir dry_run="${3:-}"
    repo_root=$(cd -- "$(dirname -- "$0")" && pwd)
    cd "${repo_root}"
    load_dir="${repo_root}/logs/rl_games/HcFactory/hier_2026-08-27_23-17-41"
    if [[ ! "${DEVICE}" =~ ^cuda:[0-9]+$ && "${DEVICE}" != cpu ]]; then
        echo "错误: 设备需为 cuda:N 或 cpu" >&2
        return 1
    fi
    if [[ -n "${dry_run}" && "${dry_run}" != --dry-run ]] || (( $# > 3 )); then
        echo "用法: $0 E5-no-oru [cuda:N] [--dry-run]" >&2
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
        --teacher_explore
        --autoregressive
        --wandb_activate --wandb_project HcFactory_TPA
        --wandb_name Hier4TPA-E5-no-oru-N10-S42
        --algo_variant E5-no-oru
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
        agent.params.config.teacher_explore=true
        agent.params.config.autoregressive=true
        agent.params.config.teacher_explore_ratio_start=1.0
        agent.params.config.teacher_explore_ratio_end=0.0
        agent.params.config.teacher_explore_decay_env_steps=300000
        agent.params.config.ar_n_candidates=4
        agent.params.config.ar_softmax_temperature=1.0
        agent.params.config.ar_eps_scale_A=0.5
        agent.params.config.ar_eps_scale_B=0.5
        agent.params.config.ar_eps_scale_C=1.0
        agent.params.config.ar_eps_scale_D=1.0
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
    echo "[E5-no-oru] E5 without ORU; teacher_explore+AR; seed=42"
    echo "[E5-no-oru] max_sim_episodes=${HC_MAX_TRAIN_EPISODES}; load_dir=${load_dir}; wandb=Hier4TPA-E5-no-oru-N10-S42"
    if [[ "${dry_run}" == --dry-run ]]; then
        printf '%q ' "${cmd[@]}"
        printf '\n'
        return 0
    fi
    "${cmd[@]}"
}

run_e6_no_oru_train() {
    # E6-no-oru: E6 without ORU (AR + hier + b_score + teacher_explore; no catalog).
    local repo_root load_dir dry_run="${3:-}"
    repo_root=$(cd -- "$(dirname -- "$0")" && pwd)
    cd "${repo_root}"
    load_dir="${repo_root}/logs/rl_games/HcFactory/hier_2026-08-27_23-17-41"
    if [[ ! "${DEVICE}" =~ ^cuda:[0-9]+$ && "${DEVICE}" != cpu ]]; then
        echo "错误: 设备需为 cuda:N 或 cpu" >&2
        return 1
    fi
    if [[ -n "${dry_run}" && "${dry_run}" != --dry-run ]] || (( $# > 3 )); then
        echo "用法: $0 E6-no-oru [cuda:N] [--dry-run]" >&2
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
        --teacher_explore
        --autoregressive
        --hierarchical_credit
        --b_score_rl
        --wandb_activate --wandb_project HcFactory_TPA
        --wandb_name Hier4TPA-E6-no-oru-N10-S42
        --algo_variant E6-no-oru
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
        agent.params.config.teacher_explore=true
        agent.params.config.autoregressive=true
        agent.params.config.hierarchical_credit=true
        agent.params.config.b_score_rl=true
        agent.params.config.credit_scale_A=2.0
        agent.params.config.credit_scale_B=1.5
        agent.params.config.credit_scale_CD=1.0
        agent.params.config.teacher_explore_ratio_start=1.0
        agent.params.config.teacher_explore_ratio_end=0.0
        agent.params.config.teacher_explore_decay_env_steps=300000
        agent.params.config.ar_n_candidates=4
        agent.params.config.ar_softmax_temperature=1.0
        agent.params.config.ar_eps_scale_A=0.5
        agent.params.config.ar_eps_scale_B=0.5
        agent.params.config.ar_eps_scale_C=1.0
        agent.params.config.ar_eps_scale_D=1.0
        agent.params.config.prioritized_replay=false
        agent.params.config.dueling_dqn=false
        agent.params.config.noisy_net=false
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
    echo "[E6-no-oru] E6 without ORU; AR+hier_credit+b_score+teacher_explore; seed=42"
    echo "[E6-no-oru] max_sim_episodes=${HC_MAX_TRAIN_EPISODES}; load_dir=${load_dir}; wandb=Hier4TPA-E6-no-oru-N10-S42"
    if [[ "${dry_run}" == --dry-run ]]; then
        printf '%q ' "${cmd[@]}"
        printf '\n'
        return 0
    fi
    "${cmd[@]}"
}

run_journal_stub() {
    local name="$1"
    echo "[journal] '${name}' 已在 docs/experiment_protocol.md 命名预留，代码入口尚未实现。" >&2
    echo "[journal] 见协议 §2；当前可跑: E0–E6、E*-no-oru、E*.5、TEACHER。" >&2
    return 1
}

# 5090 panel: train-best ckpts that live under isaac_factory_tpa (see docs/eval_checkpoint_selection.md).
# Default skips E0 (already finished on W&B df55hqiz). Set HC_EVAL_INCLUDE_E0=1 to include.
run_eval_5090_panel() {
    local repo_root dry_run="${3:-}" base enc
    repo_root=$(cd -- "$(dirname -- "$0")" && pwd)
    cd "${repo_root}"
    if [[ ! "${DEVICE}" =~ ^cuda:[0-9]+$ && "${DEVICE}" != cpu ]]; then
        echo "错误: 设备需为 cuda:N 或 cpu" >&2
        return 1
    fi
    if [[ -n "${dry_run}" && "${dry_run}" != --dry-run ]] || (( $# > 3 )); then
        echo "用法: $0 eval-5090 [cuda:N] [--dry-run]" >&2
        return 1
    fi

    base="${repo_root}/logs/rl_games/HcFactory"
    # variant|rel_dir|best_step
    local -a jobs=(
        "E1|hier_2026-09-13_15-20-57|1080000"
        "E2|hier_2026-09-15_18-39-11|145000"
        "E2.5|hier_2026-09-11_19-46-22|850000"
        "E6|hier_2026-09-17_10-05-15|415000"
    )

    echo "[eval-5090] device=${DEVICE}; seeds=${HC_TEST_SEEDS}; times=${HC_TEST_TIMES}; base=${base}"
    echo "[eval-5090] ckpt rule: train-best FO makespan → nearest save_interval (docs/eval_checkpoint_selection.md)"
    export HC_EVAL_SEED_CHUNK="${HC_EVAL_SEED_CHUNK:-2}"
    echo "[eval-5090] seed_chunk=${HC_EVAL_SEED_CHUNK} (fresh process per chunk; verify jsonl before next)"

    if [[ "${HC_EVAL_INCLUDE_E0:-0}" == "1" ]]; then
        echo "[eval-5090] --- E0 (include) ---"
        if [[ "${dry_run}" == --dry-run ]]; then
            echo "[eval-5090] dry-run: would run E0"
        else
            run_e0_eval "${MODE}" "${DEVICE}"
        fi
    else
        echo "[eval-5090] skip E0 (already have W&B eval); set HC_EVAL_INCLUDE_E0=1 to force"
    fi

    local spec variant rel step load_dir
    for spec in "${jobs[@]}"; do
        IFS='|' read -r variant rel step <<<"${spec}"
        load_dir="${base}/${rel}"
        enc="${load_dir}/nn/state_encoder_step_${step}.pth"
        echo "[eval-5090] --- ${variant} step=${step} dir=${rel} ---"
        if [[ "${dry_run}" == --dry-run ]]; then
            echo "[eval-5090] dry-run: HC_LOAD_DIR=${load_dir} HC_LOAD_STEP=${step} HC_EVAL_VARIANT=${variant}"
            echo "[eval-5090] dry-run: would require ${enc}"
            continue
        fi
        if [[ ! -f "${enc}" ]]; then
            echo "错误: 缺少训练最优 ckpt: ${enc}" >&2
            echo "提示: 可 ls ${load_dir}/nn/state_encoder_step_*.pth 后改 docs 表 / 本函数 step" >&2
            return 1
        fi
        # Each experiment must own its own W&B run + eval output dir.
        unset HC_WANDB_RUN_ID WANDB_RUN_ID HC_EVAL_OUTPUT_DIR HC_EVAL_EPISODE_OFFSET HC_EVAL_APPEND
        export HC_LOAD_DIR="${load_dir}"
        export HC_LOAD_STEP="${step}"
        export HC_EVAL_VARIANT="${variant}"
        export HC_WANDB_NAME="Hier4TPA-${variant}-N10-S42-step${step}-eval"
        EVAL_STEPS=""
        run_hier_eval_for_n 10
    done
    echo "[eval-5090] done"
}

# Shared helper: run a list of variant|rel_dir|step jobs under logs/rl_games/HcFactory.
run_eval_job_panel() {
    local panel_name="$1"
    shift
    local dry_run=""
    # optional trailing --dry-run in "$@" from caller
    local -a jobs=("$@")
    if [[ "${#jobs[@]}" -gt 0 && "${jobs[-1]}" == --dry-run ]]; then
        dry_run=--dry-run
        unset 'jobs[-1]'
    fi
    local repo_root base
    repo_root=$(cd -- "$(dirname -- "$0")" && pwd)
    cd "${repo_root}"
    base="${repo_root}/logs/rl_games/HcFactory"
    export HC_EVAL_SEED_CHUNK="${HC_EVAL_SEED_CHUNK:-2}"
    echo "[${panel_name}] device=${DEVICE}; seeds=${HC_TEST_SEEDS}; chunk=${HC_EVAL_SEED_CHUNK}; jobs=${#jobs[@]}"
    local spec variant rel step load_dir enc
    for spec in "${jobs[@]}"; do
        IFS='|' read -r variant rel step <<<"${spec}"
        load_dir="${base}/${rel}"
        enc="${load_dir}/nn/state_encoder_step_${step}.pth"
        echo "[${panel_name}] --- ${variant} step=${step} dir=${rel} ---"
        if [[ "${dry_run}" == --dry-run ]]; then
            echo "[${panel_name}] dry-run: would require ${enc}"
            continue
        fi
        if [[ ! -f "${enc}" ]]; then
            echo "错误: 缺少训练最优 ckpt: ${enc}" >&2
            return 1
        fi
        unset HC_WANDB_RUN_ID WANDB_RUN_ID HC_EVAL_OUTPUT_DIR HC_EVAL_EPISODE_OFFSET HC_EVAL_APPEND
        export HC_LOAD_DIR="${load_dir}"
        export HC_LOAD_STEP="${step}"
        export HC_EVAL_VARIANT="${variant}"
        export HC_WANDB_NAME="Hier4TPA-${variant}-N10-S42-step${step}-eval"
        EVAL_STEPS=""
        run_hier_eval_for_n 10
    done
    echo "[${panel_name}] done"
}

# Desk-trained E* (synced train-best steps). See docs/eval_checkpoint_selection.md.
# reverse=1 → E3 … E4（本机与 5090 eval-desk 对开，少抢同一实验）。
run_eval_desk_panel() {
    local dry_run="${3:-}" reverse="${4:-0}"
    if [[ ! "${DEVICE}" =~ ^cuda:[0-9]+$ && "${DEVICE}" != cpu ]]; then
        echo "错误: 设备需为 cuda:N 或 cpu" >&2
        return 1
    fi
    if [[ -n "${dry_run}" && "${dry_run}" != --dry-run ]] || (( $# > 4 )); then
        echo "用法: $0 eval-desk|eval-desk-rev [cuda:N] [--dry-run]" >&2
        return 1
    fi
    # Forward priority for paper table (desk-trained; synced train-best steps).
    # E5-no-oru: peak 360000 训练下降最快但 eval 差；另加后半程最优 870000。
    local -a jobs=(
        "E4|hier_2026-09-12_10-20-18|645000"
        "E5-no-oru|hier_2026-09-18_21-14-57|360000"
        "E5-no-oru-late|hier_2026-09-18_21-14-57|870000"
        "E5|hier_2026-09-17_06-43-17|750000"
        "E3.5|hier_2026-09-10_19-58-56|595000"
        "E3-no-oru|hier_2026-09-14_01-26-32|595000"
        "E1.5|hier_2026-09-09_15-05-17|755000"
        "E3|hier_2026-09-15_14-37-23|715000"
    )
    local panel_name=eval-desk
    if [[ "${reverse}" == "1" || "${HC_EVAL_DESK_REVERSE:-0}" == "1" ]]; then
        local -a rev=()
        local i
        for ((i=${#jobs[@]}-1; i>=0; i--)); do
            rev+=("${jobs[i]}")
        done
        jobs=("${rev[@]}")
        panel_name=eval-desk-rev
    fi
    if [[ "${dry_run}" == --dry-run ]]; then
        run_eval_job_panel "${panel_name}" "${jobs[@]}" --dry-run
    else
        run_eval_job_panel "${panel_name}" "${jobs[@]}"
    fi
}

# E5 desk→5090 eval (train-best step 750000). Requires synced nn/*_step_750000.pth.
run_eval_e5() {
    local dry_run="${3:-}"
    if [[ ! "${DEVICE}" =~ ^cuda:[0-9]+$ && "${DEVICE}" != cpu ]]; then
        echo "错误: 设备需为 cuda:N 或 cpu" >&2
        return 1
    fi
    if [[ -n "${dry_run}" && "${dry_run}" != --dry-run ]] || (( $# > 3 )); then
        echo "用法: $0 eval-E5 [cuda:N] [--dry-run]" >&2
        return 1
    fi
    local -a jobs=("E5|hier_2026-09-17_06-43-17|750000")
    if [[ "${dry_run}" == --dry-run ]]; then
        run_eval_job_panel eval-E5 "${jobs[@]}" --dry-run
    else
        run_eval_job_panel eval-E5 "${jobs[@]}"
    fi
}

# E5-no-oru late (half2 best step 870000) — peak 360000 下降快但 eval 差。
run_eval_e5_no_oru_late() {
    local dry_run="${3:-}"
    if [[ ! "${DEVICE}" =~ ^cuda:[0-9]+$ && "${DEVICE}" != cpu ]]; then
        echo "错误: 设备需为 cuda:N 或 cpu" >&2
        return 1
    fi
    if [[ -n "${dry_run}" && "${dry_run}" != --dry-run ]] || (( $# > 3 )); then
        echo "用法: $0 eval-E5-no-oru-late [cuda:N] [--dry-run]" >&2
        return 1
    fi
    local -a jobs=("E5-no-oru-late|hier_2026-09-18_21-14-57|870000")
    if [[ "${dry_run}" == --dry-run ]]; then
        run_eval_job_panel eval-E5-no-oru-late "${jobs[@]}" --dry-run
    else
        run_eval_job_panel eval-E5-no-oru-late "${jobs[@]}"
    fi
}

case "${MODE}" in
    E0) run_e0_eval "$@" ;;
    E1) run_e1_train "$@" ;;
    E1.5|E1_5) run_e1_5_train "$@" ;;
    E2) run_e2_train "$@" ;;
    E2.5|E2_5) run_e2_5_train "$@" ;;
    E3) run_e3_train "$@" ;;
    E3-no-oru|E3_no_oru) run_e3_no_oru_train "$@" ;;
    E3.5|E3_5) run_e3_5_train "$@" ;;
    E4) run_e4_train "$@" ;;
    E4-no-oru|E4_no_oru) run_e4_no_oru_train "$@" ;;
    E5) run_e5_train "$@" ;;
    E5-no-oru|E5_no_oru) run_e5_no_oru_train "$@" ;;
    E6) run_e6_train "$@" ;;
    E6-no-oru|E6_no_oru) run_e6_no_oru_train "$@" ;;
    E6-no-guide|E6_no_guide|E6-no-hier|E6_no_hier|E6-no-ar|E6_no_ar|E6-plus-replay|E6_plus_replay|E6-plus-curriculum|E6_plus_curriculum|E6-plus-staged|E6_plus_staged|E2-random-data|E2_random_data)
        run_journal_stub "${MODE}"
        ;;
    ""|-h|--help|help) usage; exit 0 ;;
    T0) ./batch_train.sh T0 "${DEVICE}" ;;
    T1|train) ./batch_train.sh T1 "${DEVICE}" ;;
    T1R) ./batch_train.sh T1R "${DEVICE}" ;;
    T1RH) ./batch_train.sh T1RH "${DEVICE}" ;;
    TEACHER|teacher|E2-collect)
        # Do not inherit journal default T1_random_ep20 into teacher catalog.
        unset HC_CATALOG_TAG HC_CATALOG_SOURCE HC_EXPLORE_CATALOG_DIR
        ./batch_train.sh TEACHER "${DEVICE}"
        ;;
    eval-T0) run_eval_variant T0 ;;
    eval-T1) run_eval_variant T1 ;;
    eval-T1R) run_eval_variant T1R ;;
    eval-T1RH) run_eval_variant T1RH ;;
    eval-5090|eval_5090) run_eval_5090_panel "$@" ;;
    eval-E5|eval_E5|E5-eval) run_eval_e5 "$@" ;;
    eval-E5-no-oru-late|eval_E5_no_oru_late|E5-no-oru-late-eval) run_eval_e5_no_oru_late "$@" ;;
    eval-desk|eval_desk) run_eval_desk_panel "$@" ;;
    eval-desk-rev|eval_desk_rev)
        # MODE DEVICE [--dry-run] → pass reverse=1 as 4th arg to panel
        run_eval_desk_panel "${MODE}" "${DEVICE}" "${3:-}" 1
        ;;
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
