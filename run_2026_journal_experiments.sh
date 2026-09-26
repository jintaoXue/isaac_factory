#!/bin/bash
set -euo pipefail

# 训练/数值评测默认纯逻辑后端：不启动 Isaac，不推进物理或渲染，仍逐逻辑步更新。
# 可视化使用 train.py --visualize；引擎兼容检查可设 HC_SIM_BACKEND=isaac。

# 快速运行（conda activate isaac-lab，进入本仓库；默认 cuda:0、60 局）
#
# === G 系列（gap 动力学；序号 G0–G4）===
#   G0                hard 基线（gap；无教师）
#   G0-human-match    同 scratch + 人因奖励 + D match（无教师）
#   G1 / G2 / G3      可选：无教师 AR / G0 热启 no-oru / 仅人因
#   eval-G0 / eval-G0-human-match
# Usage 也支持 G0|G0-human-match
# gap 禁止挂在 E*/T0 上；见 docs/experiment_human.md §G
#
# === E 系列（legacy 动力学；旧协议）===
#   HC_HUMAN_SKILL_PROFILE=fast bash run_2026_journal_experiments.sh E5-human-match cuda:0
# W&B / 目录：E 人因 = {入口}-{profile}；G = G{n}-N10-S42 / hier_G{n}
#
# Journal entry — E0–E6 + G0–G4；见 docs/experiment_protocol.md / experiment_human.md.
# Usage:
#   ./run_2026_journal_experiments.sh G0|G0-human-match [cuda:0]
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
# E1–E6 微调默认 60；T0/T1 hard 与 G0-human-match* 默认 100
_HC_USER_MAX_TRAIN_EPISODES="${HC_MAX_TRAIN_EPISODES-}"
export HC_MAX_TRAIN_EPISODES="${HC_MAX_TRAIN_EPISODES:-60}"
export HC_MAX_HARD_EPISODES="${HC_MAX_HARD_EPISODES:-100}"
# G0-human-match*（含旧别名 G5-human-match*）未显式设预算时用 100，与 G0 hard 对齐
if [[ ( "${MODE}" == G0-human* || "${MODE}" == G5-human-match* ) && -z "${_HC_USER_MAX_TRAIN_EPISODES}" ]]; then
    export HC_MAX_TRAIN_EPISODES=100
fi

EVAL_STEPS="${HC_EVAL_STEPS:-}"

# --- G series helpers (gap dynamics; separate from legacy E*/T0) -----------------
g_normalize_profile() {
    local key="${1:-legacy}"
    key="$(echo "${key}" | tr '[:upper:]' '[:lower:]' | tr '_' '-')"
    case "${key}" in
        gap|strong-fast|contrast|hybrid|sharp) echo gap ;;
        strong|skill-strong-v1|strong-v1|v1-strong) echo strong ;;
        fast|skill-fast-v1|fast-v1|optimistic|short) echo fast ;;
        legacy|default|v0|original|"") echo legacy ;;
        *) echo "${key}" ;;
    esac
}

g_reject_gap_on_e_series() {
    local prof
    prof="$(g_normalize_profile "${HC_HUMAN_SKILL_PROFILE:-legacy}")"
    [[ "${prof}" == gap ]] || return 0
    case "${MODE}" in
        G*|eval-G*) return 0 ;;
        E*|eval-E*|T0|T1|T1R|T1RH|TEACHER|teacher|E2-collect)
            echo "错误: HC_HUMAN_SKILL_PROFILE=gap 只能用于 G 系列（G0–G4 / eval-G*）。" >&2
            echo "      当前 mode=${MODE} 属于 legacy E/T 协议；请改用对应 G* 入口，勿把 gap 挂在 E* 上。" >&2
            return 1
            ;;
    esac
    return 0
}

g_latest_run_dir() {
    # Newest among hier_${base} and hier_${base}-v* (mtime). Empty if none.
    local repo_root="$1" base="$2"
    local log_root="${repo_root}/logs/rl_games/HcFactory"
    local best="" best_t=-1 t d
    shopt -s nullglob
    for d in "${log_root}/hier_${base}" "${log_root}/hier_${base}-v"*; do
        [[ -d "${d}" ]] || continue
        t="$(stat -c %Y "${d}" 2>/dev/null || echo -1)"
        if (( t >= best_t )); then
            best_t=${t}
            best="${d}"
        fi
    done
    shopt -u nullglob
    echo "${best}"
}

g_teacher_dir() {
    local repo_root="$1"
    if [[ -n "${HC_G_TEACHER_DIR:-}" ]]; then
        echo "${HC_G_TEACHER_DIR}"
        return 0
    fi
    local latest
    latest="$(g_latest_run_dir "${repo_root}" G0)"
    if [[ -n "${latest}" ]]; then
        echo "${latest}"
    else
        echo "${repo_root}/logs/rl_games/HcFactory/hier_G0"
    fi
}

g_resolve_teacher_step() {
    # Prefer HC_G_LOAD_STEP / HC_LOAD_STEP; else latest state_encoder_step_*.pth under teacher dir.
    local load_dir="$1"
    local step="${HC_G_LOAD_STEP:-${HC_LOAD_STEP:-}}"
    if [[ -n "${step}" ]]; then
        echo "${step}"
        return 0
    fi
    local latest
    latest="$(ls -1 "${load_dir}/nn"/state_encoder_step_*.pth 2>/dev/null \
        | sed -n 's/.*_step_\([0-9]*\)\.pth/\1/p' | sort -n | tail -1 || true)"
    if [[ -z "${latest}" ]]; then
        echo "错误: 无法解析 G 教师 step；请先跑 G0，或设 HC_G_LOAD_STEP=..." >&2
        return 1
    fi
    echo "${latest}"
}

g_wandb_train_name() {
    # args: short_id  → {id}-N10-S42
    echo "${1}-N10-S42"
}


g_alloc_run_id() {
    # Unused run id under logs/rl_games/HcFactory/hier_*.
    # Bare name if free; else ${base}-v1, -v2, ... Explicit HC_RUN_TAG / HC_HUMAN_RUN_TAG pins suffix.
    local base="$1"
    local repo_root="${2:?g_alloc_run_id needs repo_root}"
    local log_root="${repo_root}/logs/rl_games/HcFactory"
    local tag="${HC_RUN_TAG:-${HC_HUMAN_RUN_TAG:-}}"
    local n
    mkdir -p "${log_root}"
    if [[ -n "${tag}" ]]; then
        echo "${base}-${tag}"
        return 0
    fi
    if [[ ! -e "${log_root}/hier_${base}" ]]; then
        echo "${base}"
        return 0
    fi
    n=1
    while [[ -e "${log_root}/hier_${base}-v${n}" ]]; do
        n=$((n + 1))
    done
    echo "${base}-v${n}"
}




g_rainbow_per() {
    # G series: enable PER (Rainbow). E/T stay off for protocol comparability.
    local mode="${1:-${MODE:-}}"
    case "${mode}" in
        G*|eval-G*|G-hard*|G5-*) echo true ;;
        *) echo false ;;
    esac
}

g_rainbow_dueling() {
    local mode="${1:-${MODE:-}}"
    case "${mode}" in
        G*|eval-G*|G-hard*|G5-*) echo true ;;
        *) echo false ;;
    esac
}

g_horizon_anchor() {
    # G series: actual T=35000 under N10 (t_max_anchor 56000 × 10/16).
    # E/T protocol stays T=40000 (anchor 64000). Override: HC_G_T_MAX_ANCHOR / HC_G_MAX_EPISODIC_STEPS.
    local mode="${1:-${MODE:-}}"
    case "${mode}" in
        G*|eval-G*|G-hard*|G5-*)
            echo "${HC_G_T_MAX_ANCHOR:-56000}"
            ;;
        *)
            echo 64000
            ;;
    esac
}

g_horizon_steps() {
    local mode="${1:-${MODE:-}}"
    case "${mode}" in
        G*|eval-G*|G-hard*|G5-*)
            echo "${HC_G_MAX_EPISODIC_STEPS:-35000}"
            ;;
        *)
            echo 40000
            ;;
    esac
}

g_source_wandb_env() {
    local local_env="${HC_WANDB_LOCAL_ENV:-.wandb_local.env}"
    if [[ -f "${local_env}" ]]; then
        set -a
        # shellcheck disable=SC1090
        source "${local_env}"
        set +a
    fi
    if [[ -n "${HC_WANDB_API_KEY:-}" ]]; then
        export WANDB_API_KEY="${HC_WANDB_API_KEY}"
    fi
    export WANDB_ENTITY="${HC_WANDB_ENTITY:-${WANDB_ENTITY:-rl-driving}}"
    export WANDB_MODE="${HC_WANDB_MODE:-${WANDB_MODE:-online}}"
}

g_check_device() {
    if [[ ! "${DEVICE}" =~ ^cuda:[0-9]+$ && "${DEVICE}" != cpu ]]; then
        echo "错误: 设备需为 cuda:N 或 cpu" >&2
        return 1
    fi
}

usage() {
    cat <<EOF
用法: $0 <mode> [cuda:N]

G 系列（gap 动力学；序号 G0–G4，勿与 E* 混用）:
  G0                 hard 基线（gap；无教师；默认 ${HC_MAX_HARD_EPISODES:-100} ep）
  G0-human-match     同 scratch + 人因奖励 + D match（无教师；默认 100 ep）
  G0-human-match-c   同上 + C match 汇总（无教师；默认 100 ep）
  G1                 无教师热启：AR + 低 lr（可选）
  G2                 G0 热启 + 教师探索 + AR（可选对照）
  G3                 G2 + 仅人因奖励（可选）
  eval-G0|eval-G0-human-match|eval-G0-human-match-c|eval-G1|eval-G2|eval-G3  协议评测

训练（E 系列 / legacy 动力学）:
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
  E5-human-pair  新 D-human 配对残差网络＋人因奖励（旧教师热启动）
  E5-pair        新网络、不加人因奖励（网络单独消融）
  E5-human-pair-c      D pair＋C 任务人员汇总残差
  E5-human-pair-aux    D pair＋实际完成耗时辅助学习
  E5-human-pair-c-aux  两项同时开启（先做单项实验）
  E5-human-match       D 双塔 match 打分（工时对齐特征）＋人因奖励 ★网络改进
  E5-human-match-c     D match＋C match 汇总
  eval-E5-human-pair-c|eval-E5-human-pair-aux|eval-E5-human-pair-c-aux 对应协议评测
  eval-E5-human-match|eval-E5-human-match-c  match 网络协议评测
  eval-E5-human-pair | eval-E5-pair  新网络协议评测；默认对应训练目录 / step300000
  E5-human   E5-no-oru＋人因奖励；HC_HUMAN_RUN_TAG 命名；HC_HUMAN_REWARD=false 消融
  eval-E5-human  同协议 43–52×1；默认对应训练目录 / step300000；支持 --dry-run
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
          默认 HC_EVAL_SEED_CHUNK=2；按 episodes.jsonl 实际行数推进 offset，缺局即停
  eval-E5 [cuda:N] [--dry-run]
          评测从工位同步过来的 E5 训练最优 ckpt（step 750000）
  eval-desk [cuda:N] [--dry-run]
          评测工位 E*：E4 / E5-no-oru / E5 / E3.5 / E3-no-oru / E1.5 / E3
  eval-desk-rev [cuda:N] [--dry-run]
          同上倒序（本机与 5090 对开）
  eval-E5-no-oru-near [cuda:N] [--dry-run]
          评 E5-no-oru 峰值附近剩余存盘：340k–380k（跳过已满评的 335k）
  eval-E5-no-oru-far [cuda:N] [--dry-run]
          评峰值窗外另外 10 个训练最优存盘（makespan 次优档，供 5090）

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
        --wandb_name E0-N10-S42-step1290000-eval
        --ftg_thresh_phy 0.95
        --seed 42
        agent.params.config.t_max_anchor="$(g_horizon_anchor)"
        agent.params.config.max_episodic_steps="$(g_horizon_steps)"
        agent.params.config.parallel_producing_limit=10
        agent.params.config.c_forbid_none_mode=always
        agent.params.config.curriculum=false
        agent.params.config.explore=false
        agent.params.config.explore_catalog=false
        agent.params.config.catalog_collect=false
        agent.params.config.oru=false
        agent.params.config.prioritized_replay="$(g_rainbow_per)"
        agent.params.config.dueling_dqn="$(g_rainbow_dueling)"
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
        --wandb_name E1-N10-S42
        --algo_variant E1
        --ftg_thresh_phy 0.95
        agent.params.config.t_max_anchor="$(g_horizon_anchor)"
        agent.params.config.max_episodic_steps="$(g_horizon_steps)"
        agent.params.config.parallel_producing_limit=10
        agent.params.config.c_forbid_none_mode=always
        agent.params.config.curriculum=false
        agent.params.config.explore=false
        agent.params.config.explore_catalog=false
        agent.params.config.catalog_collect=false
        agent.params.config.oru=false
        agent.params.config.prioritized_replay="$(g_rainbow_per)"
        agent.params.config.dueling_dqn="$(g_rainbow_dueling)"
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
    echo "[E1] max_sim_episodes=${HC_MAX_TRAIN_EPISODES}; load_dir=${load_dir}; project=HcFactory_TPA; wandb=E1-N10-S42"
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
        --wandb_name E1.5-N10-S42
        --algo_variant E1.5
        --ftg_thresh_phy 0.95
        agent.params.config.t_max_anchor="$(g_horizon_anchor)"
        agent.params.config.max_episodic_steps="$(g_horizon_steps)"
        agent.params.config.parallel_producing_limit=10
        agent.params.config.c_forbid_none_mode=always
        agent.params.config.curriculum=false
        agent.params.config.explore=false
        agent.params.config.explore_catalog=false
        agent.params.config.catalog_collect=false
        agent.params.config.oru=false
        agent.params.config.prioritized_replay="$(g_rainbow_per)"
        agent.params.config.dueling_dqn="$(g_rainbow_dueling)"
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
    echo "[E1.5] max_sim_episodes=${HC_MAX_TRAIN_EPISODES}; load_dir=${load_dir}; wandb=E1.5-N10-S42"
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
        --wandb_name E2-N10-S42
        --algo_variant E2
        --ftg_thresh_phy 0.95
        agent.params.config.t_max_anchor="$(g_horizon_anchor)"
        agent.params.config.max_episodic_steps="$(g_horizon_steps)"
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
        agent.params.config.prioritized_replay="$(g_rainbow_per)"
        agent.params.config.dueling_dqn="$(g_rainbow_dueling)"
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
    echo "[E2] max_sim_episodes=${HC_MAX_TRAIN_EPISODES}; load_dir=${load_dir}; project=HcFactory_TPA; wandb=E2-N10-S42"
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
        --wandb_name E2.5-N10-S42
        --algo_variant E2.5
        --ftg_thresh_phy 0.95
        agent.params.config.t_max_anchor="$(g_horizon_anchor)"
        agent.params.config.max_episodic_steps="$(g_horizon_steps)"
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
        agent.params.config.prioritized_replay="$(g_rainbow_per)"
        agent.params.config.dueling_dqn="$(g_rainbow_dueling)"
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
    echo "[E2.5] max_sim_episodes=${HC_MAX_TRAIN_EPISODES}; load_dir=${load_dir}; wandb=E2.5-N10-S42"
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
        --wandb_name E3-N10-S42
        --algo_variant E3
        --ftg_thresh_phy 0.95
        agent.params.config.t_max_anchor="$(g_horizon_anchor)"
        agent.params.config.max_episodic_steps="$(g_horizon_steps)"
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
        agent.params.config.prioritized_replay="$(g_rainbow_per)"
        agent.params.config.dueling_dqn="$(g_rainbow_dueling)"
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
    echo "[E3] max_sim_episodes=${HC_MAX_TRAIN_EPISODES}; load_dir=${load_dir}; project=HcFactory_TPA; wandb=E3-N10-S42"
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
        --wandb_name E3.5-N10-S42
        --algo_variant E3.5
        --ftg_thresh_phy 0.95
        agent.params.config.t_max_anchor="$(g_horizon_anchor)"
        agent.params.config.max_episodic_steps="$(g_horizon_steps)"
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
        agent.params.config.prioritized_replay="$(g_rainbow_per)"
        agent.params.config.dueling_dqn="$(g_rainbow_dueling)"
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
    echo "[E3.5] max_sim_episodes=${HC_MAX_TRAIN_EPISODES}; load_dir=${load_dir}; wandb=E3.5-N10-S42"
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
        --wandb_name E4-N10-S42
        --algo_variant E4
        --ftg_thresh_phy 0.95
        agent.params.config.t_max_anchor="$(g_horizon_anchor)"
        agent.params.config.max_episodic_steps="$(g_horizon_steps)"
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
        agent.params.config.prioritized_replay="$(g_rainbow_per)"
        agent.params.config.dueling_dqn="$(g_rainbow_dueling)"
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
    echo "[E4] max_sim_episodes=${HC_MAX_TRAIN_EPISODES}; load_dir=${load_dir}; project=HcFactory_TPA; wandb=E4-N10-S42"
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
        --wandb_name E3-no-oru-N10-S42
        --algo_variant E3-no-oru
        --ftg_thresh_phy 0.95
        agent.params.config.t_max_anchor="$(g_horizon_anchor)"
        agent.params.config.max_episodic_steps="$(g_horizon_steps)"
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
        agent.params.config.prioritized_replay="$(g_rainbow_per)"
        agent.params.config.dueling_dqn="$(g_rainbow_dueling)"
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
    echo "[E3-no-oru] max_sim_episodes=${HC_MAX_TRAIN_EPISODES}; load_dir=${load_dir}; wandb=E3-no-oru-N10-S42"
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
        --wandb_name E4-no-oru-N10-S42
        --algo_variant E4-no-oru
        --ftg_thresh_phy 0.95
        agent.params.config.t_max_anchor="$(g_horizon_anchor)"
        agent.params.config.max_episodic_steps="$(g_horizon_steps)"
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
        agent.params.config.prioritized_replay="$(g_rainbow_per)"
        agent.params.config.dueling_dqn="$(g_rainbow_dueling)"
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
    echo "[E4-no-oru] max_sim_episodes=${HC_MAX_TRAIN_EPISODES}; load_dir=${load_dir}; wandb=E4-no-oru-N10-S42"
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
        --wandb_name E5-N10-S42
        --algo_variant E5
        --ftg_thresh_phy 0.95
        agent.params.config.t_max_anchor="$(g_horizon_anchor)"
        agent.params.config.max_episodic_steps="$(g_horizon_steps)"
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
        agent.params.config.prioritized_replay="$(g_rainbow_per)"
        agent.params.config.dueling_dqn="$(g_rainbow_dueling)"
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
    echo "[E5] max_sim_episodes=${HC_MAX_TRAIN_EPISODES}; load_dir=${load_dir}; wandb=E5-N10-S42"
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
        --wandb_name E6-N10-S42
        --algo_variant E6
        --ftg_thresh_phy 0.95
        agent.params.config.t_max_anchor="$(g_horizon_anchor)"
        agent.params.config.max_episodic_steps="$(g_horizon_steps)"
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
        agent.params.config.prioritized_replay="$(g_rainbow_per)"
        agent.params.config.dueling_dqn="$(g_rainbow_dueling)"
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
    echo "[E6] max_sim_episodes=${HC_MAX_TRAIN_EPISODES}; load_dir=${load_dir}; wandb=E6-N10-S42"
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
        --wandb_name E5-no-oru-N10-S42
        --algo_variant E5-no-oru
        --ftg_thresh_phy 0.95
        agent.params.config.t_max_anchor="$(g_horizon_anchor)"
        agent.params.config.max_episodic_steps="$(g_horizon_steps)"
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
        agent.params.config.prioritized_replay="$(g_rainbow_per)"
        agent.params.config.dueling_dqn="$(g_rainbow_dueling)"
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
    echo "[E5-no-oru] max_sim_episodes=${HC_MAX_TRAIN_EPISODES}; load_dir=${load_dir}; wandb=E5-no-oru-N10-S42"
    if [[ "${dry_run}" == --dry-run ]]; then
        printf '%q ' "${cmd[@]}"
        printf '\n'
        return 0
    fi
    "${cmd[@]}"
}

# Human 系列：目录 / W&B 默认为 ${variant}-${skill_profile}；HC_HUMAN_RUN_TAG 仅可选防撞后缀。
human_run_id() {
    # args: variant skill_profile [optional_tag]
    local vid="$1" prof="$2" extra="${3:-}"
    if [[ -n "${extra}" ]]; then
        echo "${vid}-${prof}-${extra}"
    else
        echo "${vid}-${prof}"
    fi
}

# Resolve train dir: new name first, then HC_HUMAN_RUN_TAG old style, then known legacy aliases.
human_resolve_train_dir() {
    local root="$1" variant="$2" skill_profile="$3" tag="${4:-}"
    local base="${root}/logs/rl_games/HcFactory"
    local -a candidates=()
    candidates+=("${base}/hier_$(human_run_id "${variant}" "${skill_profile}" "${tag}")")
    if [[ -n "${tag}" ]]; then
        candidates+=("${base}/hier_${variant}-${tag}")
    fi
    case "${variant}" in
        E5-human)
            candidates+=(
                "${base}/hier_E5-human-human-logic-v1"
                "${base}/hier_E5-human-formal-v4"
                "${base}/hier_E5-human-formal-v3"
                "${base}/hier_E5-human-formal-v2"
                "${base}/hier_E5-human-formal-v1"
                "${base}/hier_E5-human-home-v1"
            )
            ;;
        E5-human-match)
            candidates+=("${base}/hier_E5-human-match-match-v1" "${base}/hier_E5-human-match-match-strong-v1")
            ;;
        E5-human-match-c)
            candidates+=("${base}/hier_E5-human-match-c-match-c-logic-v1" "${base}/hier_E5-human-match-c-match-c-v1")
            ;;
        E5-human-pair)
            candidates+=("${base}/hier_E5-human-pair-pair-formal-v1" "${base}/hier_E5-human-pair-pair-formal-v2")
            ;;
        E5-human-pair-c)
            candidates+=("${base}/hier_E5-human-pair-c-c-v1" "${base}/hier_E5-human-pair-c-c-logic-v1")
            ;;
        E5-human-pair-aux)
            candidates+=("${base}/hier_E5-human-pair-aux-aux-logic-v1" "${base}/hier_E5-human-pair-aux-aux-v1")
            ;;
        E5-human-pair-c-aux)
            candidates+=("${base}/hier_E5-human-pair-c-aux-c-aux-logic-v1" "${base}/hier_E5-human-pair-c-aux-c-aux-v1")
            ;;
        E5-pair)
            candidates+=("${base}/hier_E5-pair-pair-formal-v1")
            ;;
    esac
    local d
    for d in "${candidates[@]}"; do
        if [[ -d "${d}/nn" ]]; then
            echo "${d}"
            return 0
        fi
    done
    echo "${candidates[0]}"
}

run_e5_human_train() {
    # P0: same backbone as E5-no-oru; separate output/name, bounded reward only.
    # G0-human*: gap + match/reward, NO teacher (same scratch spirit as G0).
    # G3/G4: gap + G0 warmstart + teacher_explore + AR.
    local repo_root load_dir dry_run="${3:-}" teacher_step=1290000 wandb_name
    local tag="${HC_HUMAN_RUN_TAG:-}"
    local enabled="${HC_HUMAN_REWARD:-true}"
    local g_series=false
    local g0_scratch=false
    local skill_profile="${HC_HUMAN_SKILL_PROFILE:-legacy}"
    if [[ "${MODE}" == G0-human* ]]; then
        g_series=true
        g0_scratch=true
        skill_profile=gap
    elif [[ "${MODE}" == G3 || "${MODE}" == G4 || "${MODE}" == G4-c ]]; then
        g_series=true
        skill_profile=gap
    fi
    case "${skill_profile}" in
        strong|skill-strong-v1|strong-v1|v1-strong) skill_profile=strong ;;
        fast|skill-fast-v1|fast-v1|optimistic|short) skill_profile=fast ;;
        gap|strong-fast|contrast|hybrid|sharp) skill_profile=gap ;;
        legacy|default|v0|original|"") skill_profile=legacy ;;
        *) echo "Invalid HC_HUMAN_SKILL_PROFILE=${HC_HUMAN_SKILL_PROFILE} (use legacy|strong|fast|gap)" >&2; return 1 ;;
    esac
    if [[ "${g_series}" != true && "${skill_profile}" == gap ]]; then
        echo "错误: gap 请用 G0-human-match / G3 / G4，不要设 HC_HUMAN_SKILL_PROFILE=gap 跑 E5-human*" >&2
        return 1
    fi
    export HC_HUMAN_SKILL_PROFILE="${skill_profile}"
    if [[ -n "${tag}" && ! "${tag}" =~ ^[A-Za-z0-9_-]+$ ]]; then
        echo "Invalid HC_HUMAN_RUN_TAG" >&2; return 1
    fi
    [[ "${enabled}" == true || "${enabled}" == false ]] || { echo "HC_HUMAN_REWARD=true|false" >&2; return 1; }
    local variant=E5-human
    [[ "${enabled}" == true ]] || variant=E5-human-off
    local pair_head=false task_pair=false duration_aux=false match_head=false task_match=false
    if [[ "${MODE}" == E5-human-pair* || "${MODE}" == E5-pair ]]; then
        pair_head=true
        [[ "${MODE}" != E5-pair ]] || enabled=false
        variant=E5-human-pair
        [[ "${enabled}" == true ]] || variant=E5-pair
    fi
    if [[ "${MODE}" == E5-human-match* || "${MODE}" == G0-human-match* || "${MODE}" == G4 || "${MODE}" == G4-c ]]; then
        match_head=true
        variant=${MODE}
    fi
    case "${MODE}" in
        E5-human-pair-c) task_pair=true; variant=${MODE} ;;
        E5-human-pair-aux) duration_aux=true; variant=${MODE} ;;
        E5-human-pair-c-aux) task_pair=true; duration_aux=true; variant=${MODE} ;;
        E5-human-match-c|G0-human-match-c|G4-c) task_match=true; variant=${MODE} ;;
        G0-human-match) variant=G0-human-match ;;
        G3) variant=G3 ;;
        G4) variant=G4 ;;
    esac
    if [[ "${MODE}" == E5-human-pair-* && "${enabled}" != true ]]; then
        echo "新扩展入口固定人因奖励开启；奖励消融请用 E5-pair" >&2; return 1
    fi
    if [[ "${MODE}" == *human-match* && "${enabled}" != true ]]; then
        echo "match 入口固定人因奖励开启" >&2; return 1
    fi
    repo_root=$(cd -- "$(dirname -- "$0")" && pwd)
    cd "${repo_root}"
    local run_id
    if [[ "${g_series}" == true ]]; then
        if [[ -n "${tag}" ]]; then
            run_id="${variant}-${tag}"
        else
            run_id="$(g_alloc_run_id "${variant}" "${repo_root}")"
        fi
        wandb_name="$(g_wandb_train_name "${run_id}")"
    else
        if [[ -n "${tag}" ]]; then
            run_id="$(human_run_id "${variant}" "${skill_profile}" "${tag}")"
        else
            run_id="$(g_alloc_run_id "$(human_run_id "${variant}" "${skill_profile}")" "${repo_root}")"
        fi
        wandb_name="${run_id}"
    fi
    if [[ -z "${tag}" ]]; then
        local _base_expect
        if [[ "${g_series}" == true ]]; then
            _base_expect="${variant}"
        else
            _base_expect="$(human_run_id "${variant}" "${skill_profile}")"
        fi
        if [[ "${run_id}" != "${_base_expect}" ]]; then
            echo "[${variant}] 目录已占用 → 自动使用 run_id=${run_id}"
        fi
    fi
    load_dir=""
    teacher_step=""
    if [[ "${g0_scratch}" == true ]]; then
        : # no teacher / no warmstart
    elif [[ "${g_series}" == true ]]; then
        # Never fall back to HC_HUMAN_TEACHER_DIR (legacy E/T0).
        load_dir="${HC_G_TEACHER_DIR:-$(g_teacher_dir "${repo_root}")}"
        if [[ -d "${load_dir}/nn" ]]; then
            teacher_step="$(g_resolve_teacher_step "${load_dir}")" || return 1
        elif [[ "${dry_run}" == --dry-run ]]; then
            teacher_step="${HC_G_LOAD_STEP:-${HC_LOAD_STEP:-0}}"
        else
            echo "错误: 缺少 G 教师目录: ${load_dir}（请先跑 G0；可用 HC_G_TEACHER_DIR 覆盖）" >&2
            return 1
        fi
        if [[ "${load_dir}" == *hier_2026-08-27_23-17-41* ]]; then
            echo "错误: G 系列禁止使用 legacy T0 教师目录: ${load_dir}" >&2
            return 1
        fi
    else
        load_dir="${HC_HUMAN_TEACHER_DIR:-${repo_root}/logs/rl_games/HcFactory/hier_2026-08-27_23-17-41}"
        teacher_step=1290000
    fi
    local out="${repo_root}/logs/rl_games/HcFactory/hier_${run_id}"
    if [[ "${dry_run}" != --dry-run && -e "${out}" ]]; then
        echo "拒绝覆盖现有训练目录: ${out}; 请设置 HC_HUMAN_RUN_TAG=... 作为防撞后缀" >&2
        return 1
    fi
    if [[ ! "${DEVICE}" =~ ^cuda:[0-9]+$ && "${DEVICE}" != cpu ]]; then
        echo "错误: 设备需为 cuda:N 或 cpu" >&2
        return 1
    fi
    if [[ -n "${dry_run}" && "${dry_run}" != --dry-run ]] || (( $# > 3 )); then
        echo "用法: $0 E5-human|G0-human-match [cuda:N] [--dry-run]" >&2
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
    if [[ "${g0_scratch}" != true && "${dry_run}" != --dry-run ]]; then
        local head
        for head in state_encoder agent_A agent_B agent_C agent_D_human agent_D_robot; do
            [[ -s "${load_dir}/nn/${head}_step_${teacher_step}.pth" ]] || {
                echo "缺少教师模型: ${load_dir}/nn/${head}_step_${teacher_step}.pth" >&2; return 1;
            }
        done
    fi
    export HC_WARMSTART=""
    unset HC_WANDB_RUN_ID WANDB_RUN_ID HC_WANDB_RESUME WANDB_RESUME
    local max_ep="${HC_MAX_TRAIN_EPISODES}"
    # G0-human*：与 G0 hard 同预算口径（默认已是 100）
    if [[ "${g0_scratch}" == true ]]; then
        max_ep="${HC_MAX_TRAIN_EPISODES:-${HC_MAX_HARD_EPISODES}}"
    fi
    local -a cmd=(
        python train.py --task HRTPaHC-v1 --algo hier
        --device "${DEVICE}" --num_envs 1 --headless --seed 42
        --train_n_products 10 --max_parallel_cd_dispatch 10
        --max_sim_episodes "${max_ep}"
    )
    if [[ "${g0_scratch}" == true ]]; then
        # Scratch under gap: no load / no teacher_explore / no AR (fair vs G0).
        cmd+=(
            --wandb_activate --wandb_project HcFactory_TPA
            --wandb_name "${wandb_name}"
            --algo_variant "${variant}"
            --ftg_thresh_phy 0.95
            "+agent.params.config.full_experiment_name=${run_id}"
            "agent.params.config.human_pair_head=${pair_head}"
            "agent.params.config.task_pair_head=${task_pair}"
            "agent.params.config.human_match_head=${match_head}"
            "agent.params.config.task_match_head=${task_match}"
            "agent.params.config.human_duration_aux=${duration_aux}"
            "agent.params.config.duration_aux_weight=${HC_DURATION_AUX_WEIGHT:-0.05}"
            "agent.params.config.duration_aux_scale=${HC_DURATION_AUX_SCALE:-1000.0}"
            "agent.params.config.human_aware_reward=${enabled}"
            agent.params.config.human_reward_metrics=true
            "agent.params.config.human_mismatch_coef=${HC_HUMAN_MISMATCH_COEF:-0.05}"
            "agent.params.config.human_overwork_coef=${HC_HUMAN_OVERWORK_COEF:-0.01}"
            "agent.params.config.human_recovery_coef=${HC_HUMAN_RECOVERY_COEF:-0.0}"
            "agent.params.config.human_fatigue_threshold=${HC_HUMAN_FATIGUE_THRESHOLD:-0.8}"
            "agent.params.config.human_shaping_cap=${HC_HUMAN_SHAPING_CAP:-0.04}"
            agent.params.config.t_max_anchor="$(g_horizon_anchor)"
            agent.params.config.max_episodic_steps="$(g_horizon_steps)"
            agent.params.config.parallel_producing_limit=10
            agent.params.config.c_forbid_none_mode=always
            agent.params.config.curriculum=false
            agent.params.config.explore=false
            agent.params.config.explore_catalog=false
            agent.params.config.catalog_collect=false
            agent.params.config.oru=false
            agent.params.config.teacher_explore=false
            agent.params.config.autoregressive=false
            agent.params.config.prioritized_replay="$(g_rainbow_per)"
            agent.params.config.dueling_dqn="$(g_rainbow_dueling)"
            agent.params.config.noisy_net=false
            agent.params.config.hierarchical_credit=false
            agent.params.config.b_score_rl=false
            agent.params.config.env_rule_based_exploration=false
            'agent.params.config.warmstart=""'
            'agent.params.config.load_name=""'
        )
    else
        cmd+=(
            --load_dir "${load_dir}" --load_step "${teacher_step}"
            --teacher_explore
            --autoregressive
            --wandb_activate --wandb_project HcFactory_TPA
            --wandb_name "${wandb_name}"
            --algo_variant "${variant}"
            --ftg_thresh_phy 0.95
            "+agent.params.config.full_experiment_name=${run_id}"
            "agent.params.config.human_pair_head=${pair_head}"
            "agent.params.config.task_pair_head=${task_pair}"
            "agent.params.config.human_match_head=${match_head}"
            "agent.params.config.task_match_head=${task_match}"
            "agent.params.config.human_duration_aux=${duration_aux}"
            "agent.params.config.duration_aux_weight=${HC_DURATION_AUX_WEIGHT:-0.05}"
            "agent.params.config.duration_aux_scale=${HC_DURATION_AUX_SCALE:-1000.0}"
            "agent.params.config.human_aware_reward=${enabled}"
            agent.params.config.human_reward_metrics=true
            "agent.params.config.human_mismatch_coef=${HC_HUMAN_MISMATCH_COEF:-0.05}"
            "agent.params.config.human_overwork_coef=${HC_HUMAN_OVERWORK_COEF:-0.01}"
            "agent.params.config.human_recovery_coef=${HC_HUMAN_RECOVERY_COEF:-0.0}"
            "agent.params.config.human_fatigue_threshold=${HC_HUMAN_FATIGUE_THRESHOLD:-0.8}"
            "agent.params.config.human_shaping_cap=${HC_HUMAN_SHAPING_CAP:-0.04}"
            agent.params.config.t_max_anchor="$(g_horizon_anchor)"
            agent.params.config.max_episodic_steps="$(g_horizon_steps)"
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
            agent.params.config.prioritized_replay="$(g_rainbow_per)"
            agent.params.config.dueling_dqn="$(g_rainbow_dueling)"
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
    fi
    echo "[${variant}] skill_profile=${skill_profile}; run_id=${run_id}; seed=42; output=${out}"
    if [[ "${g0_scratch}" == true ]]; then
        echo "[${variant}] NO teacher (scratch like G0); pair=${pair_head} match=${match_head} task_match=${task_match} reward=${enabled}; max_ep=${max_ep}; wandb=${wandb_name}"
    else
        echo "[${variant}] teacher=${load_dir} step=${teacher_step}; pair=${pair_head} task_pair=${task_pair} match=${match_head} task_match=${task_match} duration_aux=${duration_aux} reward=${enabled}; wandb=${wandb_name}"
    fi
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
        --wandb_name E6-no-oru-N10-S42
        --algo_variant E6-no-oru
        --ftg_thresh_phy 0.95
        agent.params.config.t_max_anchor="$(g_horizon_anchor)"
        agent.params.config.max_episodic_steps="$(g_horizon_steps)"
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
        agent.params.config.prioritized_replay="$(g_rainbow_per)"
        agent.params.config.dueling_dqn="$(g_rainbow_dueling)"
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
    echo "[E6-no-oru] max_sim_episodes=${HC_MAX_TRAIN_EPISODES}; load_dir=${load_dir}; wandb=E6-no-oru-N10-S42"
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
        export HC_WANDB_NAME="${variant}-N10-S42-step${step}-eval"
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
        export HC_WANDB_NAME="${variant}-N10-S42-step${step}-eval"
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
    # E5-no-oru 峰值附近 10 ckpt 见 eval-E5-no-oru-near（不塞满 desk 面板）。
    local -a jobs=(
        "E4|hier_2026-09-12_10-20-18|645000"
        "E5-no-oru|hier_2026-09-18_21-14-57|360000"
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

# E5-no-oru near-peak sweep around best ep (Train/step≈358003 → 360k band).
# Skip 335000: already full 10/10 protocol eval (mean≈17220). Remaining 340k–380k.
e5_no_oru_near_jobs() {
    local dir=hier_2026-09-18_21-14-57
    local -a steps=(340000 345000 350000 355000 360000 365000 370000 375000 380000)
    local s
    local -a jobs=()
    for s in "${steps[@]}"; do
        jobs+=("E5-no-oru-${s}|${dir}|${s}")
    done
    printf '%s\n' "${jobs[@]}"
}

run_eval_e5_no_oru_near() {
    local dry_run="${3:-}"
    if [[ ! "${DEVICE}" =~ ^cuda:[0-9]+$ && "${DEVICE}" != cpu ]]; then
        echo "错误: 设备需为 cuda:N 或 cpu" >&2
        return 1
    fi
    if [[ -n "${dry_run}" && "${dry_run}" != --dry-run ]] || (( $# > 3 )); then
        echo "用法: $0 eval-E5-no-oru-near [cuda:N] [--dry-run]" >&2
        return 1
    fi
    local -a jobs=()
    mapfile -t jobs < <(e5_no_oru_near_jobs)
    if [[ "${dry_run}" == --dry-run ]]; then
        run_eval_job_panel eval-E5-no-oru-near "${jobs[@]}" --dry-run
    else
        run_eval_job_panel eval-E5-no-oru-near "${jobs[@]}"
    fi
}

# E5-no-oru far sweep: top-10 train-best saves OUTSIDE the 335k–380k peak window
# (≤-aligned to save_interval; files verified on desk). Prefer 5090 for this panel.
e5_no_oru_far_jobs() {
    local dir=hier_2026-09-18_21-14-57
    # ms@ep: 15216,15769,15877,16018,16100,16164,16252,16332,16390,16395
    local -a steps=(240000 495000 870000 205000 85000 990000 480000 1010000 595000 940000)
    local s
    local -a jobs=()
    for s in "${steps[@]}"; do
        jobs+=("E5-no-oru-${s}|${dir}|${s}")
    done
    printf '%s\n' "${jobs[@]}"
}

run_eval_e5_no_oru_far() {
    local dry_run="${3:-}"
    if [[ ! "${DEVICE}" =~ ^cuda:[0-9]+$ && "${DEVICE}" != cpu ]]; then
        echo "错误: 设备需为 cuda:N 或 cpu" >&2
        return 1
    fi
    if [[ -n "${dry_run}" && "${dry_run}" != --dry-run ]] || (( $# > 3 )); then
        echo "用法: $0 eval-E5-no-oru-far [cuda:N] [--dry-run]" >&2
        return 1
    fi
    local -a jobs=()
    mapfile -t jobs < <(e5_no_oru_far_jobs)
    if [[ "${dry_run}" == --dry-run ]]; then
        run_eval_job_panel eval-E5-no-oru-far "${jobs[@]}" --dry-run
    else
        run_eval_job_panel eval-E5-no-oru-far "${jobs[@]}"
    fi
}

run_eval_e5_human() {
    local dry_run="${3:-}" repo_root head
    repo_root=$(cd -- "$(dirname -- "$0")" && pwd)
    cd "${repo_root}"
    [[ -z "${dry_run}" || "${dry_run}" == --dry-run ]] || return 1
    local skill_profile="${HC_HUMAN_SKILL_PROFILE:-legacy}"
    local g_series=false
    local variant="${MODE#eval-}"
    if [[ "${variant}" == G0-human* || "${variant}" == G3 || "${variant}" == G4 || "${variant}" == G4-c ]]; then
        g_series=true
        skill_profile=gap
    fi
    case "${skill_profile}" in
        strong|skill-strong-v1|strong-v1|v1-strong) skill_profile=strong ;;
        fast|skill-fast-v1|fast-v1|optimistic|short) skill_profile=fast ;;
        gap|strong-fast|contrast|hybrid|sharp) skill_profile=gap ;;
        legacy|default|v0|original|"") skill_profile=legacy ;;
        *) echo "Invalid HC_HUMAN_SKILL_PROFILE=${HC_HUMAN_SKILL_PROFILE} (use legacy|strong|fast|gap)" >&2; return 1 ;;
    esac
    if [[ "${g_series}" != true && "${skill_profile}" == gap ]]; then
        echo "错误: gap 评测请用 eval-G0-human-match / eval-G3 / eval-G4" >&2
        return 1
    fi
    local tag="${HC_HUMAN_RUN_TAG:-}"
    if [[ -n "${tag}" && ! "${tag}" =~ ^[A-Za-z0-9_-]+$ ]]; then
        echo "Invalid HC_HUMAN_RUN_TAG" >&2; return 1
    fi
    local run_id
    if [[ "${g_series}" == true ]]; then
        if [[ -n "${tag}" ]]; then
            run_id="${variant}-${tag}"
        else
            run_id="${variant}"
        fi
    else
        run_id="$(human_run_id "${variant}" "${skill_profile}" "${tag}")"
    fi
    export HC_HUMAN_SKILL_PROFILE="${skill_profile}"
    local load_dir
    if [[ -n "${HC_LOAD_DIR:-}" ]]; then
        load_dir="${HC_LOAD_DIR}"
    elif [[ "${g_series}" == true ]]; then
        load_dir="${repo_root}/logs/rl_games/HcFactory/hier_${run_id}"
    else
        load_dir="$(human_resolve_train_dir "${repo_root}" "${variant}" "${skill_profile}" "${tag}")"
    fi
    local step="${HC_LOAD_STEP:-300000}"
    if [[ "${g_series}" == true && -z "${HC_LOAD_STEP:-}" && -d "${load_dir}/nn" ]]; then
        step="$(g_resolve_teacher_step "${load_dir}")" || step=300000
    fi
    [[ "${step}" =~ ^[0-9]+$ ]] || { echo "HC_LOAD_STEP 必须是整数" >&2; return 1; }
    local stamp="$(date +%Y%m%d_%H%M%S)_$$"
    export HC_HUMAN_PAIR_EVAL=false HC_TASK_PAIR_EVAL=false HC_DURATION_AUX_EVAL=false
    export HC_HUMAN_MATCH_EVAL=false HC_TASK_MATCH_EVAL=false
    if [[ "${MODE}" == eval-E5-human-pair* || "${MODE}" == eval-E5-pair ]]; then
        export HC_HUMAN_PAIR_EVAL=true
    fi
    if [[ "${MODE}" == eval-E5-human-match* || "${MODE}" == eval-G0-human-match* || "${MODE}" == eval-G4 || "${MODE}" == eval-G4-c ]]; then
        export HC_HUMAN_MATCH_EVAL=true
    fi
    case "${MODE}" in
        eval-E5-human-pair-c) export HC_TASK_PAIR_EVAL=true ;;
        eval-E5-human-pair-aux) export HC_DURATION_AUX_EVAL=true ;;
        eval-E5-human-pair-c-aux) export HC_TASK_PAIR_EVAL=true HC_DURATION_AUX_EVAL=true ;;
        eval-E5-human-match-c|eval-G0-human-match-c|eval-G4-c) export HC_TASK_MATCH_EVAL=true ;;
    esac
    export HC_LOAD_DIR="${load_dir}" HC_LOAD_STEP="${step}"
    export HC_TEST_SEEDS=43,44,45,46,47,48,49,50,51,52 HC_TEST_TIMES=1
    export HC_TRAIN_N_PRODUCTS=10 HC_T_MAX_ANCHOR=64000 HC_MULTI_K=10
    export HC_HUMAN_EVAL=1 HC_EVAL_VARIANT="${variant}"
    if [[ "${g_series}" == true ]]; then
        export HC_WANDB_NAME="${run_id}-N10-S42-step${step}-eval"
    else
        export HC_WANDB_NAME="${run_id}-step${step}-eval"
    fi
    export HC_WANDB_TEST_PROJECT=HcFactory_TPA_Eval HC_WARMSTART=""
    export HC_EVAL_OUTPUT_DIR="${load_dir}/eval_step${step}_${stamp}"
    export HC_EVAL_KEEP_PRIOR=0
    unset HC_WANDB_RUN_ID WANDB_RUN_ID HC_WANDB_RESUME WANDB_RESUME
    echo "[eval-${variant}] N10 K10 T40000 epsilon=0 seeds=43..52 x1; shaping OFF; pair=${HC_HUMAN_PAIR_EVAL} match=${HC_HUMAN_MATCH_EVAL} task_match=${HC_TASK_MATCH_EVAL}"
    echo "[eval-${variant}] load=${load_dir} step=${step}; wandb=${HC_WANDB_NAME}; out=${HC_EVAL_OUTPUT_DIR}"
    if [[ "${dry_run}" == --dry-run ]]; then
        echo "bash batch_train.sh 29 ${DEVICE} (fixed protocol above, existing chunked eval)"
        return 0
    fi
    for head in state_encoder agent_A agent_B agent_C agent_D_human agent_D_robot; do
        [[ -s "${load_dir}/nn/${head}_step_${step}.pth" ]] || {
            echo "缺少权重: ${load_dir}/nn/${head}_step_${step}.pth" >&2; return 1;
        }
    done
    bash batch_train.sh 29 "${DEVICE}"
}

run_g0_train() {
    # Gap dynamics hard train → new teacher (legacy T0 role).
    local repo_root dry_run="${3:-}" run_id=G0 out wandb_name
    repo_root=$(cd -- "$(dirname -- "$0")" && pwd)
    cd "${repo_root}"
    g_check_device || return 1
    if [[ -n "${dry_run}" && "${dry_run}" != --dry-run ]] || (( $# > 3 )); then
        echo "用法: $0 G0 [cuda:N] [--dry-run]" >&2
        return 1
    fi
    export HC_HUMAN_SKILL_PROFILE=gap
    run_id="$(g_alloc_run_id G0 "${repo_root}")"
    out="${repo_root}/logs/rl_games/HcFactory/hier_${run_id}"
    wandb_name="$(g_wandb_train_name "${run_id}")"
    if [[ "${dry_run}" != --dry-run && -e "${out}" ]]; then
        echo "拒绝覆盖现有训练目录: ${out}；请改 HC_RUN_TAG / HC_HUMAN_RUN_TAG 或移走目录" >&2
        return 1
    fi
    if [[ "${run_id}" != G0 ]]; then
        echo "[G0] 目录已占用 → 自动使用 run_id=${run_id}"
    fi
    if [[ "${dry_run}" != --dry-run ]]; then
        g_source_wandb_env
    else
        export WANDB_ENTITY="${HC_WANDB_ENTITY:-${WANDB_ENTITY:-rl-driving}}"
        export WANDB_MODE="${HC_WANDB_MODE:-${WANDB_MODE:-online}}"
    fi
    export HC_WARMSTART=""
    unset HC_WANDB_RUN_ID WANDB_RUN_ID HC_WANDB_RESUME WANDB_RESUME
    local -a cmd=(
        python train.py --task HRTPaHC-v1 --algo hier
        --device "${DEVICE}" --num_envs 1 --headless --seed 42
        --train_n_products 10 --max_parallel_cd_dispatch 10
        --max_sim_episodes "${HC_MAX_HARD_EPISODES}"
        --wandb_activate --wandb_project HcFactory_TPA
        --wandb_name "${wandb_name}"
        --algo_variant T0
        --ftg_thresh_phy 0.95
        "+agent.params.config.full_experiment_name=${run_id}"
        agent.params.config.t_max_anchor="$(g_horizon_anchor)"
        agent.params.config.max_episodic_steps="$(g_horizon_steps)"
        agent.params.config.parallel_producing_limit=10
        agent.params.config.c_forbid_none_mode=always
        agent.params.config.curriculum=false
        agent.params.config.explore=false
        agent.params.config.explore_catalog=false
        agent.params.config.catalog_collect=false
        agent.params.config.oru=false
        agent.params.config.teacher_explore=false
        agent.params.config.autoregressive=false
        agent.params.config.prioritized_replay="$(g_rainbow_per)"
        agent.params.config.dueling_dqn="$(g_rainbow_dueling)"
        agent.params.config.noisy_net=false
        agent.params.config.hierarchical_credit=false
        agent.params.config.b_score_rl=false
        agent.params.config.env_rule_based_exploration=false
        'agent.params.config.warmstart=""'
        'agent.params.config.load_name=""'
    )
    echo "[G0] skill_profile=gap; hard train (new teacher); max_ep=${HC_MAX_HARD_EPISODES}; out=${out}"
    echo "[G0] wandb=${wandb_name}; scratch under gap（无教师）；可与 G0-human-match 并行"
    if [[ "${dry_run}" == --dry-run ]]; then
        printf '%q ' "${cmd[@]}"
        printf '\n'
        return 0
    fi
    "${cmd[@]}"
}

run_g1_train() {
    # No teacher warmstart; E5-no-oru-like AR + low lr, 60ep under gap.
    local repo_root dry_run="${3:-}" run_id=G1 out wandb_name
    repo_root=$(cd -- "$(dirname -- "$0")" && pwd)
    cd "${repo_root}"
    g_check_device || return 1
    if [[ -n "${dry_run}" && "${dry_run}" != --dry-run ]] || (( $# > 3 )); then
        echo "用法: $0 G1 [cuda:N] [--dry-run]" >&2
        return 1
    fi
    export HC_HUMAN_SKILL_PROFILE=gap
    run_id="$(g_alloc_run_id G1 "${repo_root}")"
    out="${repo_root}/logs/rl_games/HcFactory/hier_${run_id}"
    wandb_name="$(g_wandb_train_name "${run_id}")"
    if [[ "${dry_run}" != --dry-run && -e "${out}" ]]; then
        echo "拒绝覆盖现有训练目录: ${out}；请改 HC_RUN_TAG / HC_HUMAN_RUN_TAG 或移走目录" >&2
        return 1
    fi
    if [[ "${run_id}" != G1 ]]; then
        echo "[G1] 目录已占用 → 自动使用 run_id=${run_id}"
    fi
    if [[ "${dry_run}" != --dry-run ]]; then
        g_source_wandb_env
    else
        export WANDB_ENTITY="${HC_WANDB_ENTITY:-${WANDB_ENTITY:-rl-driving}}"
        export WANDB_MODE="${HC_WANDB_MODE:-${WANDB_MODE:-online}}"
    fi
    export HC_WARMSTART=""
    unset HC_WANDB_RUN_ID WANDB_RUN_ID HC_WANDB_RESUME WANDB_RESUME
    local -a cmd=(
        python train.py --task HRTPaHC-v1 --algo hier
        --device "${DEVICE}" --num_envs 1 --headless --seed 42
        --train_n_products 10 --max_parallel_cd_dispatch 10
        --max_sim_episodes "${HC_MAX_TRAIN_EPISODES}"
        --autoregressive
        --wandb_activate --wandb_project HcFactory_TPA
        --wandb_name "${wandb_name}"
        --algo_variant G1
        --ftg_thresh_phy 0.95
        "+agent.params.config.full_experiment_name=${run_id}"
        agent.params.config.t_max_anchor="$(g_horizon_anchor)"
        agent.params.config.max_episodic_steps="$(g_horizon_steps)"
        agent.params.config.parallel_producing_limit=10
        agent.params.config.c_forbid_none_mode=always
        agent.params.config.curriculum=false
        agent.params.config.explore=false
        agent.params.config.explore_catalog=false
        agent.params.config.catalog_collect=false
        agent.params.config.oru=false
        agent.params.config.teacher_explore=false
        agent.params.config.autoregressive=true
        agent.params.config.ar_n_candidates=4
        agent.params.config.ar_softmax_temperature=1.0
        agent.params.config.ar_eps_scale_A=0.5
        agent.params.config.ar_eps_scale_B=0.5
        agent.params.config.ar_eps_scale_C=1.0
        agent.params.config.ar_eps_scale_D=1.0
        agent.params.config.prioritized_replay="$(g_rainbow_per)"
        agent.params.config.dueling_dqn="$(g_rainbow_dueling)"
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
    echo "[G1] skill_profile=gap; NO teacher load/explore; AR; max_ep=${HC_MAX_TRAIN_EPISODES}; out=${out}"
    echo "[G1] wandb=${wandb_name}"
    if [[ "${dry_run}" == --dry-run ]]; then
        printf '%q ' "${cmd[@]}"
        printf '\n'
        return 0
    fi
    "${cmd[@]}"
}

run_eval_g_plain() {
    # Protocol eval for G0/G1/G2 train dirs (gap; no match heads).
    local repo_root load_dir step dry_run="${3:-}" wandb_name variant
    variant="${MODE#eval-}"
    repo_root=$(cd -- "$(dirname -- "$0")" && pwd)
    cd "${repo_root}"
    g_check_device || return 1
    if [[ -n "${dry_run}" && "${dry_run}" != --dry-run ]] || (( $# > 3 )); then
        echo "用法: $0 eval-G0|eval-G1|eval-G2 [cuda:N] [--dry-run]" >&2
        return 1
    fi
    export HC_HUMAN_SKILL_PROFILE=gap
    if [[ -n "${HC_LOAD_DIR:-}" ]]; then
        load_dir="${HC_LOAD_DIR}"
    elif [[ "${variant}" == G0 ]]; then
        load_dir="$(g_teacher_dir "${repo_root}")"
    else
        load_dir="$(g_latest_run_dir "${repo_root}" "${variant}")"
        if [[ -z "${load_dir}" ]]; then
            load_dir="${repo_root}/logs/rl_games/HcFactory/hier_${variant}"
        fi
    fi
    if [[ ! -d "${load_dir}/nn" ]]; then
        echo "错误: 缺少训练目录: ${load_dir}（请先跑 ${variant}）" >&2
        return 1
    fi
    step="$(g_resolve_teacher_step "${load_dir}")" || return 1
    wandb_name="${variant}-N10-S42-step${step}-eval"
    if [[ "${dry_run}" != --dry-run ]]; then
        g_source_wandb_env
        local head
        for head in state_encoder agent_A agent_B agent_C agent_D_human agent_D_robot; do
            [[ -s "${load_dir}/nn/${head}_step_${step}.pth" ]] || {
                echo "缺少权重: ${load_dir}/nn/${head}_step_${step}.pth" >&2; return 1;
            }
        done
    else
        export WANDB_ENTITY="${HC_WANDB_ENTITY:-${WANDB_ENTITY:-rl-driving}}"
        export WANDB_MODE="${HC_WANDB_MODE:-${WANDB_MODE:-online}}"
    fi
    export HC_WARMSTART=""
    local -a cmd=(
        python train.py --task HRTPaHC-v1 --algo hier
        --device "${DEVICE}" --num_envs 1 --headless --seed 42
        --test --test_times "${HC_TEST_TIMES}" --test_seeds "${HC_TEST_SEEDS}"
        --test_epsilon 0 --train_n_products 10 --max_parallel_cd_dispatch 10
        --load_dir "${load_dir}" --load_step "${step}"
        --wandb_activate --wandb_project HcFactory_TPA_Eval
        --wandb_name "${wandb_name}"
        --ftg_thresh_phy 0.95
        --seed 42
        agent.params.config.t_max_anchor="$(g_horizon_anchor)"
        agent.params.config.max_episodic_steps="$(g_horizon_steps)"
        agent.params.config.parallel_producing_limit=10
        agent.params.config.c_forbid_none_mode=always
        agent.params.config.curriculum=false
        agent.params.config.explore=false
        agent.params.config.explore_catalog=false
        agent.params.config.catalog_collect=false
        agent.params.config.oru=false
        agent.params.config.prioritized_replay="$(g_rainbow_per)"
        agent.params.config.dueling_dqn="$(g_rainbow_dueling)"
        agent.params.config.noisy_net=false
        agent.params.config.hierarchical_credit=false
        agent.params.config.b_score_rl=false
        agent.params.config.env_rule_based_exploration=false
        'agent.params.config.warmstart=""'
        'agent.params.config.load_name=""'
    )
    echo "[eval-${variant}] gap; N10/K10/T40000 ε=0; seeds=43..52; step=${step}"
    echo "[eval-${variant}] load_dir=${load_dir}; wandb=${wandb_name}"
    if [[ "${dry_run}" == --dry-run ]]; then
        printf '%q ' "${cmd[@]}"
        printf '\n'
        return 0
    fi
    "${cmd[@]}"
}

run_g2_train() {
    # E5-no-oru recipe under gap, warmstart from G0.
    local repo_root load_dir step dry_run="${3:-}" run_id=G2 out wandb_name
    repo_root=$(cd -- "$(dirname -- "$0")" && pwd)
    cd "${repo_root}"
    g_check_device || return 1
    if [[ -n "${dry_run}" && "${dry_run}" != --dry-run ]] || (( $# > 3 )); then
        echo "用法: $0 G2 [cuda:N] [--dry-run]" >&2
        return 1
    fi
    export HC_HUMAN_SKILL_PROFILE=gap
    load_dir="$(g_teacher_dir "${repo_root}")"
    run_id="$(g_alloc_run_id G2 "${repo_root}")"
    out="${repo_root}/logs/rl_games/HcFactory/hier_${run_id}"
    wandb_name="$(g_wandb_train_name "${run_id}")"
    if [[ "${dry_run}" != --dry-run && -e "${out}" ]]; then
        echo "拒绝覆盖现有训练目录: ${out}；请改 HC_RUN_TAG / HC_HUMAN_RUN_TAG 或移走目录" >&2
        return 1
    fi
    if [[ "${run_id}" != G2 ]]; then
        echo "[G2] 目录已占用 → 自动使用 run_id=${run_id}"
    fi
    if [[ -d "${load_dir}/nn" ]]; then
        step="$(g_resolve_teacher_step "${load_dir}")" || return 1
    elif [[ "${dry_run}" == --dry-run ]]; then
        step="${HC_G_LOAD_STEP:-${HC_LOAD_STEP:-0}}"
    else
        echo "错误: 缺少 G 教师目录: ${load_dir}（请先跑 G0）" >&2
        return 1
    fi
    if [[ "${dry_run}" != --dry-run ]]; then
        g_source_wandb_env
        local head
        for head in state_encoder agent_A agent_B agent_C agent_D_human agent_D_robot; do
            [[ -s "${load_dir}/nn/${head}_step_${step}.pth" ]] || {
                echo "缺少教师模型: ${load_dir}/nn/${head}_step_${step}.pth" >&2; return 1;
            }
        done
    else
        export WANDB_ENTITY="${HC_WANDB_ENTITY:-${WANDB_ENTITY:-rl-driving}}"
        export WANDB_MODE="${HC_WANDB_MODE:-${WANDB_MODE:-online}}"
    fi
    export HC_WARMSTART=""
    unset HC_WANDB_RUN_ID WANDB_RUN_ID HC_WANDB_RESUME WANDB_RESUME
    local -a cmd=(
        python train.py --task HRTPaHC-v1 --algo hier
        --device "${DEVICE}" --num_envs 1 --headless --seed 42
        --train_n_products 10 --max_parallel_cd_dispatch 10
        --max_sim_episodes "${HC_MAX_TRAIN_EPISODES}"
        --load_dir "${load_dir}" --load_step "${step}"
        --teacher_explore
        --autoregressive
        --wandb_activate --wandb_project HcFactory_TPA
        --wandb_name "${wandb_name}"
        --algo_variant G2
        --ftg_thresh_phy 0.95
        "+agent.params.config.full_experiment_name=${run_id}"
        agent.params.config.t_max_anchor="$(g_horizon_anchor)"
        agent.params.config.max_episodic_steps="$(g_horizon_steps)"
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
        agent.params.config.prioritized_replay="$(g_rainbow_per)"
        agent.params.config.dueling_dqn="$(g_rainbow_dueling)"
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
    echo "[G2] gap; warmstart G0 step=${step}; teacher_explore+AR; max_ep=${HC_MAX_TRAIN_EPISODES}"
    echo "[G2] load=${load_dir}; out=${out}; wandb=${wandb_name}"
    if [[ "${dry_run}" == --dry-run ]]; then
        printf '%q ' "${cmd[@]}"
        printf '\n'
        return 0
    fi
    "${cmd[@]}"
}

g_reject_gap_on_e_series || exit 1

case "${MODE}" in
    G0) run_g0_train "$@" ;;
    G1) run_g1_train "$@" ;;
    G2) run_g2_train "$@" ;;
    G0-human-match|G0-human-match-c|G3|G4|G4-c) run_e5_human_train "$@" ;;
    eval-G0|eval-G1|eval-G2) run_eval_g_plain "$@" ;;
    eval-G0-human-match|eval-G0-human-match-c|eval-G3|eval-G4|eval-G4-c) run_eval_e5_human "$@" ;;
    # legacy aliases → numbered G series
    G-hard|G_hard)
        echo "[journal] 已改名: G-hard → G0" >&2
        MODE=G0; run_g0_train "$@"
        ;;
    G-scratch|G_scratch)
        echo "[journal] 已改名: G-scratch → G1" >&2
        MODE=G1; run_g1_train "$@"
        ;;
    G5-no-oru|G5_no_oru)
        echo "[journal] 已改名: G5-no-oru → G2" >&2
        MODE=G2; run_g2_train "$@"
        ;;
    G5-human)
        echo "[journal] 已改名: G5-human → G3" >&2
        MODE=G3; run_e5_human_train "$@"
        ;;
    G5-human-match)
        echo "[journal] 已改名: G5-human-match → G0-human-match" >&2
        MODE=G0-human-match; run_e5_human_train "$@"
        ;;
    G5-human-match-c)
        echo "[journal] 已改名: G5-human-match-c → G0-human-match-c" >&2
        MODE=G0-human-match-c; run_e5_human_train "$@"
        ;;
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
    E5-human|E5-human-pair|E5-pair|E5-human-pair-c|E5-human-pair-aux|E5-human-pair-c-aux|E5-human-match|E5-human-match-c) run_e5_human_train "$@" ;;
    eval-E5-human|eval-E5-human-pair|eval-E5-pair|eval-E5-human-pair-c|eval-E5-human-pair-aux|eval-E5-human-pair-c-aux|eval-E5-human-match|eval-E5-human-match-c) run_eval_e5_human "$@" ;;
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
    eval-E5-no-oru-near|eval_E5_no_oru_near|E5-no-oru-near-eval|eval-E5-no-oru-late|eval_E5_no_oru_late|E5-no-oru-late-eval) run_eval_e5_no_oru_near "$@" ;;
    eval-E5-no-oru-far|eval_E5_no_oru_far|E5-no-oru-far-eval) run_eval_e5_no_oru_far "$@" ;;
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
