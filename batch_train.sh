#!/bin/bash

# HcFactory TPA 默认（help 与 run 共用；可用环境变量覆盖）
HC_RULE_EPISODES="${HC_RULE_EPISODES:-2}"
# job 26（N10 HierRandom）单独 episode 数，不跟 HC_RULE_EPISODES
HC_RANDOM_EPISODES="${HC_RANDOM_EPISODES:-2}"
# Baseline wrapper runs 5 seeds × HC_RULE_EPISODES; direct invocation runs one seed.
HC_N16_RANDOM_EPISODES="${HC_N16_RANDOM_EPISODES:-2}"
HC_TEST_TIMES="${HC_TEST_TIMES:-2}"
HC_TEST_SEEDS="${HC_TEST_SEEDS:-42,43,44,45,46}"
HC_TRAIN_N_PRODUCTS="${HC_TRAIN_N_PRODUCTS:-16}"
HC_MULTI_K="${HC_MULTI_K:-10}"
# 全订单 anchor：N=16 的 T_max；N=10 = round(anchor/16)*10（fatigue 后默认 4× 旧 16000/10000）
HC_T_MAX_ANCHOR="${HC_T_MAX_ANCHOR:-64000}"
HC_PER_T_MAX=$(( (HC_T_MAX_ANCHOR + 8) / 16 ))
HC_T_MAX_N10=$(( HC_PER_T_MAX * 10 ))
HC_T_MAX_N16="${HC_T_MAX_ANCHOR}"
HC_EXPLORE_EPISODES="${HC_EXPLORE_EPISODES:-20}"
HC_POLICY_CATALOG_EPISODES="${HC_POLICY_CATALOG_EPISODES:-80}"
HC_WANDB_CATALOG_PROJECT="${HC_WANDB_CATALOG_PROJECT:-HcFactory_Catalog}"

# 用法:
#   ./batch_train.sh 22 cuda:0
#   ./batch_train.sh 22 24 25 26 27 cuda:0   # N=10 推荐主路径
#   ./batch_train.sh C cuda:0                # 同上：采库→rule→random→hier 课程
#   ./batch_train.sh D cuda:0                # 采库 → curriculum
#   ./batch_train.sh A cuda:0
#   ./batch_train.sh B
#   HC_WARMSTART=/path/to/ckpt.pkl ./batch_train.sh 22   # 可选：explore/curriculum 从 pkl 续跑
#
# HcFactory 编号（按流水线，旧 1–21 不动）:
#   22 采库 → 23 采库debug → 24/25 N10 rule → 26 N10 random
#   → 27 hier课程 → 28 hier硬训 → 29 N16评测 → 30/31/32 N16 基线
if [ $# -eq 0 ]; then
    echo "用法: $0 <A|B|C|D|E|T1|序号...> [cuda:N]"
    echo "  A: 运行A组训练 (1-5)"
    echo "  B: 运行B组训练 (6-10)"
    echo "  C: N=10 主路径 (22采库 → 24/25 rule → 26 random → 27 curriculum)"
    echo "  D: 采库+训练 (22 explore → 27 curriculum)"
    echo "  E: 基线矩阵 (24 25 26 | 30 31 32)"
    echo "  1-21: 旧 RL / Perception 序号"
    echo "  ---- HcFactory Hierarchical TPA（按流程编号）----"
    echo "  22: explore 采库 (--explore, ep=${HC_EXPLORE_EPISODES}, wandb=${HC_WANDB_CATALOG_PROJECT}, MetricCatalog/*)"
    echo "  23: explore debug（N=10, T_max=${HC_T_MAX_N10}, 可视化+warmstart，不录视频不启wandb）"
    echo "  24: rule 单产品 eval (K=1, N=10, T_max=${HC_T_MAX_N10}, seeds=${HC_TEST_SEEDS}, x${HC_TEST_TIMES})"
    echo "  25: rule 多产品 eval (K=${HC_MULTI_K}, N=10, T_max=${HC_T_MAX_N10}, seeds=${HC_TEST_SEEDS}, x${HC_TEST_TIMES})"
    echo "  26: hier random eval (ε=1, N=10, T_max=${HC_T_MAX_N10}, seeds=${HC_TEST_SEEDS}, x${HC_TEST_TIMES})"
    echo "  27: hier 倒序课程 (--curriculum → target=10)"
    echo "  28: hier N=10 硬训对照 (同 27 设置，无课程, T_max=${HC_T_MAX_N10})"
    echo "  29: hier 评测 (加载 nn/, 全量 N=16, T_max=${HC_T_MAX_N16})"
    echo "  30: rule 单产品 eval (K=1, N=16, T_max=${HC_T_MAX_N16}, seeds=${HC_TEST_SEEDS}, x${HC_TEST_TIMES})"
    echo "  31: rule 多产品 eval (K=${HC_MULTI_K}, N=16, T_max=${HC_T_MAX_N16}, seeds=${HC_TEST_SEEDS}, x${HC_TEST_TIMES})"
    echo "  32: hier random eval (ε=1, N=16, T_max=${HC_T_MAX_N16}, seeds=${HC_TEST_SEEDS}, x${HC_TEST_TIMES})"
    echo "  33: policy catalog hard train (catalog_collect, ep=${HC_POLICY_CATALOG_EPISODES:-80}, wandb catalog project)"
    echo "  T1: T1 流水线 (22 random explore → 28 hard train，同一 HC_CATALOG_TAG)"
    echo "  cuda:N: 可选，指定CUDA设备，默认 cuda:0（写在最后）"
    echo "  环境变量 HC_WARMSTART: 可选 pkl，传给 22/23/27/28 的 --warmstart"
    echo "  环境变量 HC_CATALOG_TAG: catalog 实验标签（例 T2_random_ep10），22 写库与 27 读库须一致"
    echo "  环境变量 HC_CATALOG_SOURCE: random_explore | policy_explore（默认 random_explore）"
    echo "  环境变量 HC_EXPLORE_CATALOG_DIR: 显式 catalog 根路径（覆盖 TAG 拼接）"
    echo "  环境变量 HC_EXPLORE_EPISODES: job 22 采库 episode 数（默认 ${HC_EXPLORE_EPISODES}）"
    echo "  环境变量 HC_WANDB_CATALOG_PROJECT: 采库/ policy catalog 专用 wandb project（默认 HcFactory_Catalog）"
    echo "  环境变量 HC_POLICY_CATALOG_EPISODES: job 33 policy catalog 训练 episode 数（默认 80）"
    echo "  环境变量 HC_LOAD_DIR: 29 号评测的实验目录（含 nn/）"
    echo "  环境变量 HC_RULE_EPISODES: rule 基线 episode 数（默认 ${HC_RULE_EPISODES}）"
    echo "  环境变量 HC_RANDOM_EPISODES: job 26 HierRandom episode 数（默认 ${HC_RANDOM_EPISODES}）"
    echo "  环境变量 HC_T_MAX_ANCHOR: 全订单 T_max anchor（默认 ${HC_T_MAX_ANCHOR} → N10=${HC_T_MAX_N10}）"
    echo "  环境变量 HC_WANDB_MODE: online|offline（默认 online 上云+本地 metrics.jsonl；网络不稳用 offline）"
    echo "  环境变量 HC_WANDB_SYNC: 1=仅 offline 时跑完后 wandb sync（默认 1；online 无需）"
    echo "  文件 .wandb_local.env: 共享机私有 wandb（HC_WANDB_API_KEY / HC_WANDB_ENTITY），gitignore，不影响别人"
    exit 1
fi

DEVICE="cuda:0"
JOBS=()
for arg in "$@"; do
    if [[ "$arg" =~ ^cuda:[0-9]+$ ]]; then
        DEVICE="$arg"
    elif [[ "$arg" =~ ^([1-9]|1[0-9]|2[0-9]|3[0-3])$ ]] || [ "$arg" = "A" ] || [ "$arg" = "B" ] || [ "$arg" = "C" ] || [ "$arg" = "D" ] || [ "$arg" = "E" ] || [ "$arg" = "T1" ]; then
        JOBS+=("$arg")
    else
        echo "错误: 无法识别参数 '$arg'"
        echo "用法: $0 <A|B|C|序号...> [cuda:N]"
        exit 1
    fi
done

if [ ${#JOBS[@]} -eq 0 ]; then
    echo "错误: 请指定至少一个任务序号或 A/B/C"
    exit 1
fi

DEVICE_ARG="--device ${DEVICE}"
echo "使用设备: ${DEVICE}"
echo "任务列表: ${JOBS[*]}"

# HcFactory TPA 公共参数
HC_TASK="HRTPaHC-v1"
HC_WANDB_PROJECT="${HC_WANDB_TRAIN_PROJECT:-HcFactory_TPA}"
HC_WANDB_TEST_PROJECT="${HC_WANDB_TEST_PROJECT:-HcFactory_TPA_Eval}"
HC_WANDB_BASELINE_PROJECT="${HC_WANDB_BASELINE_PROJECT:-${HC_WANDB_TEST_PROJECT}}"
HC_RUN_SEED="${HC_RUN_SEED:-}"
HC_NUM_ENVS=1
HC_WARMSTART="${HC_WARMSTART:-}"
# Catalog 命名与规模（job 22 写库 / job 27 读库须用同一 HC_CATALOG_TAG 或 HC_EXPLORE_CATALOG_DIR）
HC_CATALOG_SOURCE="${HC_CATALOG_SOURCE:-random_explore}"   # random_explore | policy_explore
HC_CATALOG_TAG="${HC_CATALOG_TAG:-}"                       # 例: T2_random_ep10；空=legacy 默认路径
HC_EXPLORE_EPISODES="${HC_EXPLORE_EPISODES:-20}"           # job 22 采库 episode 数（默认见脚本顶部）
HC_EXPLORE_N_PRODUCTS="${HC_EXPLORE_N_PRODUCTS:-10}"
HC_EXPLORE_CATALOG_DIR="${HC_EXPLORE_CATALOG_DIR:-}"       # 显式绝对/相对路径，优先级最高
HC_WANDB_CATALOG_PROJECT="${HC_WANDB_CATALOG_PROJECT:-HcFactory_Catalog}"
HC_POLICY_CATALOG_EPISODES="${HC_POLICY_CATALOG_EPISODES:-80}"

# wandb: 默认 online 边训边上云；metrics.jsonl 始终写本地。网络不稳可 HC_WANDB_MODE=offline
# 本地 metrics.jsonl 始终写入 logs/rl_games/HcFactory/<exp>/
#
# 共享机：不要 wandb logout。在本仓库放 gitignore 的 .wandb_local.env，只影响本脚本：
#   HC_WANDB_API_KEY=...
#   HC_WANDB_ENTITY=你的用户名或团队
export WANDB_MODE="${HC_WANDB_MODE:-${WANDB_MODE:-online}}"
export WANDB_HTTP_TIMEOUT="${WANDB_HTTP_TIMEOUT:-90}"
export WANDB_INIT_TIMEOUT="${WANDB_INIT_TIMEOUT:-120}"
HC_WANDB_SYNC="${HC_WANDB_SYNC:-1}"

# 仅加载本仓库私有配置（不写系统 ~/.netrc，不影响同机其他人）
HC_WANDB_LOCAL_ENV="${HC_WANDB_LOCAL_ENV:-.wandb_local.env}"
if [ -f "${HC_WANDB_LOCAL_ENV}" ]; then
    # shellcheck disable=SC1090
    set -a
    # shellcheck source=/dev/null
    . "${HC_WANDB_LOCAL_ENV}"
    set +a
    echo "[wandb] loaded local env: ${HC_WANDB_LOCAL_ENV}"
fi
if [ -n "${HC_WANDB_API_KEY:-}" ]; then
    export WANDB_API_KEY="${HC_WANDB_API_KEY}"
fi
if [ -n "${HC_WANDB_ENTITY:-}" ]; then
    export WANDB_ENTITY="${HC_WANDB_ENTITY}"
fi
echo "[wandb] mode=${WANDB_MODE} entity=${WANDB_ENTITY:-"(default login)"} sync_after_job=${HC_WANDB_SYNC} (metrics.jsonl always local)"

hc_wandb_sync_latest() {
    # Only meaningful after offline runs; never fail the training job.
    if [ "${WANDB_MODE}" != "offline" ]; then
        return 0
    fi
    if [ "${HC_WANDB_SYNC}" != "1" ]; then
        return 0
    fi
    local latest
    latest=$(ls -td wandb/offline-run-* 2>/dev/null | head -1)
    if [ -z "${latest}" ]; then
        echo "[wandb] no offline-run to sync"
        return 0
    fi
    echo "[wandb] syncing ${latest} ..."
    if command -v wandb >/dev/null 2>&1; then
        wandb sync "${latest}" || echo "[wandb] sync failed; local metrics.jsonl + offline-run still kept"
    else
        echo "[wandb] CLI missing; run later: wandb sync ${latest}"
    fi
}

# 可选 --warmstart（22/23/27）
hc_warmstart_args() {
    if [ -n "${HC_WARMSTART}" ]; then
        echo "--warmstart ${HC_WARMSTART}"
    fi
}

# Optional exact checkpoint step for job 29.
hc_load_step_args() {
    if [ -n "${HC_LOAD_STEP:-}" ]; then
        echo "--load_step ${HC_LOAD_STEP}"
    fi
}
# Optional per-run seed used by distributed baseline groups.
hc_seed_args() {
    if [ -n "${HC_RUN_SEED}" ]; then
        echo "--seed ${HC_RUN_SEED}"
    fi
}

# Unified eval protocol: one wandb run = all test_seeds × test_times episodes.
hc_test_args() {
    echo "--test --test_times ${HC_TEST_TIMES} --test_seeds ${HC_TEST_SEEDS}"
}


# 统一 T_max anchor（22–32 共用；可用 HC_T_MAX_ANCHOR 覆盖做极限探测）
hc_t_max_args() {
    echo "+t_max_anchor=${HC_T_MAX_ANCHOR}"
}

# Catalog 根目录：HC_EXPLORE_CATALOG_DIR > HC_CATALOG_TAG > legacy N{n}_T{t}
hc_catalog_root() {
    if [ -n "${HC_EXPLORE_CATALOG_DIR}" ]; then
        echo "${HC_EXPLORE_CATALOG_DIR}"
        return
    fi
    local _n="${HC_EXPLORE_N_PRODUCTS}"
    local _t="${HC_T_MAX_N10}"
    if [ -n "${HC_CATALOG_TAG}" ]; then
        echo "env_checkpoints/${HC_CATALOG_SOURCE}/N${_n}_T${_t}__${HC_CATALOG_TAG}"
    else
        echo "env_checkpoints/${HC_CATALOG_SOURCE}/N${_n}_T${_t}"
    fi
}

hc_catalog_args() {
    echo "+explore_catalog_dir=$(hc_catalog_root)"
}

hc_print_catalog_hint() {
    local root
    root="$(hc_catalog_root)"
    echo "[catalog] root=${root}"
    if [ -z "${HC_CATALOG_TAG}" ] && [ -z "${HC_EXPLORE_CATALOG_DIR}" ]; then
        echo "[catalog] 提示: 并行多组实验请设置 HC_CATALOG_TAG，避免写入默认 N10_T40000 互相覆盖"
        echo "[catalog] 示例: HC_CATALOG_TAG=T2_random_ep10 ./batch_train.sh 22 cuda:0"
    fi
    echo "[catalog] 训练读库请复用: HC_CATALOG_TAG=${HC_CATALOG_TAG:-'(unset)'} HC_EXPLORE_CATALOG_DIR=${HC_EXPLORE_CATALOG_DIR:-'(unset)'} ./batch_train.sh 27 cuda:0"
}

# 定义训练函数（run_one_job 在文件末尾调用）

run_test_2() {
    echo "运行训练 2: D3QN penalty"
    python train.py --task Isaac-TaskAllocation-Direct-v1 --algo rl_filter --headless --wandb_activate ${DEVICE_ARG}
}

run_test_1() {
    echo "运行训练 1: D3QN"
    python train.py --task Isaac-TaskAllocation-Direct-v1 --algo rl_filter --headless --wandb_activate ${DEVICE_ARG}
}

run_test_3() {
    echo "运行训练 3: PF-CD3Q"
    python train.py --task Isaac-TaskAllocation-Direct-v1 --algo rl_filter --headless --wandb_activate --use_fatigue_mask ${DEVICE_ARG}
}

run_test_4() {
    echo "运行训练 4: PF-CD3QP"
    python train.py --task Isaac-TaskAllocation-Direct-v1 --algo rl_filter --headless --wandb_activate --use_fatigue_mask --other_filters ${DEVICE_ARG}
}

run_test_5() {
    echo "运行训练 5: DQN with penalty"
    python train.py --task Isaac-TaskAllocation-Direct-v1 --algo dqn --headless --wandb_activate ${DEVICE_ARG}

}

run_test_6() {
    echo "运行训练 6: PF-DQN"
    python train.py --task Isaac-TaskAllocation-Direct-v1 --algo dqn --headless --wandb_activate --use_fatigue_mask ${DEVICE_ARG}
}

run_test_7() {
    echo "运行训练 7: PPO-dis with penalty"
    python train.py --task Isaac-TaskAllocation-Direct-v1 --algo ppo_dis --headless --wandb_activate ${DEVICE_ARG}
}

run_test_8() {
    echo "运行训练 8: PF-PPO-dis"
    python train.py --task Isaac-TaskAllocation-Direct-v1 --algo ppo_dis --headless --wandb_activate --use_fatigue_mask ${DEVICE_ARG}
}

run_test_9() {
    echo "运行训练 9: PPO-lag"
    python train.py --task Isaac-TaskAllocation-Direct-v1 --algo ppolag_filter_dis --headless --wandb_activate ${DEVICE_ARG}
}

run_test_10() {
    echo "运行训练 10: PF-PPO-lag"
    python train.py --task Isaac-TaskAllocation-Direct-v1 --algo ppolag_filter_dis --headless --wandb_activate --use_fatigue_mask ${DEVICE_ARG}
}


##### ablation study #####
run_test_11() {
    echo "运行训练 11: rl_filter_no_noisy"
    python train.py --task Isaac-TaskAllocation-Direct-v1 --algo rl_filter_no_noisy --headless --wandb_activate --use_fatigue_mask ${DEVICE_ARG}
}

run_test_12() {
    echo "运行训练 12: rl_filter_no_dueling"
    python train.py --task Isaac-TaskAllocation-Direct-v1 --algo rl_filter_no_dueling --headless --wandb_activate --use_fatigue_mask ${DEVICE_ARG}
}

run_test_13() {
    echo "运行训练 13: rl_filter_selfattn"
    python train.py --task Isaac-TaskAllocation-Direct-v1 --algo rl_filter_selfattn --headless --wandb_activate --use_fatigue_mask ${DEVICE_ARG}
}

run_test_14() {
    echo "运行训练 14: rl_filter_mlp"
    python train.py --task Isaac-TaskAllocation-Direct-v1 --algo rl_filter_mlp --headless --wandb_activate --use_fatigue_mask ${DEVICE_ARG}
}



run_test_15() {
    echo "运行训练 15: HcFactory + PF-CD3Q"
    python train.py --task HRTPaHC-v1 --algo rl_filter --headless --wandb_activate --use_fatigue_mask ${DEVICE_ARG} --num_envs 2
}


run_test_16() {
    #live stream
    python train.py --task HRTPaHC-v1 --algo rl_filter --headless --active_livestream --livestream_public_ip 10.68.217.239 --livestream_port 49100 ${DEVICE_ARG}
}

run_test_17() {
    #test
    python train.py --task HRTPaHC-v1 --algo rl_filter ${DEVICE_ARG}
}

##### Perception (human-id + human-subtask) #####
# 数据集：默认 max_episodes=6，按 episode 划分 train/val/test = 4 / 1 / 1
PERCEPTION_PY="source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/src/perception.py"
PERCEPTION_DATASET="source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/output/perception_dataset"
PERCEPTION_RUNS="source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/output/perception_runs"

run_test_18() {
    # Perception 数据采集：仿真 collect，需 enable_cameras；跑满 6 个 episode
    # 输出：.../output/perception_dataset/env_XX_episode_XXXXXX/
    echo "运行 18: Perception collect (6 episodes → train 4 / val 1 / test 1)"
    python train.py \
        --task HRTPaHC-v1 \
        --algo rule_based \
        --num_envs 1 \
        --headless \
        --enable_cameras \
        ${DEVICE_ARG}
}

run_test_19() {
    # Perception 训练任务 A：多视角 human id 识别（episode 级 70/15/15）
    echo "运行 19: Perception train human-id"
    python "${PERCEPTION_PY}" train \
        --task id \
        --dataset_dir "${PERCEPTION_DATASET}" \
        --output_dir "${PERCEPTION_RUNS}" \
        --run_name perception_baseline \
        --epochs 20 \
        --batch_size 32 \
        ${DEVICE_ARG}
}

run_test_20() {
    # Perception 训练任务 B：working human 的 subtask + done
    echo "运行 20: Perception train human-subtask"
    python "${PERCEPTION_PY}" train \
        --task subtask \
        --dataset_dir "${PERCEPTION_DATASET}" \
        --output_dir "${PERCEPTION_RUNS}" \
        --run_name perception_baseline \
        --epochs 20 \
        --batch_size 32 \
        ${DEVICE_ARG}
}

run_test_21() {
    # Perception 评估：在 test 集（1 episode）上分别 eval id / subtask
    echo "运行 21: Perception eval (id + subtask on test split)"
    python "${PERCEPTION_PY}" eval \
        --task id \
        --dataset_dir "${PERCEPTION_DATASET}" \
        --checkpoint "${PERCEPTION_RUNS}/perception_baseline_id/best.pt" \
        ${DEVICE_ARG}
    python "${PERCEPTION_PY}" eval \
        --task subtask \
        --dataset_dir "${PERCEPTION_DATASET}" \
        --checkpoint "${PERCEPTION_RUNS}/perception_baseline_subtask/best.pt" \
        ${DEVICE_ARG}
}

##### HcFactory Hierarchical TPA #####
# 编号按流水线：采库 → N10 基线 → N10 训练 → N16 评测/基线
# 旧映射: 25→22, 27→23, 22→24, 23→25, 29→26, 26→27, 24→28, 28→29; 30–32 不变

run_test_22() {
    # N=10 masked random 采集库：ε=1，无 DQN backward；progress key 去重；MetricCatalog/* → HC_WANDB_CATALOG_PROJECT
    hc_print_catalog_hint
    local _wp="${HC_WANDB_CATALOG_PROJECT}"
    echo "运行 22: explore catalog (N=${HC_EXPLORE_N_PRODUCTS}, T_max=${HC_T_MAX_N10}, epsilon=1, episodes=${HC_EXPLORE_EPISODES}, wandb=${_wp})"
    python train.py \
        --task "${HC_TASK}" \
        --algo hier \
        --num_envs "${HC_NUM_ENVS}" \
        --headless \
        --explore \
        --explore_n_products "${HC_EXPLORE_N_PRODUCTS}" \
        --max_sim_episodes "${HC_EXPLORE_EPISODES}" \
        --wandb_activate \
        --wandb_project "${_wp}" \
        --wandb_name "explore_N${HC_EXPLORE_N_PRODUCTS}_T${HC_T_MAX_N10}__${HC_CATALOG_TAG:-legacy}" \
        $(hc_t_max_args) \
        $(hc_catalog_args) \
        $(hc_warmstart_args) \
        ${DEVICE_ARG}
}

run_test_23() {
    # 采集 debug：不开 --headless（可视化UI），读取本地阻塞点 pkl 并手动断点调试
    # 不开 wandb，不录视频；与 22 对齐 N=10 / T_max=${HC_T_MAX_N10}
    #
    # 你需要设置：
    #   export HC_WARMSTART=/abs/path/to/env_checkpoints/stagnation/collect/.../stalled_state.pkl
    if [ -z "${HC_WARMSTART}" ]; then
        echo "错误: run_test_23 需要先设置 HC_WARMSTART 为阻塞点 stalled_state.pkl"
        echo "示例：HC_WARMSTART=env_checkpoints/stagnation/collect/L2_env00_.../stalled_state.pkl ./batch_train.sh 23 cuda:0"
        exit 1
    fi
    echo "运行 23: explore debug (N=10, T_max=${HC_T_MAX_N10}, visual+warmstart, no wandb/no video)"
    python train.py \
        --task "${HC_TASK}" \
        --algo hier \
        --num_envs 1 \
        --explore \
        --explore_n_products 10 \
        --seed 42 \
        $(hc_t_max_args) \
        $(hc_warmstart_args) \
        +decision_ring_k=20 \
        ${DEVICE_ARG}
}

run_test_24() {
    # rule_based + 单产品决策（max_parallel_cd_dispatch=1），N=10
    echo "运行 24: rule_based single-product eval (K=1, N=10, T_max=${HC_T_MAX_N10}, seeds=${HC_TEST_SEEDS}, x${HC_TEST_TIMES}, wandb)"
    python train.py \
        --task "${HC_TASK}" \
        --algo rule_based \
        --num_envs "${HC_NUM_ENVS}" \
        --headless \
        --wandb_activate \
        --wandb_project "${HC_WANDB_BASELINE_PROJECT}" \
        --wandb_name "rule_K1_N10_T${HC_T_MAX_N10}_5seed_x${HC_TEST_TIMES}" \
        --train_n_products 10 \
        --max_parallel_cd_dispatch 1 \
        $(hc_test_args) \
        $(hc_t_max_args) \
        ${DEVICE_ARG}
}

run_test_25() {
    # rule_based + 多产品并行决策（K=10），N=10
    echo "运行 25: rule_based multi-product eval (K=${HC_MULTI_K}, N=10, T_max=${HC_T_MAX_N10}, seeds=${HC_TEST_SEEDS}, x${HC_TEST_TIMES}, wandb)"
    python train.py \
        --task "${HC_TASK}" \
        --algo rule_based \
        --num_envs "${HC_NUM_ENVS}" \
        --headless \
        --wandb_activate \
        --wandb_project "${HC_WANDB_BASELINE_PROJECT}" \
        --wandb_name "rule_K${HC_MULTI_K}_N10_T${HC_T_MAX_N10}_5seed_x${HC_TEST_TIMES}" \
        --train_n_products 10 \
        --max_parallel_cd_dispatch "${HC_MULTI_K}" \
        $(hc_test_args) \
        $(hc_t_max_args) \
        ${DEVICE_ARG}
}

run_test_26() {
    # hier masked-random makespan 基线：N=10, ε=1, 无 DQN 学习
    echo "运行 26: hier random baseline eval (ε=1, N=10, T=${HC_T_MAX_N10}, seeds=${HC_TEST_SEEDS}, x${HC_TEST_TIMES}, wandb)"
    python train.py \
        --task "${HC_TASK}" \
        --algo hier \
        --num_envs "${HC_NUM_ENVS}" \
        --headless \
        --max_parallel_cd_dispatch "${HC_MULTI_K}" \
        --wandb_activate \
        --wandb_project "${HC_WANDB_BASELINE_PROJECT}" \
        --wandb_name "random_K${HC_MULTI_K}_N10_T${HC_T_MAX_N10}_5seed_x${HC_TEST_TIMES}" \
        --train_n_products 10 \
        --test_epsilon 1 \
        $(hc_test_args) \
        $(hc_t_max_args) \
        ${DEVICE_ARG}
}

run_test_27() {
    # 倒序课程：target=10；T_budget=ΔN×per_T_max；catalog 按 start_nfin 切片
    hc_print_catalog_hint
    echo "运行 27: hier curriculum (reverse ΔN →10, per_T=${HC_PER_T_MAX}, catalog entries at bind)"
    python train.py \
        --task "${HC_TASK}" \
        --algo hier \
        --num_envs "${HC_NUM_ENVS}" \
        --headless \
        --curriculum \
        --wandb_activate \
        --wandb_project "${HC_WANDB_PROJECT}" \
        --wandb_name "hier_curriculum_K${HC_MULTI_K}_T${HC_T_MAX_N10}__${HC_CATALOG_TAG:-legacy}" \
        --max_parallel_cd_dispatch "${HC_MULTI_K}" \
        $(hc_t_max_args) \
        $(hc_catalog_args) \
        $(hc_warmstart_args) \
        ${DEVICE_ARG}
}

run_test_28() {
    # hier N=10 硬训对照：与 27 同 K / T_anchor，无 --curriculum（整单 0→10）
    hc_print_catalog_hint
    echo "运行 28: hier hard train (no curriculum, N=10, T_max=${HC_T_MAX_N10}, wandb=${HC_WANDB_PROJECT})"
    python train.py \
        --task "${HC_TASK}" \
        --algo hier \
        --num_envs "${HC_NUM_ENVS}" \
        --headless \
        --wandb_activate \
        --wandb_project "${HC_WANDB_PROJECT}" \
        --wandb_name "hier_hard_K${HC_MULTI_K}_N10_T${HC_T_MAX_N10}__${HC_CATALOG_TAG:-legacy}" \
        --max_parallel_cd_dispatch "${HC_MULTI_K}" \
        $(hc_t_max_args) \
        $(hc_catalog_args) \
        $(hc_warmstart_args) \
        ${DEVICE_ARG}
}

run_test_29() {
    # 全量 16 件评测：加载 27 训练的 nn/，不用 --curriculum
    if [ -z "${HC_LOAD_DIR}" ]; then
        echo "错误: run_test_29 需要 HC_LOAD_DIR 指向训练实验目录（含 nn/）"
        echo "示例：HC_LOAD_DIR=logs/rl_games/HcFactory/hier_2026-08-18_22-00-00 ./batch_train.sh 29 cuda:0"
        exit 1
    fi
    _eval_n="${HC_TRAIN_N_PRODUCTS}"
    _eval_t=$([ "${_eval_n}" = "10" ] && echo "${HC_T_MAX_N10}" || echo "${HC_T_MAX_N16}")
    _eval_tag="${HC_EVAL_VARIANT:-eval}"
    echo "运行 29: hier test (${_eval_tag}) full-order N=${_eval_n} T_max=${_eval_t} load=${HC_LOAD_DIR}"
    python train.py \
        --task "${HC_TASK}" \
        --algo hier \
        --num_envs 1 \
        --headless \
        --test \
        --test_times "${HC_TEST_TIMES}" \
        --test_seeds "${HC_TEST_SEEDS}" \
        --train_n_products "${_eval_n}" \
        --load_dir "${HC_LOAD_DIR}" \
        --wandb_activate \
        --wandb_project "${HC_WANDB_TEST_PROJECT}" \
        --wandb_name "hier_eval_${_eval_tag}_N${_eval_n}_step${HC_LOAD_STEP:-latest}_T${_eval_t}" \
        $(hc_load_step_args) \
        $(hc_t_max_args) \
        ${DEVICE_ARG}
}

run_test_30() {
    # rule_based + 单产品决策，N=16 全量订单
    echo "运行 30: rule_based single-product eval (K=1, N=16, T_max=${HC_T_MAX_N16}, seeds=${HC_TEST_SEEDS}, x${HC_TEST_TIMES}, wandb)"
    python train.py \
        --task "${HC_TASK}" \
        --algo rule_based \
        --num_envs "${HC_NUM_ENVS}" \
        --headless \
        --wandb_activate \
        --wandb_project "${HC_WANDB_BASELINE_PROJECT}" \
        --wandb_name "rule_K1_N16_T${HC_T_MAX_N16}_5seed_x${HC_TEST_TIMES}" \
        --train_n_products 16 \
        --max_parallel_cd_dispatch 1 \
        $(hc_test_args) \
        $(hc_t_max_args) \
        ${DEVICE_ARG}
}

run_test_31() {
    # rule_based + 多产品并行决策，N=16 全量订单
    echo "运行 31: rule_based multi-product eval (K=${HC_MULTI_K}, N=16, T_max=${HC_T_MAX_N16}, seeds=${HC_TEST_SEEDS}, x${HC_TEST_TIMES}, wandb)"
    python train.py \
        --task "${HC_TASK}" \
        --algo rule_based \
        --num_envs "${HC_NUM_ENVS}" \
        --headless \
        --wandb_activate \
        --wandb_project "${HC_WANDB_BASELINE_PROJECT}" \
        --wandb_name "rule_K${HC_MULTI_K}_N16_T${HC_T_MAX_N16}_5seed_x${HC_TEST_TIMES}" \
        --train_n_products 16 \
        --max_parallel_cd_dispatch "${HC_MULTI_K}" \
        $(hc_test_args) \
        $(hc_t_max_args) \
        ${DEVICE_ARG}
}

run_test_32() {
    # hier masked-random makespan 基线：N=16, ε=1, 无 DQN 学习
    echo "运行 32: hier random baseline eval (ε=1, N=16, T=${HC_T_MAX_N16}, seeds=${HC_TEST_SEEDS}, x${HC_TEST_TIMES}, wandb)"
    python train.py \
        --task "${HC_TASK}" \
        --algo hier \
        --num_envs "${HC_NUM_ENVS}" \
        --headless \
        --max_parallel_cd_dispatch "${HC_MULTI_K}" \
        --wandb_activate \
        --wandb_project "${HC_WANDB_BASELINE_PROJECT}" \
        --wandb_name "random_K${HC_MULTI_K}_N16_T${HC_T_MAX_N16}_5seed_x${HC_TEST_TIMES}" \
        --train_n_products 16 \
        --test_epsilon 1 \
        $(hc_test_args) \
        $(hc_t_max_args) \
        ${DEVICE_ARG}
}

run_test_33() {
    # Policy-guided catalog: hard train + catalog_collect，MetricCatalog/* + MetricTrain/* + MetricLoss/*
    hc_print_catalog_hint
    local _wp="${HC_WANDB_CATALOG_PROJECT}"
    local _tag="${HC_CATALOG_TAG:-T3_policy_ep80}"
    local _ep="${HC_POLICY_CATALOG_EPISODES}"
    HC_CATALOG_SOURCE="${HC_CATALOG_SOURCE:-policy_explore}"
    export HC_CATALOG_SOURCE
    echo "运行 33: policy catalog hard train (catalog_collect, ep=${_ep}, wandb=${_wp}, tag=${_tag})"
    python train.py \
        --task "${HC_TASK}" \
        --algo hier \
        --num_envs "${HC_NUM_ENVS}" \
        --headless \
        --catalog_collect \
        --max_sim_episodes "${_ep}" \
        --wandb_activate \
        --wandb_project "${_wp}" \
        --wandb_name "policy_catalog_K${HC_MULTI_K}_ep${_ep}__${_tag}" \
        --max_parallel_cd_dispatch "${HC_MULTI_K}" \
        $(hc_t_max_args) \
        $(hc_catalog_args) \
        $(hc_warmstart_args) \
        ${DEVICE_ARG}
}

run_t1() {
    # T1 Phase A: random explore 采库 (HcFactory_Catalog) → Phase B: hard train (HcFactory_TPA)
    export HC_CATALOG_TAG="${HC_CATALOG_TAG:-T1_random_ep20}"
    export HC_CATALOG_SOURCE="${HC_CATALOG_SOURCE:-random_explore}"
    export HC_EXPLORE_EPISODES="${HC_EXPLORE_EPISODES:-20}"
    echo "=== T1 流水线: 22 explore → 28 hard train (tag=${HC_CATALOG_TAG}) ==="
    run_test_22
    run_test_28
    echo "=== T1 流水线完成（Phase B ORU 待实现；当前 28 为 online hard train）==="
}

# 调度：按序号 / A / B / C 调用上面的 run_test_*
run_one_job() {
    local id=$1
    case $id in
        1) run_test_1 ;;
        2) run_test_2 ;;
        3) run_test_3 ;;
        4) run_test_4 ;;
        5) run_test_5 ;;
        6) run_test_6 ;;
        7) run_test_7 ;;
        8) run_test_8 ;;
        9) run_test_9 ;;
        10) run_test_10 ;;
        11) run_test_11 ;;
        12) run_test_12 ;;
        13) run_test_13 ;;
        14) run_test_14 ;;
        15) run_test_15 ;;
        16) run_test_16 ;;
        17) run_test_17 ;;
        18) run_test_18 ;;
        19) run_test_19 ;;
        20) run_test_20 ;;
        21) run_test_21 ;;
        22) run_test_22 ;;
        23) run_test_23 ;;
        24) run_test_24 ;;
        25) run_test_25 ;;
        26) run_test_26 ;;
        27) run_test_27 ;;
        28) run_test_28 ;;
        29) run_test_29 ;;
        30) run_test_30 ;;
        31) run_test_31 ;;
        32) run_test_32 ;;
        33) run_test_33 ;;
        T1) run_t1 ;;
        A)
            echo "=== 运行A组训练 (1-5) ==="
            run_test_1; run_test_2; run_test_3; run_test_4; run_test_5; run_test_6
            echo "A组训练完成！"
            ;;
        B)
            echo "=== 运行B组训练 (6-10) ==="
            run_test_7; run_test_8; run_test_9; run_test_10
            echo "B组训练完成！"
            ;;
        C)
            echo "=== 运行C组: N=10 主路径 (22采库 → 24/25 rule → 26 random → 27 curriculum) ==="
            run_test_22
            run_test_24
            run_test_25
            run_test_26
            run_test_27
            echo "C组训练完成！"
            ;;
        D)
            echo "=== 运行D组: 采库+训练 (22 explore → 27 curriculum) ==="
            run_test_22
            run_test_27
            echo "D组训练完成！"
            ;;
        E)
            echo "=== 运行E组: 基线矩阵 (N=10: 24 25 26 → N=16: 30 31 32) ==="
            run_test_24
            run_test_25
            run_test_26
            run_test_30
            run_test_31
            run_test_32
            echo "E组基线矩阵完成！"
            ;;
        *) echo "错误: 无效的训练序号 $id"; return 1 ;;
    esac
    if [[ "$id" =~ ^[0-9]+$ ]]; then
        echo "训练 $id 完成！"
    fi
}

# 依次执行任务
for job in "${JOBS[@]}"; do
    echo ">>> 开始任务: $job"
    run_one_job "$job" || exit 1
    # HcFactory 22–32：offline 跑完后尝试上传到云端
    if [[ "$job" =~ ^(2[2-9]|3[0-3]|C|D|E|T1)$ ]]; then
        hc_wandb_sync_latest
    fi
done

echo "所有训练完成！"
