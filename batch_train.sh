#!/bin/bash

# 用法:
#   ./batch_train.sh 22 cuda:0
#   ./batch_train.sh 22 23 24 cuda:0     # 依次跑多个序号
#   ./batch_train.sh C cuda:0            # HcFactory TPA 三连：rule K=1 / rule K=5 / hier
#   ./batch_train.sh D cuda:0            # 长 horizon 主路径：explore 采库 → curriculum 训练
#   ./batch_train.sh A cuda:0
#   ./batch_train.sh B
#   HC_WARMSTART=/path/to/ckpt.pkl ./batch_train.sh 25   # 可选：explore/curriculum 从 pkl 续跑
if [ $# -eq 0 ]; then
    echo "用法: $0 <A|B|C|D|序号...> [cuda:N]"
    echo "  A: 运行A组训练 (1-5)"
    echo "  B: 运行B组训练 (6-10)"
    echo "  C: HcFactory TPA (22 rule单产品 / 23 rule多产品 / 24 hier全量硬训)"
    echo "  D: 长 horizon 主路径 (25 explore采库 → 26 curriculum训练)"
    echo "  1-21: 旧 RL / Perception 序号"
    echo "  22: rule_based + 单产品决策 (K=1, 10 episodes, wandb)"
    echo "  23: rule_based + 多产品决策 (K=${HC_MULTI_K}, 10 episodes, wandb)"
    echo "  24: hier 全量硬训对照 (K=${HC_MULTI_K}, max_episodic_steps=45000, wandb)"
    echo "  25: hier masked-random 采集库 (--explore, N=16, T_max=25000, ε=1, 10 episodes)"
    echo "  26: hier 课程训练 (--curriculum, 1→16 件, wandb)"
    echo "  27: 采集 debug（可视化+warmstart，不录视频不启wandb）"
    echo "  cuda:N: 可选，指定CUDA设备，默认 cuda:0（写在最后）"
    echo "  环境变量 HC_WARMSTART: 可选 pkl，传给 25/26 的 --warmstart"
    exit 1
fi

DEVICE="cuda:0"
JOBS=()
for arg in "$@"; do
    if [[ "$arg" =~ ^cuda:[0-9]+$ ]]; then
        DEVICE="$arg"
    elif [[ "$arg" =~ ^([1-9]|1[0-9]|2[0-7])$ ]] || [ "$arg" = "A" ] || [ "$arg" = "B" ] || [ "$arg" = "C" ] || [ "$arg" = "D" ]; then
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
HC_WANDB_PROJECT="HcFactory_TPA"
HC_NUM_ENVS=1
HC_RULE_EPISODES=10
HC_MULTI_K=10
HC_WARMSTART="${HC_WARMSTART:-}"

# 可选 --warmstart（25/26）
hc_warmstart_args() {
    if [ -n "${HC_WARMSTART}" ]; then
        echo "--warmstart ${HC_WARMSTART}"
    fi
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
run_test_22() {
    # rule_based + 单产品决策（max_parallel_cd_dispatch=1），跑 10 个 episode
    echo "运行 22: rule_based single-product (K=1, ${HC_RULE_EPISODES} episodes, wandb)"
    python train.py \
        --task "${HC_TASK}" \
        --algo rule_based \
        --num_envs "${HC_NUM_ENVS}" \
        --headless \
        --wandb_activate \
        --wandb_project "${HC_WANDB_PROJECT}" \
        --wandb_name "rule_K1_single_${HC_RULE_EPISODES}ep" \
        --max_parallel_cd_dispatch 1 \
        --max_sim_episodes "${HC_RULE_EPISODES}" \
        ${DEVICE_ARG}
}

run_test_23() {
    # rule_based + 多产品并行决策（K=5），跑 10 个 episode
    echo "运行 23: rule_based multi-product (K=${HC_MULTI_K}, ${HC_RULE_EPISODES} episodes, wandb)"
    python train.py \
        --task "${HC_TASK}" \
        --algo rule_based \
        --num_envs "${HC_NUM_ENVS}" \
        --headless \
        --wandb_activate \
        --wandb_project "${HC_WANDB_PROJECT}" \
        --wandb_name "rule_K${HC_MULTI_K}_multi_${HC_RULE_EPISODES}ep" \
        --max_parallel_cd_dispatch "${HC_MULTI_K}" \
        --max_sim_episodes "${HC_RULE_EPISODES}" \
        ${DEVICE_ARG}
}

run_test_24() {
    # hier 全量硬训对照（yaml max_episodic_steps=45000，无课程）
    echo "运行 24: hier 全量硬训对照 (K=${HC_MULTI_K}, wandb)"
    python train.py \
        --task "${HC_TASK}" \
        --algo hier \
        --num_envs "${HC_NUM_ENVS}" \
        --headless \
        --wandb_activate \
        --wandb_project "${HC_WANDB_PROJECT}" \
        --wandb_name "hier_K${HC_MULTI_K}_hard45k" \
        --max_parallel_cd_dispatch "${HC_MULTI_K}" \
        ${DEVICE_ARG}
}

run_test_25() {
    # N=16 masked random 采集库：ε=1，无 DQN backward；L2/L3 死锁回退 + progress key 去重
    echo "运行 25: explore catalog (N=16, T_max=25000, epsilon=1, 10 episodes)"
    python train.py \
        --task "${HC_TASK}" \
        --algo hier \
        --num_envs "${HC_NUM_ENVS}" \
        --headless \
        --explore \
        --max_sim_episodes 10 \
        --wandb_activate \
        --wandb_project "${HC_WANDB_PROJECT}" \
        --wandb_name "explore_N16_T25000" \
        $(hc_warmstart_args) \
        ${DEVICE_ARG}
}

run_test_26() {
    # 课程训练：件数 1→16，T_max(N)=25000*N/16；stage>0 从 catalog 按 n_finished 切片 warmstart
    echo "运行 26: hier curriculum (1→16 products, wandb)"
    python train.py \
        --task "${HC_TASK}" \
        --algo hier \
        --num_envs "${HC_NUM_ENVS}" \
        --headless \
        --curriculum \
        --wandb_activate \
        --wandb_project "${HC_WANDB_PROJECT}" \
        --wandb_name "hier_curriculum_K${HC_MULTI_K}" \
        --max_parallel_cd_dispatch "${HC_MULTI_K}" \
        $(hc_warmstart_args) \
        ${DEVICE_ARG}
}

run_test_27() {
    # 采集 debug：不开 --headless（可视化UI），读取本地阻塞点 pkl 并手动断点调试
    # 不开 wandb，不录视频
    #
    # 你需要设置：
    #   export HC_WARMSTART=/abs/path/to/env_checkpoints/stagnation/collect/.../stalled_state.pkl
    if [ -z "${HC_WARMSTART}" ]; then
        echo "错误: run_test_27 需要先设置 HC_WARMSTART 为阻塞点 stalled_state.pkl"
        echo "示例：HC_WARMSTART=env_checkpoints/stagnation/collect/L2_env00_.../stalled_state.pkl ./batch_train.sh 27 cuda:0"
        exit 1
    fi
    echo "运行 27: explore debug (visual+warmstart, no wandb/no video)"
    python train.py \
        --task "${HC_TASK}" \
        --algo hier \
        --num_envs 1 \
        --explore \
        --seed 42 \
        $(hc_warmstart_args) \
        +t_max_anchor=14000 \
        +decision_ring_k=20 \
        ${DEVICE_ARG}
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
            echo "=== 运行C组: HcFactory TPA (22→23→24) ==="
            run_test_22
            run_test_23
            run_test_24
            echo "C组训练完成！"
            ;;
        D)
            echo "=== 运行D组: 长 horizon 主路径 (25 explore → 26 curriculum) ==="
            run_test_25
            run_test_26
            echo "D组训练完成！"
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
done

echo "所有训练完成！"
