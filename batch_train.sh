#!/bin/bash

# 用法:
#   ./batch_train.sh 19 cuda:0
#   ./batch_train.sh 19 20 cuda:0          # 依次跑多个序号
#   ./batch_train.sh A cuda:0
#   ./batch_train.sh B
if [ $# -eq 0 ]; then
    echo "用法: $0 <A|B|序号...> [cuda:N]"
    echo "  A: 运行A组训练 (1-5)"
    echo "  B: 运行B组训练 (6-10)"
    echo "  1-17: RL / HcFactory 训练序号（可多个，如 19 20）"
    echo "  18-21: Perception 采集 / 训练 / 评估"
    echo "  cuda:N: 可选，指定CUDA设备，默认 cuda:0（写在最后）"
    exit 1
fi

DEVICE="cuda:0"
JOBS=()
for arg in "$@"; do
    if [[ "$arg" =~ ^cuda:[0-9]+$ ]]; then
        DEVICE="$arg"
    elif [[ "$arg" =~ ^([1-9]|1[0-9]|2[01])$ ]] || [ "$arg" = "A" ] || [ "$arg" = "B" ]; then
        JOBS+=("$arg")
    else
        echo "错误: 无法识别参数 '$arg'"
        echo "用法: $0 <A|B|序号...> [cuda:N]"
        exit 1
    fi
done

if [ ${#JOBS[@]} -eq 0 ]; then
    echo "错误: 请指定至少一个任务序号或 A/B"
    exit 1
fi

DEVICE_ARG="--device ${DEVICE}"
echo "使用设备: ${DEVICE}"
echo "任务列表: ${JOBS[*]}"

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

# 调度：按序号 / A / B 调用上面的 run_test_*
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