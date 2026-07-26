#!/bin/bash

# 用法:
#   ./batch_train.sh 19 cuda:0
#   ./batch_train.sh 19 20 cuda:0          # 依次跑多个序号
#   ./batch_train.sh A cuda:0
#   ./batch_train.sh B
#   ./batch_train.sh P cuda:0              # ICCBEI 论文实验全流程
#   ./batch_train.sh 30 cuda:0             # 仅 TPA grid
if [ $# -eq 0 ]; then
    echo "用法: $0 <A|B|P|序号...> [cuda:N]"
    echo "  A: 运行A组训练 (1-5)"
    echo "  B: 运行B组训练 (6-10)"
    echo "  P: ICCBEI paper_exp 全流程 (30-39)"
    echo "  1-17: RL / HcFactory 训练序号（可多个，如 19 20）"
    echo "  18-21: Perception 旧入口（baseline）"
    echo "  30-39: ICCBEI paper_exp（见下方注释）"
    echo "  cuda:N: 可选，指定CUDA设备，默认 cuda:0（写在最后）"
    exit 1
fi

DEVICE="cuda:0"
JOBS=()
for arg in "$@"; do
    if [[ "$arg" =~ ^cuda:[0-9]+$ ]]; then
        DEVICE="$arg"
    elif [[ "$arg" =~ ^([1-9]|[1-3][0-9]|A|B|P)$ ]] || [ "$arg" = "A" ] || [ "$arg" = "B" ] || [ "$arg" = "P" ]; then
        JOBS+=("$arg")
    else
        echo "错误: 无法识别参数 '$arg'"
        echo "用法: $0 <A|B|P|序号...> [cuda:N]"
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

##### ICCBEI paper_exp (main.tex Experiments) #####
# 原始结果根目录：source/.../hc_factory/output/paper_exp/
#   datasets/Nh{h}_O{o}/     感知采集
#   tpa/Nh{h}_O{o}/          makespan/idle JSONL
#   perception_runs/         checkpoints + history.json
#   metrics/                 eval JSON（供 ICCBEI 画图）
PAPER_EXP_ROOT="source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/output/paper_exp"
PAPER_DATASETS="${PAPER_EXP_ROOT}/datasets"
PAPER_TPA="${PAPER_EXP_ROOT}/tpa"
PAPER_RUNS="${PAPER_EXP_ROOT}/perception_runs"
PAPER_METRICS="${PAPER_EXP_ROOT}/metrics"
SOURCE_DS="${PAPER_DATASETS}/Nh5_O5"
AGG_PY="source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/scripts/aggregate_paper_exp.py"

_paper_mkdir() {
    mkdir -p "${PAPER_DATASETS}" "${PAPER_TPA}" "${PAPER_RUNS}" "${PAPER_METRICS}"
}

run_test_30() {
    # TPA scalability：Nh×O grid，无相机，记 makespan / idle
    echo "运行 30: Paper TPA scalability grid (Nh=1..5, O=1..5)"
    _paper_mkdir
    for nh in 1 2 3 4 5; do
        for o in 1 2 3 4 5; do
            cell="Nh${nh}_O${o}"
            echo "=== TPA cell ${cell} ==="
            python train.py \
                --task HRTPaHC-v1 \
                --algo rule_based \
                --num_envs 1 \
                --headless \
                --disable_perception \
                --num_humans "${nh}" \
                --product_order "${o}" \
                --max_episodes 3 \
                --tpa_metrics_dir "${PAPER_TPA}/${cell}" \
                ${DEVICE_ARG}
        done
    done
}

run_test_31() {
    # 源设置感知采集 (5,5)，6 episodes → train/val/test = 4/1/1
    echo "运行 31: Paper collect source Nh5_O5 (6 episodes, cameras on)"
    _paper_mkdir
    python train.py \
        --task HRTPaHC-v1 \
        --algo rule_based \
        --num_envs 1 \
        --headless \
        --enable_cameras \
        --num_humans 5 \
        --product_order 5 \
        --max_episodes 6 \
        --perception_max_episodes 6 \
        --perception_output_dir "${SOURCE_DS}" \
        --tpa_metrics_dir "${PAPER_TPA}/Nh5_O5" \
        ${DEVICE_ARG}
}

run_test_32() {
    # OOD 测试采集：每个 (Nh,O) 1 个 episode（含源设置额外 test 也可复用 31）
    echo "运行 32: Paper collect OOD grid (1 episode / cell, cameras on)"
    _paper_mkdir
    for nh in 1 2 3 4 5; do
        for o in 1 2 3 4 5; do
            cell="Nh${nh}_O${o}"
            # 源设置已有 6 ep；仍补 1 个独立 OOD 评测 episode 到同目录或单独 ood_ 前缀
            out="${PAPER_DATASETS}/${cell}"
            if [ "${nh}" -eq 5 ] && [ "${o}" -eq 5 ]; then
                echo "=== skip collect ${cell} (use job 31 source); will eval on its test split ==="
                continue
            fi
            echo "=== collect OOD ${cell} ==="
            python train.py \
                --task HRTPaHC-v1 \
                --algo rule_based \
                --num_envs 1 \
                --headless \
                --enable_cameras \
                --num_humans "${nh}" \
                --product_order "${o}" \
                --max_episodes 1 \
                --perception_max_episodes 1 \
                --perception_output_dir "${out}" \
                --tpa_metrics_dir "${PAPER_TPA}/${cell}" \
                ${DEVICE_ARG}
        done
    done
}

run_test_33() {
    # Learning curve Task A (id)：25/50/75/100%
    echo "运行 33: Paper learning-curve train human-id"
    _paper_mkdir
    for frac in 0.25 0.50 0.75 1.0; do
        tag=$(python3 - <<PY
f=float("${frac}")
print(f"curve_id_f{int(round(f*100)):03d}")
PY
)
        echo "=== train ${tag} ==="
        python "${PERCEPTION_PY}" train \
            --task id \
            --dataset_dir "${SOURCE_DS}" \
            --output_dir "${PAPER_RUNS}" \
            --run_name "${tag}" \
            --train_fraction "${frac}" \
            --epochs 20 \
            --batch_size 32 \
            ${DEVICE_ARG}
    done
}

run_test_34() {
    # Learning curve Task B (subtask)
    echo "运行 34: Paper learning-curve train human-subtask"
    _paper_mkdir
    for frac in 0.25 0.50 0.75 1.0; do
        tag=$(python3 - <<PY
f=float("${frac}")
print(f"curve_subtask_f{int(round(f*100)):03d}")
PY
)
        echo "=== train ${tag} ==="
        python "${PERCEPTION_PY}" train \
            --task subtask \
            --dataset_dir "${SOURCE_DS}" \
            --output_dir "${PAPER_RUNS}" \
            --run_name "${tag}" \
            --train_fraction "${frac}" \
            --epochs 20 \
            --batch_size 32 \
            ${DEVICE_ARG}
    done
}

run_test_35() {
    # 全量源设置 checkpoint（与 curve f100 同设定，固定命名便于后续 eval）
    echo "运行 35: Paper full source train id+subtask"
    _paper_mkdir
    python "${PERCEPTION_PY}" train \
        --task id \
        --dataset_dir "${SOURCE_DS}" \
        --output_dir "${PAPER_RUNS}" \
        --run_name "source_full" \
        --train_fraction 1.0 \
        --epochs 20 \
        --batch_size 32 \
        ${DEVICE_ARG}
    python "${PERCEPTION_PY}" train \
        --task subtask \
        --dataset_dir "${SOURCE_DS}" \
        --output_dir "${PAPER_RUNS}" \
        --run_name "source_full" \
        --train_fraction 1.0 \
        --epochs 20 \
        --batch_size 32 \
        ${DEVICE_ARG}
}

run_test_36() {
    # Ablation: w/o process-task embedding
    echo "运行 36: Paper ablate no task embedding (subtask)"
    _paper_mkdir
    python "${PERCEPTION_PY}" train \
        --task subtask \
        --dataset_dir "${SOURCE_DS}" \
        --output_dir "${PAPER_RUNS}" \
        --run_name "ablate_no_taskemb" \
        --no_task_embedding \
        --train_fraction 1.0 \
        --epochs 20 \
        --batch_size 32 \
        ${DEVICE_ARG}
}

run_test_37() {
    # In-distribution eval + Task B diagnostics
    echo "运行 37: Paper ID eval + detailed subtask diagnostics"
    _paper_mkdir
    python "${PERCEPTION_PY}" eval \
        --task id \
        --dataset_dir "${SOURCE_DS}" \
        --checkpoint "${PAPER_RUNS}/source_full_id/best.pt" \
        --split test \
        --metrics_out "${PAPER_METRICS}/id_source_test.json" \
        ${DEVICE_ARG}
    python "${PERCEPTION_PY}" eval \
        --task subtask \
        --dataset_dir "${SOURCE_DS}" \
        --checkpoint "${PAPER_RUNS}/source_full_subtask/best.pt" \
        --split test \
        --detailed \
        --metrics_out "${PAPER_METRICS}/subtask_source_test_detailed.json" \
        ${DEVICE_ARG}
    python "${PERCEPTION_PY}" eval \
        --task subtask \
        --dataset_dir "${SOURCE_DS}" \
        --checkpoint "${PAPER_RUNS}/ablate_no_taskemb_subtask/best.pt" \
        --split test \
        --metrics_out "${PAPER_METRICS}/ablate_no_taskemb_test.json" \
        ${DEVICE_ARG}
    # learning-curve test points
    for frac in 025 050 075 100; do
        python "${PERCEPTION_PY}" eval \
            --task id \
            --dataset_dir "${SOURCE_DS}" \
            --checkpoint "${PAPER_RUNS}/curve_id_f${frac}_id/best.pt" \
            --split test \
            --metrics_out "${PAPER_METRICS}/curve_id_f${frac}_test.json" \
            ${DEVICE_ARG} || true
        python "${PERCEPTION_PY}" eval \
            --task subtask \
            --dataset_dir "${SOURCE_DS}" \
            --checkpoint "${PAPER_RUNS}/curve_subtask_f${frac}_subtask/best.pt" \
            --split test \
            --metrics_out "${PAPER_METRICS}/curve_subtask_f${frac}_test.json" \
            ${DEVICE_ARG} || true
    done
}

run_test_38() {
    # Cross-setting OOD eval（固定 source_full checkpoint）
    echo "运行 38: Paper OOD eval grid"
    _paper_mkdir
    for nh in 1 2 3 4 5; do
        for o in 1 2 3 4 5; do
            cell="Nh${nh}_O${o}"
            ds="${PAPER_DATASETS}/${cell}"
            split="all"
            if [ "${nh}" -eq 5 ] && [ "${o}" -eq 5 ]; then
                ds="${SOURCE_DS}"
                split="test"
            fi
            if [ ! -d "${ds}" ]; then
                echo "[WARN] missing dataset ${ds}, skip"
                continue
            fi
            python "${PERCEPTION_PY}" eval \
                --task id \
                --dataset_dir "${ds}" \
                --checkpoint "${PAPER_RUNS}/source_full_id/best.pt" \
                --split "${split}" \
                --metrics_out "${PAPER_METRICS}/ood_id_${cell}.json" \
                ${DEVICE_ARG} || true
            python "${PERCEPTION_PY}" eval \
                --task subtask \
                --dataset_dir "${ds}" \
                --checkpoint "${PAPER_RUNS}/source_full_subtask/best.pt" \
                --split "${split}" \
                --metrics_out "${PAPER_METRICS}/ood_subtask_${cell}.json" \
                ${DEVICE_ARG} || true
        done
    done
}

run_test_39() {
    echo "运行 39: Aggregate paper_exp metrics"
    python "${AGG_PY}"
}

# 调度：按序号 / A / B / P 调用上面的 run_test_*
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
        30) run_test_30 ;;
        31) run_test_31 ;;
        32) run_test_32 ;;
        33) run_test_33 ;;
        34) run_test_34 ;;
        35) run_test_35 ;;
        36) run_test_36 ;;
        37) run_test_37 ;;
        38) run_test_38 ;;
        39) run_test_39 ;;
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
        P)
            echo "=== ICCBEI paper_exp (30-39) ==="
            run_test_30
            run_test_31
            run_test_32
            run_test_33
            run_test_34
            run_test_35
            run_test_36
            run_test_37
            run_test_38
            run_test_39
            echo "paper_exp 完成！原始结果在 ${PAPER_EXP_ROOT}"
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