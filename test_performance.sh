#!/bin/bash

# 检查命令行参数
if [ $# -eq 0 ]; then
    echo "用法: $0 [A|B|1-15] [更多编号...]"
    echo "  A: 运行A组测试 (1-5)"
    echo "  B: 运行B组测试 (6-10)"
    echo "  1-15: 运行单个或多个测试序号 (支持逗号或空格分隔, 例如: '3,9,10' 或 '3 9 10')"
    exit 1
fi

GROUP=$1

# 检查是否为数字（1-11）
if [[ "$GROUP" =~ ^([1-9]|10|11|12|13|14|15)$ ]]; then
    echo "运行单个测试序号: $GROUP"
    SINGLE_TEST=true
else
    if [ "$GROUP" != "A" ] && [ "$GROUP" != "B" ]; then
        echo "错误: 参数必须是 A、B 或 1-15 中的数字"
        echo "用法: $0 [A|B|1-15] [更多编号...]"
        exit 1
    fi
    SINGLE_TEST=false
fi

# 定义测试函数
run_test_1() {
    echo "运行测试 1: D3QN nomask_4090_rl_filter_2025-07-25_15-02-16"
    list=(49600)
    for num in "${list[@]}"
    do
        python train.py --task Isaac-TaskAllocation-Direct-v1 --algo rl_filter --headless --wandb_activate --test --test_all_settings --other_filters \
            --load_dir "/rl_filter_2025-07-25_15-02-16/nn" --load_name "/HRTA_direct_ep_$num.pth" --wandb_project test_HRTA_fatigue --test_times 50
    done
}

run_test_2() {
    echo "运行测试 2: D3QN penalty_4070_rl_filter_2025-07-29_22-22-18"
    list=(49600)
    for num in "${list[@]}"
    do
        python train.py --task Isaac-TaskAllocation-Direct-v1 --algo rl_filter --headless --wandb_activate --test --test_all_settings --other_filters \
            --load_dir "/rl_filter_2025-07-29_22-22-18/nn" --load_name "/HRTA_direct_ep_$num.pth" --wandb_project test_HRTA_fatigue --test_times 50
    done
}

run_test_3() {
    echo "运行测试 3: PF-CD3Q 4070_rl_filter_2025-07-20_12-17-12"
    list=(
        # 49600
        # 44800
        # 46800
        # 52400
        # 54400
        # 54000
        # 53600
        # 53200
        # 52800
        52400
        52000
        51600
        51200
        50800
        50400
    )
    for num in "${list[@]}"
    do
        python train.py --task Isaac-TaskAllocation-Direct-v1 --algo rl_filter --headless --wandb_activate --test --use_fatigue_mask --test_all_settings --other_filters \
            --load_dir "/rl_filter_2025-07-20_12-17-12/nn" --load_name "/HRTA_direct_ep_$num.pth" --wandb_project test_HRTA_fatigue --test_times 50
    done
}

run_test_4() {
    echo "运行测试 4: mask_penalty_4090_rl_filter_2025-07-27_14-41-12"
    list=(49600)
    for num in "${list[@]}"
    do
        python train.py --task Isaac-TaskAllocation-Direct-v1 --algo rl_filter --headless --wandb_activate --test --use_fatigue_mask --test_all_settings --other_filters \
            --load_dir "/rl_filter_2025-07-27_14-41-12/nn" --load_name "/HRTA_direct_ep_$num.pth" --wandb_project test_HRTA_fatigue --test_times 50
    done
}

run_test_5() {
    echo "运行测试 5: DQN with penalty penalty_4070_dqn_2025-07-27_11-39-32"
    list=(49600)
    for num in "${list[@]}"
    do
        python train.py --task Isaac-TaskAllocation-Direct-v1 --algo dqn --headless --wandb_activate --test --test_all_settings --other_filters \
            --load_dir "/dqn_2025-07-27_11-39-32/nn" --load_name "/HRTA_direct_ep_$num.pth" --wandb_project test_HRTA_fatigue --test_times 50
    done
}

run_test_6() {
    echo "运行测试 6: PF-DQN 4090_dqn_2025-07-29_13-21-06"
    list=(49600)
    for num in "${list[@]}"
    do
        python train.py --task Isaac-TaskAllocation-Direct-v1 --algo dqn --headless --wandb_activate --test --use_fatigue_mask --test_all_settings --other_filters \
            --load_dir "/dqn_2025-07-29_13-21-06/nn" --load_name "/HRTA_direct_ep_$num.pth" --wandb_project test_HRTA_fatigue --test_times 50
    done
}

run_test_7() {
    echo "运行测试 7: PPO-dis with penalty 4070_penalty_ppo_dis_2025-07-31_13-37-58"
    list=(49600)
    for num in "${list[@]}"
    do
        python train.py --task Isaac-TaskAllocation-Direct-v1 --algo ppo_dis --headless --wandb_activate --test --test_all_settings --other_filters \
            --load_dir "/ppo_dis_2025-07-31_13-37-58/nn" --load_name "/HRTA_direct_ep_$num.pth" --wandb_project test_HRTA_fatigue --test_times 50
    done
}

run_test_8() {
    echo "运行测试 8: PF-PPO-dis 4090_ppo_dis_2025-07-30_13-18-07"
    list=(49600)
    for num in "${list[@]}"
    do
        python train.py --task Isaac-TaskAllocation-Direct-v1 --algo ppo_dis --headless --wandb_activate --test --use_fatigue_mask --test_all_settings --other_filters \
            --load_dir "/ppo_dis_2025-07-30_13-18-07/nn" --load_name "/HRTA_direct_ep_$num.pth" --wandb_project test_HRTA_fatigue --test_times 50
    done
}

run_test_9() {
    echo "运行测试 9: PPO-lag 4070_9_ppolag_filter_dis_2025-08-08_13-49-16"
    list=(
        # 49600
        # 52400
        # 52000
        # 51600
        # 51200
        # 50800
        # 50400
        49200
        48800
        48400
        48000
        47600
        47200
        46800
    )
    for num in "${list[@]}"
    do
        python train.py --task Isaac-TaskAllocation-Direct-v1 --algo ppolag_filter_dis --headless --wandb_activate --test --test_all_settings --other_filters \
            --load_dir "/ppolag_filter_dis_2025-08-08_13-49-16/nn" --load_name "/HRTA_direct_ep_$num.pth" --wandb_project test_HRTA_fatigue --test_times 50
    done
}

run_test_10() {
    echo "运行测试 10: PF-PPO-lag 4090_10_ppolag_filter_dis_2025-08-08_13-46-57"
    list=(49600)
    for num in "${list[@]}"
    do
        python train.py --task Isaac-TaskAllocation-Direct-v1 --algo ppolag_filter_dis --headless --wandb_activate --test --use_fatigue_mask --test_all_settings --other_filters \
            --load_dir "/ppolag_filter_dis_2025-08-08_13-46-57/nn" --load_name "/HRTA_direct_ep_$num.pth" --wandb_project test_HRTA_fatigue --test_times 50
    done
}

run_test_11() {
    echo "运行测试 11: CPO 4090_cpo_filter_2025-11-13_20-16-30_ep_400 4090"
    list=(400)
    for num in "${list[@]}"
    do
        python train.py --task Isaac-TaskAllocation-Direct-v1 --algo cpo_filter --headless --wandb_activate --test --test_all_settings \
            --load_dir "/cpo_filter_2025-11-13_20-16-30/nn" --load_name "/HRTA_direct_ep_$num.pth" --wandb_project test_HRTA_fatigue --test_times 50
    done
}

## ablation study ##

run_test_12() {
    echo "运行测试 12: rl_filter_no_noisy_2025-11-25_15-04-29 server"
    list=(49600)
    for num in "${list[@]}"
    do
        python train.py --task Isaac-TaskAllocation-Direct-v1 --algo rl_filter_no_noisy --headless --wandb_activate --test --use_fatigue_mask --test_all_settings \
            --load_dir "/rl_filter_no_noisy_2025-11-25_15-04-29/nn" --load_name "/HRTA_direct_ep_$num.pth" --wandb_project test_HRTA_fatigue --test_times 50
    done
}

run_test_13() {
    echo "运行测试 13: rl_filter_no_dueling_2025-11-29_18-11-35 4070"
    list=(
        49200
        48800
    )
    for num in "${list[@]}"
    do
        python train.py --task Isaac-TaskAllocation-Direct-v1 --algo rl_filter_no_dueling --headless --wandb_activate --test --use_fatigue_mask --test_all_settings \
            --load_dir "/rl_filter_no_dueling_2025-11-29_18-11-35/nn" --load_name "/HRTA_direct_ep_$num.pth" --wandb_project test_HRTA_fatigue --test_times 50
    done
}

run_test_14() {
    echo "运行测试 14: rl_filter_selfattn_2025-11-26_16-56-20 4070"
    list=(49600)
    for num in "${list[@]}"
    do
        python train.py --task Isaac-TaskAllocation-Direct-v1 --algo rl_filter_selfattn --headless --wandb_activate --test --use_fatigue_mask --test_all_settings \
            --load_dir "/rl_filter_selfattn_2025-11-26_16-56-20/nn" --load_name "/HRTA_direct_ep_$num.pth" --wandb_project test_HRTA_fatigue --test_times 50
    done
}

run_test_15() {
    echo "运行测试 15: rl_filter_mlp_2025-12-07_21-13-48 4090"
    list=(
        13650
    )
    for num in "${list[@]}"
    do
        python train.py --task Isaac-TaskAllocation-Direct-v1 --algo rl_filter_mlp --headless --wandb_activate --test --use_fatigue_mask --test_all_settings \
            --load_dir "/rl_filter_mlp_2025-12-07_21-13-48/nn" --load_name "/HRTA_direct_ep_$num.pth" --wandb_project test_HRTA_fatigue --test_times 50
    done
}


# 支持多个编号（逗号或空格分隔）
if [ "$GROUP" != "A" ] && [ "$GROUP" != "B" ]; then
    TEST_IDS=()
    for arg in "$@"; do
        IFS=',' read -ra parts <<< "$arg"
        for p in "${parts[@]}"; do
            if [[ "$p" =~ ^([1-9]|10|11|12|13|14|15)$ ]]; then
                TEST_IDS+=("$p")
            elif [ -n "$p" ]; then
                echo "错误: 无效的测试序号 $p"
                exit 1
            fi
        done
    done

    if [ ${#TEST_IDS[@]} -gt 0 ]; then
        echo "运行测试序号列表: ${TEST_IDS[*]}"
        for id in "${TEST_IDS[@]}"; do
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
            esac
        done
        echo "测试列表完成！"
        exit 0
    fi
fi

# 单个测试
if [ "$SINGLE_TEST" = true ]; then
    case $GROUP in
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
        *) echo "错误: 无效的测试序号 $GROUP" ;;
    esac
    echo "测试 $GROUP 完成！"
    exit 0
fi

# A组测试 (1-5)
if [ "$GROUP" = "A" ]; then
    echo "=== 运行A组测试 (1-5) ==="
    run_test_1
    run_test_2
    run_test_3
    run_test_11
    echo "A组测试完成！"
fi

# B组测试 (6-10)
if [ "$GROUP" = "B" ]; then
    echo "=== 运行B组测试 (6-10) ==="
    run_test_4
    run_test_5
    run_test_6
    run_test_7
    run_test_9
    run_test_10
    echo "B组测试完成！"
fi

echo "所有测试完成！"