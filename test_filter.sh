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
    echo "运行测试 1: PF-CD3Q rl_filter_2025-07-20_12-17-12"
    list=(49600)
    for num in "${list[@]}"
    do
        python train.py --task Isaac-TaskAllocation-Direct-v1 --algo rl_filter --headless --test --test_all_settings --other_filters --use_fatigue_mask \
            --load_dir "/rl_filter_2025-07-20_12-17-12/nn" --load_name "/HRTA_direct_ep_$num.pth" --test_times 10 --wandb_activate \
            --wandb_project test_filter_latency 
     done
}


# 定义测试函数
run_test_2() {
    echo "运行测试 2: PF-CD3Q rl_filter_2025-07-20_12-17-12，num_particles=100~1000"
    list=(49600)
    particles=(100 200 300 400 500 600 700 800 900 1000)

    for num in "${list[@]}"; do
        for n_part in "${particles[@]}"; do
            echo "  -> num_particles = $n_part"
            python train.py \
                --task Isaac-TaskAllocation-Direct-v1 \
                --algo rl_filter \
                --headless \
                --test \
                --use_fatigue_mask \
                --num_particles "$n_part" \
                --load_dir "/rl_filter_2025-07-20_12-17-12/nn" \
                --load_name "/HRTA_direct_ep_$num.pth" \
                --test_times 10 \
                --test_all_settings \
                --wandb_activate \
                --wandb_project test_filter_latency
        done
    done
}


# noisy test filter
run_test_3() {
    echo "运行测试 3: PF-CD3Q rl_filter_2025-07-20_12-17-12，measure_noise_sigma 从 0.1 -> 0.00005"
    list=(49600)
    # 噪声从 0.1 开始，每次 x0.1，直到 0.00005（最后一项单独补）
    noise_list=(0.1 0.01 0.001 0.0001 0.00005)

    for num in "${list[@]}"; do
        for noise in "${noise_list[@]}"; do
            echo "  -> measure_noise_sigma = $noise"
            python train.py \
                --task Isaac-TaskAllocation-Direct-v1 \
                --algo rl_filter \
                --headless \
                --test \
                --use_fatigue_mask \
                --other_filters \
                --test_all_settings \
                --measure_noise_sigma "$noise" \
                --load_dir "/rl_filter_2025-07-20_12-17-12/nn" \
                --load_name "/HRTA_direct_ep_$num.pth" \
                --test_times 50 \
                --wandb_activate \
                --wandb_project test_noisy
        done
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