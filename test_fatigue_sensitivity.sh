#!/bin/bash

set -euo pipefail

usage() {
    cat <<'EOF'
用法: ./test_fatigue_sensitivity.sh [选项]

选项:
  --algo <名称>    仅运行指定算法，可重复。可选值: 3, pf-cd3q, 9, ppo-lag
  --ftg <数值>     仅测试指定疲劳阈值，可重复
  --ftg-range <起始> <终止> <步长>  指定连续疲劳阈值范围 (包含端点)
  -h, --help       显示本帮助

默认：遍历全部疲劳阈值 (0.0~1.0, 步长0.05) 并运行所有算法。
EOF
}

FTG_VALUES=($(python - <<'PY'
vals = [f"{i/20:.2f}".rstrip('0').rstrip('.') for i in range(21)]
print(" ".join(vals))
PY
))

declare -a SELECTED_FTGS=()
declare -a SELECTED_ALGOS=()
declare -a FTG_RANGE=()
declare -a RANGE_FTG_VALUES=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --algo)
            [[ $# -lt 2 ]] && { echo "错误: --algo 需要参数" >&2; exit 1; }
            SELECTED_ALGOS+=("$2")
            shift 2
            ;;
        --ftg)
            [[ $# -lt 2 ]] && { echo "错误: --ftg 需要参数" >&2; exit 1; }
            SELECTED_FTGS+=("$2")
            shift 2
            ;;
        --ftg-range)
            if [[ $# -lt 4 ]]; then
                echo "错误: --ftg-range 需要三个参数 (起始 终止 步长)" >&2
                exit 1
            fi
            FTG_RANGE=("$2" "$3" "$4")
            shift 4
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "未知参数: $1" >&2
            usage
            exit 1
            ;;
    esac
done

declare -A ALGO_MAP=(
    ["3"]="run_test_3"
    ["pf-cd3q"]="run_test_3"
    ["9"]="run_test_9"
    ["ppo-lag"]="run_test_9"
)

declare -a ACTIVE_TESTS=()
if [[ ${#SELECTED_ALGOS[@]} -eq 0 ]]; then
    ACTIVE_TESTS=("run_test_3" "run_test_9")
else
    for name in "${SELECTED_ALGOS[@]}"; do
        key="${name,,}"
        if [[ -z "${ALGO_MAP[$key]+x}" ]]; then
            echo "未知算法: ${name}" >&2
            echo "可选: 3, pf-cd3q, 9, ppo-lag" >&2
            exit 1
        fi
        ACTIVE_TESTS+=("${ALGO_MAP[$key]}")
    done
fi

if [[ ${#FTG_RANGE[@]} -eq 3 ]]; then
    RANGE_FTG_VALUES=($(python - <<PY
start = float("${FTG_RANGE[0]}")
end = float("${FTG_RANGE[1]}")
step = float("${FTG_RANGE[2]}")
if step <= 0:
    raise ValueError("步长必须为正数")
values = []
current = start
epsilon = 1e-8
while current <= end + epsilon:
    values.append(f"{current:.3f}".rstrip('0').rstrip('.'))
    current += step
print(" ".join(values))
PY
))
fi

describe_algo() {
    case "$1" in
        run_test_3) echo "PF-CD3Q" ;;
        run_test_9) echo "PPO-lag" ;;
        *) echo "$1" ;;
    esac
}

if [[ ${#SELECTED_FTGS[@]} -eq 0 ]]; then
    if [[ ${#RANGE_FTG_VALUES[@]} -gt 0 ]]; then
        SELECTED_FTGS=("${RANGE_FTG_VALUES[@]}")
    else
        SELECTED_FTGS=("${FTG_VALUES[@]}")
    fi
else
    if [[ ${#RANGE_FTG_VALUES[@]} -gt 0 ]]; then
        SELECTED_FTGS+=("${RANGE_FTG_VALUES[@]}")
    fi
fi

run_test_3() {
    local ftg_value=$1
    local checkpoints=(49600)
    echo "运行测试 3: PF-CD3Q | ftg_thresh_phy=${ftg_value}"
    for num in "${checkpoints[@]}"; do
        python train.py \
            --task Isaac-TaskAllocation-Direct-v1 \
            --algo rl_filter \
            --headless \
            --wandb_activate \
            --test \
            --test_all_settings \
            --use_fatigue_mask \
            --other_filters \
            --load_dir "/rl_filter_2025-07-20_12-17-12/nn" \
            --load_name "/HRTA_direct_ep_${num}.pth" \
            --wandb_project test_ftg_sensitivity \
            --test_times 50 \
            --ftg_thresh_phy "${ftg_value}"
    done
}

run_test_9() {
    local ftg_value=$1
    local checkpoints=(49600)
    echo "运行测试 9: PPO-lag | ftg_thresh_phy=${ftg_value}"
    for num in "${checkpoints[@]}"; do
        python train.py \
            --task Isaac-TaskAllocation-Direct-v1 \
            --algo ppolag_filter_dis \
            --headless \
            --wandb_activate \
            --test \
            --test_all_settings \
            --other_filters \
            --load_dir "/ppolag_filter_dis_2025-08-08_13-49-16/nn" \
            --load_name "/HRTA_direct_ep_${num}.pth" \
            --wandb_project test_ftg_sensitivity \
            --test_times 10 \
            --ftg_thresh_phy "${ftg_value}"
    done
}

for test_fn in "${ACTIVE_TESTS[@]}"; do
    algo_name="$(describe_algo "${test_fn}")"
    echo "===== 开始算法 ${algo_name} ====="
    for ftg in "${SELECTED_FTGS[@]}"; do
        echo "--- 开始疲劳阈值 ${ftg} (${algo_name}) ---"
        "${test_fn}" "${ftg}"
        echo "--- 完成疲劳阈值 ${ftg} (${algo_name}) ---"
    done
    echo "===== 完成算法 ${algo_name} ====="
done

echo "所有疲劳敏感性测试完成！"