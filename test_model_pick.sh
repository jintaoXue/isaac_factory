#!/usr/bin/env bash
set -euo pipefail

# 用法:
#   bash test_pick.sh [-a|--asc | -d|--desc] [模型目录]
# 说明:
#   -a/--asc   按编号正序运行(默认)
#   -d/--desc  按编号倒序运行
#   模型目录   可选，默认为 DEFAULT_DIR
# 功能: 提取目录下所有 HRTA_direct_ep_*.pth 的编号, 过滤 <15000 以及已完成编号, 并逐个运行测试

# 默认目录(可由命令行参数覆盖)
DEFAULT_DIR="/home/xue/work/Isaac-Production/logs/rl_games/HRTA_direct/rl_filter_2025-07-20_12-17-12/nn"
ORDER="asc"   # asc 或 desc
DIR="$DEFAULT_DIR"

# 解析参数
DIR_SET=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        -a|--asc)
            ORDER="asc"
            shift
            ;;
        -d|--desc)
            ORDER="desc"
            shift
            ;;
        -h|--help)
            echo "用法: bash $0 [-a|--asc | -d|--desc] [模型目录]"
            exit 0
            ;;
        *)
            if [[ $DIR_SET -eq 0 ]]; then
                DIR="$1"
                DIR_SET=1
                shift
            else
                echo "错误: 多余参数: $1"
                exit 1
            fi
            ;;
    esac
done

if [ ! -d "$DIR" ]; then
    echo "错误: 目录不存在: $DIR"
    exit 1
fi

SORT_OPT="-n"
if [[ "$ORDER" == "desc" ]]; then
    SORT_OPT="-nr"
fi

echo "扫描目录: $DIR"
echo "排序方式: $( [[ "$ORDER" == "asc" ]] && echo 正序 || echo 倒序 )"

# 收集所有符合命名的模型编号
mapfile -t all_nums < <( \
    find "$DIR" -maxdepth 1 -type f -name 'HRTA_direct_ep_*.pth' \
    | sed -E 's/.*HRTA_direct_ep_([0-9]+)\.pth/\1/' \
    | sort ${SORT_OPT} | uniq \
)

if [ ${#all_nums[@]} -eq 0 ]; then
    echo "未找到任何模型文件: HRTA_direct_ep_*.pth"
    exit 0
fi

# 过滤 <15000
filtered_nums=()
for n in "${all_nums[@]}"; do
    if [ "$n" -ge 15000 ]; then
        filtered_nums+=("$n")
    fi
done

# 过滤 done_list 中的编号
# 已测试过的模型编号(填入数字, 用空格分隔)
done_list=(
    49600
    44800
    46800
    52400
    54400
    54000
    53600
    53200
    52800
    52400
    52000
    51600
    51200
    50800
    50400
)

is_done() {
    local val=$1
    for d in "${done_list[@]:-}"; do
        if [ "$d" = "$val" ]; then
            return 0
        fi
    done
    return 1
}

to_run=()
for n in "${filtered_nums[@]}"; do
    if ! is_done "$n"; then
        to_run+=("$n")
    fi
done

if [ ${#to_run[@]} -eq 0 ]; then
    echo "没有需要运行的模型编号 (均已过滤或已完成)"
    exit 0
fi

echo "将要运行的模型编号: ${to_run[*]}"

# 逐个运行
for num in "${to_run[@]}"; do
    echo "运行模型: HRTA_direct_ep_${num}.pth"
    python train.py \
        --task Isaac-TaskAllocation-Direct-v1 \
        --algo rl_filter \
        --headless \
        --wandb_activate \
        --test \
        --test_all_settings \
        --other_filters \
        --load_dir "/rl_filter_2025-07-20_12-17-12/nn" \
        --load_name "/HRTA_direct_ep_${num}.pth" \
        --wandb_project test_HRTA_fatigue \
        --test_times 50
done

echo "全部运行完成。"
