#!/bin/bash
# 瓶颈数据集 Stage-C 聚合：bottleneck_dataset 下每个 run 文件夹一条。
# 不占 GPU，可与 Isaac 采集并行（建议另开 tmux）。
#
# 文件夹约定（采集结束后请把时间戳目录改成这些名字）:
#   old_{machine,human,logistics,material}2.0
#   norm                         # 无扰动对照（只聚合，不进训练）
#   new_{machine,human,logistics,material}{1.0,2.0,3.0}
#
# 用法（仓库根目录；python 能 import bn_agg 即可，推荐 conda activate env_isaaclab）:
#   ./batch_bn_agg.sh OLD          # 旧四维 I=2.0
#   ./batch_bn_agg.sh NORM         # norm（对照）
#   ./batch_bn_agg.sh I1           # 新四维 I=1.0（缺目录则跳过）
#   ./batch_bn_agg.sh I2           # 新四维 I=2.0
#   ./batch_bn_agg.sh I3           # 新四维 I=3.0
#   ./batch_bn_agg.sh ALL          # OLD → NORM → I1 → I2 → I3
#   ./batch_bn_agg.sh old_human2.0 new_machine1.0   # 指定文件夹
#
# 已有 derived/ 想跳过: SKIP_EXISTING=1 ./batch_bn_agg.sh OLD
#
# tmux:
#   tmux new -s bn_agg
#   conda activate env_isaaclab
#   cd ~/work/isaac_factory
#   ./batch_bn_agg.sh OLD
# 采集还在跑、要边关窗边聚:
#   python -m bn_agg --run_dir .../output/bottleneck_dataset/<run_id> --follow

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${ROOT}/source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/output/bottleneck_dataset"
TOOLS_DIR="${ROOT}/source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/tools"
SKIP_EXISTING="${SKIP_EXISTING:-0}"

DIMS=(machine human logistics material)

if [ $# -eq 0 ]; then
    echo "用法: $0 <OLD|NORM|I1|I2|I3|ALL|文件夹名> [I1|I2|...|文件夹名 ...]"
    echo "  OLD: old_machine2.0 / old_human2.0 / old_logistics2.0 / old_material2.0"
    echo "  NORM: norm（无扰动对照，只聚合不训练）"
    echo "  I1:  new_{dim}1.0"
    echo "  I2:  new_{dim}2.0"
    echo "  I3:  new_{dim}3.0"
    echo "  ALL: OLD → NORM → I1 → I2 → I3（缺目录跳过）"
    echo "  也可直接写文件夹名，如 old_human2.0"
    echo "  SKIP_EXISTING=1 时跳过已有 derived/ 的 run"
    exit 1
fi

expand_job() {
    local id=$1
    case $id in
        OLD)
            local d
            for d in "${DIMS[@]}"; do
                echo "old_${d}2.0"
            done
            ;;
        NORM)
            echo "norm"
            ;;
        I1)
            local d
            for d in "${DIMS[@]}"; do
                echo "new_${d}1.0"
            done
            ;;
        I2)
            local d
            for d in "${DIMS[@]}"; do
                echo "new_${d}2.0"
            done
            ;;
        I3)
            local d
            for d in "${DIMS[@]}"; do
                echo "new_${d}3.0"
            done
            ;;
        ALL)
            expand_job OLD
            expand_job NORM
            expand_job I1
            expand_job I2
            expand_job I3
            ;;
        *)
            echo "$id"
            ;;
    esac
}

RUNS=()
for arg in "$@"; do
    while IFS= read -r name; do
        [ -n "$name" ] || continue
        RUNS+=("$name")
    done < <(expand_job "$arg")
done

# 去重、保序
DEDUPED=()
seen=""
for name in "${RUNS[@]}"; do
    case " $seen " in
        *" $name "*) continue ;;
    esac
    seen="${seen} ${name}"
    DEDUPED+=("$name")
done
RUNS=("${DEDUPED[@]}")

echo "工作目录: ${ROOT}"
echo "数据根: ${DATA_ROOT}"
echo "SKIP_EXISTING=${SKIP_EXISTING}"
echo "计划: ${RUNS[*]}"

agg_one() {
    local name=$1
    local run_dir="${DATA_ROOT}/${name}"

    if [ ! -d "$run_dir" ]; then
        echo "[$(date '+%F %T')] 跳过 ${name}：目录不存在（采集未完成或尚未改名）"
        return 0
    fi
    if [ ! -d "${run_dir}/episode_00" ] && [ ! -d "${run_dir}/env_00" ]; then
        echo "[$(date '+%F %T')] 跳过 ${name}：没有 episode_*/ 或 env_00/"
        return 0
    fi
    if [ "$SKIP_EXISTING" = "1" ] && [ -d "${run_dir}/derived" ]; then
        echo "[$(date '+%F %T')] 跳过 ${name}：已有 derived/"
        return 0
    fi

    echo "========================================"
    echo "[$(date '+%F %T')] 开始聚合  ${name}"
    echo "========================================"
    (
        cd "${TOOLS_DIR}"
        PYTHONPATH=. python3 -m bn_agg \
            --run_dir "${run_dir}" \
            --window_sizes 30,60 \
            --horizon 180 \
            --score_threshold 0.55 \
            --min_event_windows 1
    )
    echo "[$(date '+%F %T')] 完成聚合  ${name}"
}

n_ok=0
n_skip=0
for name in "${RUNS[@]}"; do
    before=$(date +%s)
    if [ ! -d "${DATA_ROOT}/${name}" ]; then
        agg_one "$name"
        n_skip=$((n_skip + 1))
        continue
    fi
    agg_one "$name"
    after=$(date +%s)
    if [ -d "${DATA_ROOT}/${name}/derived" ]; then
        n_ok=$((n_ok + 1))
        echo "  耗时 $((after - before))s"
    else
        n_skip=$((n_skip + 1))
    fi
done

echo "[$(date '+%F %T')] 全部结束  已聚合=${n_ok}  跳过=${n_skip}"
echo "输出在各 run 的 derived/ 下；下一步导出见 00.运行方式.md"
