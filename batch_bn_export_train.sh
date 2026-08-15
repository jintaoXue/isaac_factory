#!/bin/bash
# 把命令里选中的组合并成一份张量，再训一次（不是每组各自导出+训练）。
# 缺目录 / 缺 derived/ 的 run 跳过。文件夹名不带 FactoryBN。
#
# 组（只决定收哪些源文件夹）:
#   OLD  old_{machine,human,logistics,material}2.0
#   NORM new_norm10 + new_norm20
#   I1   new_{dim}1.0
#   I2   new_{dim}2.0
#   I3   new_{dim}3.0
#   ALL  上面五组
#
# 输出标签:
#   单组     old2.0 / norm / new1.0 / new2.0 / new3.0
#   多组     用 _ 拼接，如 norm_new1.0
#   ALL      all
#   → raw_data/<tag>/episodes.npz
#   → libcity/cache/model_cache/<tag>/BNPDFormer_best.pt
#
# 用法（仓库根目录，先 conda activate bn_pdformer）:
#   ./batch_bn_export_train.sh OLD
#   ./batch_bn_export_train.sh NORM I1
#   ./batch_bn_export_train.sh ALL
#   SKIP_EXISTING=1 ./batch_bn_export_train.sh OLD   # 已有 episodes.npz 则跳过导出，仍训练
#   SKIP_TRAIN=1    ./batch_bn_export_train.sh OLD   # 只导出
#   WANDB=1         ./batch_bn_export_train.sh OLD   # 打开 wandb
#
# 训练不要和 Isaac 采集抢同一张 GPU。
#
# tmux:
#   tmux new -s bn_train
#   conda activate bn_pdformer
#   cd ~/work/isaac_factory
#   ./batch_bn_export_train.sh OLD

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${ROOT}/source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/output/bottleneck_dataset"
PDFORMER="${ROOT}/source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/PDFormer"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
WANDB="${WANDB:-0}"
MAX_EPOCH="${MAX_EPOCH:-50}"
DEVICE="${DEVICE:-cuda}"
WINDOW_SIZE="${WINDOW_SIZE:-60}"

DIMS=(machine human logistics material)
ALL_GROUPS=(OLD NORM I1 I2 I3)

if [ $# -eq 0 ]; then
    echo "用法: $0 <OLD|NORM|I1|I2|I3|ALL> [OLD|NORM|I1 ...]"
    echo "  选中的组合并成一份 raw_data/<标签>，再训一次"
    echo "  OLD:  old_*2.0"
    echo "  NORM: new_norm10 / new_norm20"
    echo "  I1:   new_*1.0"
    echo "  I2:   new_*2.0"
    echo "  I3:   new_*3.0"
    echo "  ALL:  五组合并 → raw_data/all"
    echo "  多组例如 NORM I1 → raw_data/norm_new1.0"
    echo "  SKIP_EXISTING=1 跳过已有 episodes.npz 的导出；SKIP_TRAIN=1 只导出"
    echo "  WANDB=1 打开 wandb；MAX_EPOCH / DEVICE 可覆盖"
    exit 1
fi

WANT_ALL=0
JOBS=()
for arg in "$@"; do
    if [ "$arg" = "ALL" ]; then
        WANT_ALL=1
        JOBS+=("${ALL_GROUPS[@]}")
    elif [ "$arg" = "OLD" ] || [ "$arg" = "NORM" ] || [ "$arg" = "I1" ] || [ "$arg" = "I2" ] || [ "$arg" = "I3" ]; then
        JOBS+=("$arg")
    else
        echo "错误: 无法识别参数 '$arg'"
        echo "用法: $0 <OLD|NORM|I1|I2|I3|ALL>"
        exit 1
    fi
done

DEDUPED=()
seen=""
for id in "${JOBS[@]}"; do
    case " $seen " in
        *" $id "*) continue ;;
    esac
    seen="${seen} ${id}"
    DEDUPED+=("$id")
done
JOBS=("${DEDUPED[@]}")

if [ "$WANT_ALL" = "1" ] || [ "${JOBS[*]}" = "${ALL_GROUPS[*]}" ]; then
    JOBS=("${ALL_GROUPS[@]}")
    TAG="all"
else
    TAG=""
    for id in "${JOBS[@]}"; do
        case $id in
            OLD) part="old2.0" ;;
            NORM) part="norm" ;;
            I1) part="new1.0" ;;
            I2) part="new2.0" ;;
            I3) part="new3.0" ;;
            *) part="$id" ;;
        esac
        if [ -z "$TAG" ]; then
            TAG="$part"
        else
            TAG="${TAG}_${part}"
        fi
    done
fi

group_runs() {
    local id=$1
    local d
    case $id in
        OLD)
            for d in "${DIMS[@]}"; do echo "old_${d}2.0"; done
            ;;
        NORM)
            echo "new_norm10"
            echo "new_norm20"
            ;;
        I1)
            for d in "${DIMS[@]}"; do echo "new_${d}1.0"; done
            ;;
        I2)
            for d in "${DIMS[@]}"; do echo "new_${d}2.0"; done
            ;;
        I3)
            for d in "${DIMS[@]}"; do echo "new_${d}3.0"; done
            ;;
    esac
}

usable_run() {
    local name=$1
    local run_dir="${DATA_ROOT}/${name}"
    if [ ! -d "$run_dir" ]; then
        echo "[$(date '+%F %T')] 跳过 ${name}：目录不存在" >&2
        return 1
    fi
    if [ ! -d "${run_dir}/derived" ]; then
        echo "[$(date '+%F %T')] 跳过 ${name}：没有 derived/（先跑 ./batch_bn_agg.sh）" >&2
        return 1
    fi
    return 0
}

OUT_DIR="${PDFORMER}/raw_data/${TAG}"
SAVE_DIR="${PDFORMER}/libcity/cache/model_cache/${TAG}"

echo "工作目录: ${ROOT}"
echo "选中组: ${JOBS[*]}  →  标签 ${TAG}"
echo "SKIP_EXISTING=${SKIP_EXISTING} SKIP_TRAIN=${SKIP_TRAIN} WANDB=${WANDB} DEVICE=${DEVICE} MAX_EPOCH=${MAX_EPOCH}"

RUN_ARGS=()
echo "将合并导出的 run:"
for id in "${JOBS[@]}"; do
    while IFS= read -r name; do
        [ -n "$name" ] || continue
        if usable_run "$name"; then
            RUN_ARGS+=(--run_dir "${DATA_ROOT}/${name}")
            echo "  + ${name}"
        fi
    done < <(group_runs "$id")
done

if [ ${#RUN_ARGS[@]} -eq 0 ]; then
    echo "[$(date '+%F %T')] 没有任何可导出的 run，退出"
    exit 1
fi

echo "========================================"
echo "[$(date '+%F %T')] 合并导出  tag=${TAG}  n_run=$(( ${#RUN_ARGS[@]} / 2 ))"
echo "========================================"

if [ "$SKIP_EXISTING" = "1" ] && [ -f "${OUT_DIR}/episodes.npz" ]; then
    echo "[$(date '+%F %T')] 跳过导出 ${TAG}：已有 episodes.npz"
else
    (
        cd "${PDFORMER}"
        python -m factory_bn.export_dataset \
            "${RUN_ARGS[@]}" \
            --out_dir "${OUT_DIR}" \
            --window_size "${WINDOW_SIZE}"
    )
    echo "[$(date '+%F %T')] 导出完成  ${OUT_DIR}/episodes.npz"
fi

if [ "$SKIP_TRAIN" = "1" ]; then
    echo "[$(date '+%F %T')] SKIP_TRAIN=1，不训练 ${TAG}"
    echo "张量: ${OUT_DIR}/episodes.npz"
    exit 0
fi
if [ ! -f "${OUT_DIR}/episodes.npz" ]; then
    echo "[$(date '+%F %T')] 没有 ${OUT_DIR}/episodes.npz，跳过训练"
    exit 1
fi

echo "========================================"
echo "[$(date '+%F %T')] 训练  ${TAG}  epoch=${MAX_EPOCH}  device=${DEVICE}"
echo "========================================"
WANDB_ARGS=()
if [ "$WANDB" = "1" ]; then
    WANDB_ARGS+=(--wandb_activate --wandb_project FactoryBN_PDFormer --wandb_name "${TAG}_ep${MAX_EPOCH}")
fi
(
    cd "${PDFORMER}"
    python -m factory_bn.train \
        --config factory_bn/configs/FactoryBN.json \
        --data_dir "${OUT_DIR}" \
        --save_dir "${SAVE_DIR}" \
        --max_epoch "${MAX_EPOCH}" \
        --device "${DEVICE}" \
        "${WANDB_ARGS[@]}"
)
echo "[$(date '+%F %T')] 训练完成  ${SAVE_DIR}/BNPDFormer_best.pt"
echo "张量: ${OUT_DIR}/episodes.npz"
echo "权重: ${SAVE_DIR}/BNPDFormer_best.pt"
