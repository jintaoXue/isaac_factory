#!/bin/bash
# 瓶颈扰动采集：I1/I2/I3 各四维，另加 NORM 无扰动对照。
# 同一强度四维串行（一组）。只有一张卡（cuda:0），不要并行多组。
#
# 用法（仓库根目录，先 conda activate env_isaaclab）:
#   ./batch_bn_collect.sh I1
#   ./batch_bn_collect.sh I2
#   ./batch_bn_collect.sh I3
#   ./batch_bn_collect.sh NORM   # dim=none 对照；采完把时间戳目录改名为 norm
#   ./batch_bn_collect.sh ALL    # I1 → I2 → I3（12 个 run，不含 NORM）
#
# NORM 只做正常情况对照，不进训练。采完改名为:
#   output/bottleneck_dataset/norm
#
# tmux 示例（同一时间只跑一个会话）:
#   tmux new -s bn_I1
#   conda activate env_isaaclab
#   cd ~/work/isaac_factory
#   ./batch_bn_collect.sh I1
#   # Ctrl-b d 脱离；本组跑完再 tmux new -s bn_I2

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

DEVICE="cuda:0"

if [ $# -eq 0 ]; then
    echo "用法: $0 <I1|I2|I3|NORM|ALL> [I1|I2|I3|NORM ...]"
    echo "  I1: intensity=1.0，依次 machine / human / logistics / material（各 20 ep）"
    echo "  I2: intensity=2.0，同上四维"
    echo "  I3: intensity=3.0，同上四维"
    echo "  NORM: dim=none 无扰动对照（采完改名为 bottleneck_dataset/norm；不进训练）"
    echo "  ALL: I1 → I2 → I3（12 个 run，不含 NORM）"
    echo "  设备写死 ${DEVICE}（本机只有一张卡）"
    exit 1
fi

JOBS=()
for arg in "$@"; do
    if [[ "$arg" =~ ^cuda: ]]; then
        echo "忽略 '$arg'：设备已写死为 ${DEVICE}"
        continue
    elif [ "$arg" = "I1" ] || [ "$arg" = "I2" ] || [ "$arg" = "I3" ] || [ "$arg" = "NORM" ] || [ "$arg" = "ALL" ]; then
        JOBS+=("$arg")
    else
        echo "错误: 无法识别参数 '$arg'"
        echo "用法: $0 <I1|I2|I3|NORM|ALL> [I1|I2|I3|NORM ...]"
        exit 1
    fi
done

if [ ${#JOBS[@]} -eq 0 ]; then
    echo "错误: 请指定 I1 / I2 / I3 / NORM / ALL"
    exit 1
fi

HC_TASK="HRTPaHC-v1"
HC_EPISODES=20
HC_SEED=42
HC_NUM_ENVS=1
DIMS=(machine human logistics material)

echo "工作目录: ${ROOT}"
echo "设备: ${DEVICE}"
echo "每 run: ${HC_EPISODES} episodes（--max_episodes 与 --max_sim_episodes）"
echo "任务组: ${JOBS[*]}"

collect_one() {
    local dim=$1
    local intensity=$2
    echo "========================================"
    echo "[$(date '+%F %T')] 开始  dim=${dim}  intensity=${intensity}  device=${DEVICE}"
    echo "========================================"
    python train.py \
        --task "${HC_TASK}" \
        --algo rule_based \
        --num_envs "${HC_NUM_ENVS}" \
        --seed "${HC_SEED}" \
        --device "${DEVICE}" \
        --headless \
        --disturbance_dim "${dim}" \
        --disturbance_intensity "${intensity}" \
        --max_episodes "${HC_EPISODES}" \
        --max_sim_episodes "${HC_EPISODES}"
    echo "[$(date '+%F %T')] 完成  dim=${dim}  intensity=${intensity}"
}

run_intensity_group() {
    local intensity=$1
    echo "=== 强度组 I=${intensity}：${DIMS[*]} ==="
    local dim
    for dim in "${DIMS[@]}"; do
        collect_one "${dim}" "${intensity}"
    done
    echo "=== 强度组 I=${intensity} 完成 ==="
}

run_one_job() {
    local id=$1
    case $id in
        I1) run_intensity_group 1.0 ;;
        I2) run_intensity_group 2.0 ;;
        I3) run_intensity_group 3.0 ;;
        NORM)
            echo "=== NORM：dim=none 对照（采完请把时间戳目录改名为 norm；不进训练） ==="
            collect_one "none" 1.0
            echo "=== NORM 完成。改名:  mv .../bottleneck_dataset/<时间戳>_seed42  .../bottleneck_dataset/norm ==="
            ;;
        ALL)
            run_intensity_group 1.0
            run_intensity_group 2.0
            run_intensity_group 3.0
            ;;
        *) echo "错误: 无效组 $id"; return 1 ;;
    esac
}

for job in "${JOBS[@]}"; do
    echo ">>> 开始组: $job"
    run_one_job "$job"
done

echo "[$(date '+%F %T')] 全部采集完成！"
