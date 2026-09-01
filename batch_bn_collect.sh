#!/usr/bin/env bash
# 瓶颈扰动采集（四分区厂：4 龙门 / 4 AGV 标称）。一张卡，组内串行。
#
# 用法（仓库根目录，先 conda activate env_isaaclab）:
#   ./batch_bn_collect.sh I1                 # machine / human / logistics / material @ 1.0
#   ./batch_bn_collect.sh I2
#   ./batch_bn_collect.sh I3
#   ./batch_bn_collect.sh NORM               # dim=none 对照（不进训练）
#   ./batch_bn_collect.sh FIVE               # NORM + I1
#   ./batch_bn_collect.sh ALL                # I1 → I2 → I3（不含 NORM）
#   ./batch_bn_collect.sh material           # 单维，默认 I=1.0
#   ./batch_bn_collect.sh machine human
#   N_EP=1 ./batch_bn_collect.sh FIVE        # 1 局冒烟
#   I=2.0 ./batch_bn_collect.sh material     # 单维改强度
#
# 环境变量: N_EP（默认 20） DEVICE（默认 cuda:0） SEED（默认 42） I（单维强度）
#
# tmux:
#   tmux new -s bn_I1
#   conda activate env_isaaclab
#   cd ~/work/isaac_factory
#   ./batch_bn_collect.sh I1
#   # Ctrl-b d 脱离。同一时间只跑一个会话。
#
# logistics I=1.0 留 4 台分区龙门、AGV 4→2，不要再传 --disturbance_*_count。

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

N_EP="${N_EP:-20}"
DEVICE="${DEVICE:-cuda:0}"
SEED="${SEED:-42}"
DIM_INTENSITY="${I:-1.0}"
HC_TASK="HRTPaHC-v1"
DIMS=(machine human logistics material)
OUT_BASE="${ROOT}/source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/output/bottleneck_dataset"
LOGDIR="${ROOT}/output/collect_logs"

usage() {
    cat <<EOF
用法: $0 <I1|I2|I3|NORM|FIVE|ALL|none|machine|human|logistics|material> [...]

  I1     intensity=1.0，四维各 ${N_EP} ep
  I2     intensity=2.0，同上
  I3     intensity=3.0，同上
  NORM   dim=none 对照（采完链到 bottleneck_dataset/norm；不进训练）
  FIVE   NORM + I1（zone4 五组）
  ALL    I1 → I2 → I3（12 个 run，不含 NORM）
  单维   none / machine / human / logistics / material（强度 I=${DIM_INTENSITY}）

  N_EP=1 $0 FIVE     冒烟
  I=2.0 $0 material  单维改强度
  设备写死 ${DEVICE}（可用 DEVICE= 覆盖）
EOF
}

if [ $# -eq 0 ]; then
    usage
    exit 1
fi

if [[ "${CONDA_DEFAULT_ENV:-}" != "env_isaaclab" ]]; then
    echo "ERROR: activate env_isaaclab first:"
    echo "  conda activate env_isaaclab"
    echo "  $0 I1"
    exit 1
fi

JOBS=()
for arg in "$@"; do
    if [[ "$arg" =~ ^cuda: ]]; then
        echo "忽略 '$arg'：设备已写死为 ${DEVICE}（DEVICE= 可覆盖）"
        continue
    fi
    case "$arg" in
        I1|I2|I3|NORM|FIVE|ALL|none|machine|human|logistics|material)
            JOBS+=("$arg")
            ;;
        *)
            echo "错误: 无法识别参数 '$arg'"
            usage
            exit 1
            ;;
    esac
done

if [ ${#JOBS[@]} -eq 0 ]; then
    echo "错误: 请指定 I1 / I2 / I3 / NORM / FIVE / ALL 或维度名"
    exit 1
fi

mkdir -p "${LOGDIR}" "${OUT_BASE}"
LOGFILE="${LOGDIR}/collect.log"
exec > >(tee -a "${LOGFILE}") 2>&1

link_run() {
    local name=$1
    local src
    src="$(ls -1dt "${OUT_BASE}"/20*_seed"${SEED}" 2>/dev/null | head -1 || true)"
    if [[ -z "${src}" ]]; then
        echo "[$(date '+%F %T')] ERROR: no run dir for ${name}"
        return 1
    fi
    ln -sfn "$(basename "${src}")" "${OUT_BASE}/${name}"
    echo "[$(date '+%F %T')] linked ${name} -> $(basename "${src}")"
}

collect_one() {
    local dim=$1
    local intensity=$2
    local tag=$3
    echo "========================================"
    echo "[$(date '+%F %T')] START ${tag}  dim=${dim}  I=${intensity}  episodes=${N_EP}  device=${DEVICE}"
    echo "========================================"
    python train.py \
        --task "${HC_TASK}" \
        --algo rule_based \
        --num_envs 1 \
        --seed "${SEED}" \
        --device "${DEVICE}" \
        --headless \
        --disturbance_dim "${dim}" \
        --disturbance_intensity "${intensity}" \
        --max_episodes "${N_EP}" \
        --max_sim_episodes "${N_EP}"
    link_run "${tag}"
    echo "[$(date '+%F %T')] DONE ${tag}"
}

tag_for() {
    local dim=$1
    local intensity=$2
    if [ "$dim" = "none" ]; then
        echo "norm"
        return
    fi
    echo "new_${dim}${intensity}"
}

run_intensity_group() {
    local intensity=$1
    echo "=== 强度组 I=${intensity}：${DIMS[*]}  N_EP=${N_EP} ==="
    local dim tag
    for dim in "${DIMS[@]}"; do
        tag="$(tag_for "$dim" "$intensity")"
        collect_one "$dim" "$intensity" "$tag"
        if [ "$intensity" = "1.0" ]; then
            link_run "zone4_${dim}1.0"
        fi
    done
    echo "=== 强度组 I=${intensity} 完成 ==="
}

run_norm() {
    echo "=== NORM：dim=none 对照（不进训练） ==="
    collect_one "none" 1.0 "norm"
    link_run "zone4_norm"
    echo "=== NORM 完成 ==="
}

run_one_job() {
    local id=$1
    case $id in
        I1) run_intensity_group 1.0 ;;
        I2) run_intensity_group 2.0 ;;
        I3) run_intensity_group 3.0 ;;
        NORM) run_norm ;;
        FIVE)
            run_norm
            run_intensity_group 1.0
            ;;
        ALL)
            run_intensity_group 1.0
            run_intensity_group 2.0
            run_intensity_group 3.0
            ;;
        none)
            collect_one "none" "${DIM_INTENSITY}" "$(tag_for none "${DIM_INTENSITY}")"
            link_run "zone4_norm"
            ;;
        machine|human|logistics|material)
            collect_one "$id" "${DIM_INTENSITY}" "$(tag_for "$id" "${DIM_INTENSITY}")"
            if [ "${DIM_INTENSITY}" = "1.0" ]; then
                link_run "zone4_${id}1.0"
                if [ "$id" = "material" ]; then
                    link_run "zone4_material1.0_kit"
                fi
            fi
            ;;
        *) echo "错误: 无效组 $id"; return 1 ;;
    esac
}

echo "[$(date '+%F %T')] collect start  cwd=${ROOT}  conda=${CONDA_PREFIX}"
echo "设备: ${DEVICE}  seed=${SEED}  N_EP=${N_EP}  任务: ${JOBS[*]}"
nvidia-smi -L || echo "nvidia-smi unavailable (collect still continues)"

for job in "${JOBS[@]}"; do
    echo ">>> 开始组: $job"
    run_one_job "$job"
done

echo "[$(date '+%F %T')] 全部采集完成！"
ls -ld "${OUT_BASE}"/norm "${OUT_BASE}"/new_* "${OUT_BASE}"/zone4_* 2>/dev/null || true
