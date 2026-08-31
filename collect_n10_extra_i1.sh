#!/usr/bin/env bash
# Extra I=1.0 episodes for each train dim. Does NOT retarget
# n10_* / new_* / material20 / zone4_* production links.
#
# Waits until the material recapture (seed 43, 20 ep) in tmux materialdata:0
# has exited, then collects N_EP (default 10) per dim with SEED (default 44).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

N_EP="${N_EP:-10}"
DEVICE="${DEVICE:-cuda:0}"
SEED="${SEED:-44}"
HC_TASK="HRTPaHC-v1"
DIMS=(machine human logistics material)
OUT_BASE="${ROOT}/source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/output/bottleneck_dataset"
LOGDIR="${ROOT}/output/collect_n10_extra_i1_logs"
# Only match the Isaac recapture train.py, not this script / pgrep / stale shells.
recapture_running() {
    pgrep -af 'python train.py' | grep -F -- '--disturbance_dim material' | grep -F -- '--max_episodes 20' \
        | grep -v collect_n10_extra | grep -v pgrep | grep -q .
}

mkdir -p "${LOGDIR}" "${OUT_BASE}"
LOGFILE="${LOGDIR}/collect.log"
exec > >(tee -a "${LOGFILE}") 2>&1

if [[ "${CONDA_DEFAULT_ENV:-}" != "env_isaaclab" ]]; then
    echo "ERROR: activate env_isaaclab first"
    exit 1
fi

echo "[$(date '+%F %T')] extra collect start  N_EP=${N_EP} seed=${SEED} device=${DEVICE}"
echo "will not touch n10_* / new_* / material20 / zone4_*"

if recapture_running; then
    echo "[$(date '+%F %T')] waiting for material recapture to finish"
    while recapture_running; do
        echo "[$(date '+%F %T')] recapture still running; sleep 120s"
        sleep 120
    done
    echo "[$(date '+%F %T')] recapture gone; starting extra dims"
else
    echo "[$(date '+%F %T')] no recapture process; start extra dims now"
fi

MANIFEST="${LOGDIR}/run_dirs.txt"
: > "${MANIFEST}"

for dim in "${DIMS[@]}"; do
    echo "========================================"
    echo "[$(date '+%F %T')] START extra dim=${dim} I=1.0 episodes=${N_EP} seed=${SEED}"
    echo "========================================"
    python train.py \
        --task "${HC_TASK}" \
        --algo rule_based \
        --num_envs 1 \
        --seed "${SEED}" \
        --device "${DEVICE}" \
        --headless \
        --disturbance_dim "${dim}" \
        --disturbance_intensity 1.0 \
        --max_episodes "${N_EP}" \
        --max_sim_episodes "${N_EP}"
    src="$(ls -1dt "${OUT_BASE}"/20*_seed"${SEED}" 2>/dev/null | head -1 || true)"
    if [[ -z "${src}" ]]; then
        echo "[$(date '+%F %T')] ERROR: no run dir after ${dim}"
        exit 1
    fi
    tag="extra_${dim}1.0_seed${SEED}"
    ln -sfn "$(basename "${src}")" "${OUT_BASE}/${tag}"
    echo "${dim} ${src}" >> "${MANIFEST}"
    echo "[$(date '+%F %T')] linked ${tag} -> $(basename "${src}")  (production links unchanged)"
done

echo "[$(date '+%F %T')] extra collect done"
cat "${MANIFEST}"
ls -ld "${OUT_BASE}"/extra_*1.0_seed"${SEED}" 2>/dev/null || true
