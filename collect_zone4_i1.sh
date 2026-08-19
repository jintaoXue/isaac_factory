#!/usr/bin/env bash
# Sequential 1-episode collect on the 4-gantry / 4-AGV plant:
#   zone4_norm + four I=1.0 dims (machine / human / logistics / material)
# Logistics I=1.0 keeps 4+4 (only slowdown + L2 freeze), matching the new plant.
#
# 用法（仓库根目录，先 conda activate env_isaaclab）:
#   bash collect_zone4_i1.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

if [[ "${CONDA_DEFAULT_ENV:-}" != "env_isaaclab" ]]; then
  echo "ERROR: activate env_isaaclab first:"
  echo "  conda activate env_isaaclab"
  echo "  bash collect_zone4_i1.sh"
  exit 1
fi

OUT_BASE="${ROOT}/source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/output/bottleneck_dataset"
LOGDIR="${ROOT}/output/collect_zone4_i1_logs"
mkdir -p "${LOGDIR}" "${OUT_BASE}"

link_run() {
  local name=$1
  local src
  src="$(ls -1dt "${OUT_BASE}"/20*_seed42 2>/dev/null | head -1 || true)"
  if [[ -z "${src}" ]]; then
    echo "[$(date '+%F %T')] ERROR: no run dir for ${name}"
    return 1
  fi
  ln -sfn "$(basename "${src}")" "${OUT_BASE}/${name}"
  echo "[$(date '+%F %T')] linked ${name} -> $(basename "${src}")"
}

run_one() {
  local tag=$1
  local dim=$2
  shift 2
  echo "========================================"
  echo "[$(date '+%F %T')] START ${tag}  dim=${dim} $*"
  echo "========================================"
  python train.py \
    --task HRTPaHC-v1 \
    --algo rule_based \
    --num_envs 1 \
    --seed 42 \
    --device cuda:0 \
    --headless \
    --disturbance_dim "${dim}" \
    --disturbance_intensity 1.0 \
    --max_episodes 1 \
    --max_sim_episodes 1 \
    "$@"
  link_run "${tag}"
  echo "[$(date '+%F %T')] DONE ${tag}"
}

LOGFILE="${LOGDIR}/collect.log"
exec > >(tee -a "${LOGFILE}") 2>&1

echo "[$(date '+%F %T')] zone4 1-ep collect start  cwd=${ROOT}  conda=${CONDA_PREFIX}"
nvidia-smi -L || echo "nvidia-smi unavailable (collect still continues)"

run_one zone4_norm none
run_one zone4_machine1.0 machine
run_one zone4_human1.0 human
run_one zone4_logistics1.0 logistics --disturbance_gantry_count 4 --disturbance_agv_count 4
run_one zone4_material1.0 material

echo "[$(date '+%F %T')] ALL FIVE RUNS DONE"
ls -ld "${OUT_BASE}"/zone4_*
