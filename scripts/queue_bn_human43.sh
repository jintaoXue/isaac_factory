#!/usr/bin/env bash
set -euo pipefail

echo "[$(date '+%F %T')] waiting for HcFactory to finish..."
while pgrep -x HcFactory >/dev/null; do
  echo "[$(date '+%H:%M:%S')] still running..."
  sleep 60
done

echo "[$(date '+%F %T')] finished, starting next run (human, intensity=2.0, seed=43)"
source /home/sci/repos/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
cd /home/sci/work/isaac_factory

python train.py --task HRTPaHC-v1 --algo rule_based --num_envs 1 --device cuda:0 --headless \
  --disturbance_dim human --disturbance_intensity 2.0 \
  --seed 43 \
  --max_episodes 50

echo "[$(date '+%F %T')] next run exited with code $?"
