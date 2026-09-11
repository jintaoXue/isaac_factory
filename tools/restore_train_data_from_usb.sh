#!/usr/bin/env bash
# Restore packed train data into an isaac_factory repo root.
# Usage:
#   bash restore_train_data_from_usb.sh /path/to/isaac_factory
#   bash restore_train_data_from_usb.sh /path/to/isaac_factory /media/$USER/XUE_DISK1/isaac_factory_train_data

set -euo pipefail

REPO="${1:-}"
SRC="${2:-}"

if [[ -z "$REPO" ]]; then
  echo "用法: $0 <isaac_factory仓库根目录> [分包目录]" >&2
  exit 1
fi

if [[ -z "$SRC" ]]; then
  HERE="$(cd "$(dirname "$0")" && pwd)"
  if ls "$HERE"/isaac_factory_train_data.tar.*.tarpart >/dev/null 2>&1; then
    SRC="$HERE"
  else
    echo "请传入分包目录，或把本脚本放在含 .tarpart 的目录里运行。" >&2
    exit 1
  fi
fi

mkdir -p "$REPO"
echo "解压 $SRC → $REPO"
cat "$SRC"/isaac_factory_train_data.tar.*.tarpart | tar -xvf - -C "$REPO"

NN="$REPO/logs/rl_games/HcFactory/hier_2026-08-27_23-17-41/nn"
EP="$REPO/env_checkpoints/policy_explore/N10_T40000__E2_teacher_ep50/offline_replay/episodes"
echo
echo "校验:"
ls -lh "$NN" || true
echo -n "episodes 数: "
ls "$EP" 2>/dev/null | wc -l || echo 0
echo "完成。可在仓库根运行: ./run_2026_journal_experiments.sh E0 cuda:0"
