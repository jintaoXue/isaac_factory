#!/bin/bash
# Sidecar monitor for long training runs. Use in tmux pane 2 while train.py runs in pane 1.
#
# Examples:
#   ./tools/monitor_training.sh
#   ./tools/monitor_training.sh "train.py.*hier" 30
#   ./tools/monitor_training.sh "train.py.*HRTPaHC" 15

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

MATCH="${1:-train.py}"
INTERVAL="${2:-30}"

python tools/monitor_training.py \
  --match "$MATCH" \
  --interval "$INTERVAL" \
  --output-dir output/train_monitor
