#!/bin/bash
# Sidecar monitor for long training runs. Use in tmux pane 2 while train.py runs in pane 1.
#
# train.py renames itself via setproctitle (e.g. HcFactory-hier-xjt), so match that
# instead of "train.py" once the process has started.
#
# Examples:
#   ./tools/monitor_training.sh
#   ./tools/monitor_training.sh "HcFactory-hier" 30
#   ./tools/monitor_training.sh "HcFactory-rl_filter" 30

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

MATCH="${1:-HcFactory-}"
INTERVAL="${2:-30}"

python tools/monitor_training.py \
  --match "$MATCH" \
  --interval "$INTERVAL" \
  --output-dir outputs/train_monitor
