#!/bin/bash
# Sidecar monitor for long training runs. Use in tmux pane 2 while train.py runs in pane 1.
#
# train.py renames itself via setproctitle (e.g. HcFactory-hier-xjt), so match that
# instead of "train.py" once the process has started.
#
# Examples:
#   ./tools/monitor_training.sh
#   ./tools/monitor_training.sh "HcFactory-" 15
#   ./tools/monitor_training.sh "HcFactory-" 10 freeze   # desk freeze hunt preset
#   ./tools/monitor_training.sh "HcFactory-G0" 10 freeze

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

MATCH="${1:-HcFactory-}"
INTERVAL="${2:-30}"
MODE="${3:-}"

EXTRA=()
if [[ "${MODE}" == freeze || "${MODE}" == --freeze-hunt || "${INTERVAL}" == freeze ]]; then
    if [[ "${INTERVAL}" == freeze ]]; then
        INTERVAL=10
    fi
    EXTRA+=(--freeze-hunt)
fi

python tools/monitor_training.py \
  --match "$MATCH" \
  --interval "$INTERVAL" \
  --output-dir outputs/train_monitor \
  --watch-display \
  "${EXTRA[@]}"
