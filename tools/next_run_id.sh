#!/bin/bash
# Print the next free training run_id under logs/rl_games/HcFactory/hier_*.
# Same rules as g_alloc_run_id in run_2026_journal_experiments.sh:
#   bare name if free, else ${base}-v1, -v2, ...
# Journal train entries call this logic automatically — you do not need to pass HC_RUN_TAG.
#
# Usage:
#   ./tools/next_run_id.sh G0
#   ./tools/next_run_id.sh G0-greedy
#   HC_RUN_TAG=retry ./tools/next_run_id.sh G0   # pins G0-retry

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BASE="${1:?usage: $0 <base_id>   e.g. G0}"
LOG_ROOT="${ROOT}/logs/rl_games/HcFactory"
TAG="${HC_RUN_TAG:-${HC_HUMAN_RUN_TAG:-}}"

mkdir -p "${LOG_ROOT}"
if [[ -n "${TAG}" ]]; then
    echo "${BASE}-${TAG}"
    exit 0
fi
if [[ ! -e "${LOG_ROOT}/hier_${BASE}" ]]; then
    echo "${BASE}"
    exit 0
fi
n=1
while [[ -e "${LOG_ROOT}/hier_${BASE}-v${n}" ]]; do
    n=$((n + 1))
done
echo "${BASE}-v${n}"
