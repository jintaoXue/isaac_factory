#!/bin/bash
# Rebuild the shared bn_agg tables and B2-B5 dataset from existing raw runs.
#
# Usage (repository root):
#   ./batch_factory_baseline_build.sh MH
#   ./batch_factory_baseline_build.sh I1
#   ./batch_factory_baseline_build.sh new_machine1.0 new_human1.0
#
# Environment overrides:
#   BENCHMARK_TAG=factory_main_aligned_v1
#   CLEAN_DERIVED=1   # remove shared_bn_agg_unsupervised_v2 before rebuilding
#   STRICT_RAW=1      # fail if any raw episode is rejected
#   EXPECTED_ACCEPTED_EPISODES=134
#   SEED=42

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${ROOT}/source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/output/bottleneck_dataset"
TOOLS_DIR="${ROOT}/source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/tools"
BENCHMARK_TAG="${BENCHMARK_TAG:-factory_main_aligned_v1}"
BENCHMARK_DIR="${DATA_ROOT}/experiments/${BENCHMARK_TAG}"
CLEAN_DERIVED="${CLEAN_DERIVED:-1}"
STRICT_RAW="${STRICT_RAW:-1}"
EXPECTED_ACCEPTED_EPISODES="${EXPECTED_ACCEPTED_EPISODES:-}"
SEED="${SEED:-42}"

WINDOW_SIZE="${WINDOW_SIZE:-60}"
INPUT_WINDOWS="${INPUT_WINDOWS:-30}"
HORIZON="${HORIZON:-180}"
SCORE_THRESHOLD="${SCORE_THRESHOLD:-0.55}"
MIN_EVENT_WINDOWS="${MIN_EVENT_WINDOWS:-8}"
OCCUPANCY_HORIZON_WINDOWS="${OCCUPANCY_HORIZON_WINDOWS:-15}"
HOT_GAP_WINDOWS="${HOT_GAP_WINDOWS:-1}"

if [ $# -eq 0 ]; then
    echo "Usage: $0 <MH|I1|raw run directory> [raw run directory ...]"
    echo "  MH: new_machine1.0 + new_human1.0"
    echo "  I1: new_machine1.0 + new_human1.0 + new_logistics1.0 + new_material1.0"
    echo "  A raw run may be a name under bottleneck_dataset or an absolute path."
    exit 1
fi

expand_run() {
    case "$1" in
        MH)
            echo "new_machine1.0"
            echo "new_human1.0"
            ;;
        I1)
            echo "new_machine1.0"
            echo "new_human1.0"
            echo "new_logistics1.0"
            echo "new_material1.0"
            ;;
        *)
            echo "$1"
            ;;
    esac
}

RUN_DIRS=()
SEEN=""
for arg in "$@"; do
    while IFS= read -r raw_run; do
        [ -n "$raw_run" ] || continue
        if [[ "$raw_run" = /* ]]; then
            run_dir="$raw_run"
        else
            run_dir="${DATA_ROOT}/${raw_run}"
        fi
        case " ${SEEN} " in
            *" ${run_dir} "*) continue ;;
        esac
        SEEN="${SEEN} ${run_dir}"
        RUN_DIRS+=("$run_dir")
    done < <(expand_run "$arg")
done

for run_dir in "${RUN_DIRS[@]}"; do
    if [ ! -d "$run_dir" ]; then
        echo "Missing raw run directory: $run_dir" >&2
        exit 1
    fi
done

echo "Repository: ${ROOT}"
echo "Benchmark: ${BENCHMARK_DIR}"
echo "Raw runs (${#RUN_DIRS[@]}):"
printf '  %s\n' "${RUN_DIRS[@]}"
echo "Protocol: window=${WINDOW_SIZE}s history=${INPUT_WINDOWS} occupancy_horizon=${OCCUPANCY_HORIZON_WINDOWS} min_event=${MIN_EVENT_WINDOWS}"
echo "Raw gate: strict=${STRICT_RAW} expected_accepted=${EXPECTED_ACCEPTED_EPISODES:-unset}"

if [ "$CLEAN_DERIVED" = "1" ]; then
    for run_dir in "${RUN_DIRS[@]}"; do
        derived_dir="${run_dir}/shared_bn_agg_unsupervised_v2"
        if [ -d "$derived_dir" ]; then
            echo "Removing stale derived tables: ${derived_dir}"
            rm -rf -- "$derived_dir"
        fi
    done
elif [ "$CLEAN_DERIVED" != "0" ]; then
    echo "CLEAN_DERIVED must be 0 or 1, got: ${CLEAN_DERIVED}" >&2
    exit 1
fi

if [ "$STRICT_RAW" != "0" ] && [ "$STRICT_RAW" != "1" ]; then
    echo "STRICT_RAW must be 0 or 1, got: ${STRICT_RAW}" >&2
    exit 1
fi

BUILD_ARGS=(
    python "${TOOLS_DIR}/build_shared_benchmark.py"
    --run_dirs "${RUN_DIRS[@]}"
    --out_dir "${BENCHMARK_DIR}"
    --window_size "${WINDOW_SIZE}"
    --input_windows "${INPUT_WINDOWS}"
    --horizon "${HORIZON}"
    --score_threshold "${SCORE_THRESHOLD}"
    --min_event_windows "${MIN_EVENT_WINDOWS}"
    --occupancy_horizon_windows "${OCCUPANCY_HORIZON_WINDOWS}"
    --hot_gap_windows "${HOT_GAP_WINDOWS}"
    --seed "${SEED}"
)
if [ "$STRICT_RAW" = "1" ]; then
    BUILD_ARGS+=(--strict)
fi
if [ -n "$EXPECTED_ACCEPTED_EPISODES" ]; then
    BUILD_ARGS+=(--expected_accepted_episodes "$EXPECTED_ACCEPTED_EPISODES")
fi

"${BUILD_ARGS[@]}"

echo "Build completed: ${BENCHMARK_DIR}"
echo "Dataset: ${BENCHMARK_DIR}/dataset.pt"
echo "Manifest: ${BENCHMARK_DIR}/dataset_manifest.json"
