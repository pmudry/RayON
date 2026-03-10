#!/bin/bash
#
# RayON Performance Benchmark Script
# Runs the offline CUDA renderer on the default scene with fixed parameters
# and records wall-clock time. Results are appended to a CSV for regression
# tracking across git commits.
#
# Usage:
#   ./scripts/benchmark.sh              # Run benchmark, append to CSV
#   ./scripts/benchmark.sh --compare    # Run + compare against last recorded commit
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="$PROJECT_DIR/build"
RESULTS_FILE="$PROJECT_DIR/bench_results.csv"
BINARY="$BUILD_DIR/rayon"

# Benchmark parameters
SAMPLES=1024
RESOLUTION=720
METHOD=2  # CUDA offline
WARMUP_RUNS=1
BENCH_RUNS=3

COMPARE=false

for arg in "$@"; do
    case "$arg" in
        --compare) COMPARE=true ;;
        --help|-h)
            echo "Usage: $0 [--compare]"
            echo ""
            echo "  --compare   Compare results against the last recorded commit"
            echo ""
            echo "Config: ${WARMUP_RUNS} warmup + ${BENCH_RUNS} timed runs @ ${RESOLUTION}p, ${SAMPLES} spp"
            exit 0
            ;;
    esac
done

# Ensure binary exists
if [[ ! -x "$BINARY" ]]; then
    echo "Binary not found at $BINARY — building..."
    (cd "$BUILD_DIR" && cmake .. -DCMAKE_BUILD_TYPE=Release > /dev/null 2>&1 && make -j"$(nproc)" > /dev/null 2>&1)
fi

# Git info
GIT_COMMIT=$(git -C "$PROJECT_DIR" rev-parse --short HEAD 2>/dev/null || echo "unknown")
GIT_BRANCH=$(git -C "$PROJECT_DIR" rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")
GIT_DIRTY=""
if ! git -C "$PROJECT_DIR" diff --quiet 2>/dev/null; then
    GIT_DIRTY="-dirty"
fi
COMMIT_LABEL="${GIT_COMMIT}${GIT_DIRTY}"
TIMESTAMP=$(date -Iseconds)

# GPU info
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | xargs || echo "unknown")

echo "========================================"
echo " RayON Benchmark"
echo "========================================"
echo " Commit:     $COMMIT_LABEL ($GIT_BRANCH)"
echo " GPU:        $GPU_NAME"
echo " Resolution: ${RESOLUTION}p | Samples: ${SAMPLES} spp"
echo " Runs:       ${WARMUP_RUNS} warmup + ${BENCH_RUNS} timed"
echo "========================================"
echo ""

# Create CSV header if file doesn't exist or is empty
if [[ ! -s "$RESULTS_FILE" ]] || ! head -1 "$RESULTS_FILE" | grep -q "^timestamp,"; then
    echo "timestamp,commit,branch,gpu,resolution,samples,run,time_s" > "$RESULTS_FILE"
fi

# Warmup
printf "Warmup...  "
for ((w = 1; w <= WARMUP_RUNS; w++)); do
    (cd "$BUILD_DIR" && ./rayon -m $METHOD -s $SAMPLES -r $RESOLUTION > /dev/null 2>&1) < /dev/null
    printf "done "
done
echo ""

# Timed runs
times=()
printf "Benchmark: "
for ((r = 1; r <= BENCH_RUNS; r++)); do
    start=$(date +%s%N)
    (cd "$BUILD_DIR" && ./rayon -m $METHOD -s $SAMPLES -r $RESOLUTION > /dev/null 2>&1) < /dev/null
    end=$(date +%s%N)

    t=$(echo "scale=3; ($end - $start) / 1000000000" | bc)
    printf "%ss " "$t"
    times+=("$t")

    echo "$TIMESTAMP,$COMMIT_LABEL,$GIT_BRANCH,$GPU_NAME,$RESOLUTION,$SAMPLES,$r,$t" >> "$RESULTS_FILE"
done

# Compute median
median=$(printf '%s\n' "${times[@]}" | sort -n | awk '{a[NR]=$1} END{if(NR%2==1) print a[(NR+1)/2]; else print (a[NR/2]+a[NR/2+1])/2}')
printf " -> median: %ss\n" "$median"

echo ""
echo "Results saved to: $RESULTS_FILE"

# Comparison
if $COMPARE; then
    echo ""
    echo "========================================"
    echo " Comparison with previous commit"
    echo "========================================"

    PREV_COMMIT=$(awk -F',' -v cur="$COMMIT_LABEL" 'NR>1 && $2!=cur && $2!="" {print $2}' "$RESULTS_FILE" | tail -1)

    if [[ -z "$PREV_COMMIT" ]]; then
        echo "No previous commit data found. Run benchmark on another commit first."
    else
        # Get previous median
        prev_median=$(awk -F',' -v commit="$PREV_COMMIT" \
            'NR>1 && $2==commit {print $8}' "$RESULTS_FILE" \
            | sort -n | awk '{a[NR]=$1} END{if(NR>0){if(NR%2==1) print a[(NR+1)/2]; else print (a[NR/2]+a[NR/2+1])/2}}')

        if [[ -z "$prev_median" || "$prev_median" == "0" ]]; then
            echo "Previous: $PREV_COMMIT (no valid timing data)"
        else
            pct=$(echo "scale=1; ($median - $prev_median) / $prev_median * 100" | bc)

            echo "Previous: $PREV_COMMIT  ${prev_median}s"
            echo "Current:  $COMMIT_LABEL  ${median}s"
            echo ""

            if (( $(echo "$pct > 5" | bc -l) )); then
                echo "REGRESSION: +${pct}% slower"
                exit 1
            elif (( $(echo "$pct < -5" | bc -l) )); then
                echo "Improvement: ${pct}% faster"
            else
                echo "No significant change (${pct}%)"
            fi
        fi
    fi
fi
