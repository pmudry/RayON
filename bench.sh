#!/bin/bash
# Benchmark script for RayON CUDA renderer
# Usage: ./bench.sh [runs] [label]
#   runs:  number of iterations (default: 5)
#   label: descriptive label for this benchmark run

RUNS=${1:-5}
LABEL=${2:-"unlabeled"}
RESULTS_FILE="bench_results.csv"

# Create results file with header if it doesn't exist
if [ ! -f "$RESULTS_FILE" ]; then
    echo "timestamp,label,run,time_s,rays_traced,rays_per_sec" > "$RESULTS_FILE"
fi

echo "=== Benchmark: $LABEL ($RUNS runs) ==="

times=()
rays_arr=()
rps_arr=()

for i in $(seq 1 $RUNS); do
    output=$(./rayon -m 2 -s 1024 -r 1080 2>&1)
    time_s=$(echo "$output" | grep -oP 'completed in \K[0-9.]+')
    rays=$(echo "$output" | grep -oP 'Rays traced: \K[0-9,]+' | tr -d ',')
    rps=$(echo "$output" | grep -oP 'Rays/sec: \K[0-9,]+' | tr -d ',')

    times+=($time_s)
    rays_arr+=($rays)
    rps_arr+=($rps)

    ts=$(date +%Y-%m-%dT%H:%M:%S)
    echo "$ts,$LABEL,$i,$time_s,$rays,$rps" >> "$RESULTS_FILE"
    printf "  Run %d: %6.2fs  |  Rays: %s  |  Rays/s: %s\n" "$i" "$time_s" "$rays" "$rps"
done

# Compute averages using awk
avg_time=$(printf '%s\n' "${times[@]}" | awk '{s+=$1} END {printf "%.2f", s/NR}')
avg_rays=$(printf '%s\n' "${rays_arr[@]}" | awk '{s+=$1} END {printf "%.0f", s/NR}')
avg_rps=$(printf '%s\n' "${rps_arr[@]}" | awk '{s+=$1} END {printf "%.0f", s/NR}')

# Compute std dev for time
std_time=$(printf '%s\n' "${times[@]}" | awk -v avg="$avg_time" '{d=$1-avg; s+=d*d} END {printf "%.2f", sqrt(s/NR)}')

echo "---"
printf "  Avg:   %6.2fs ± %4.2fs  |  Avg Rays: %s  |  Avg Rays/s: %s\n" "$avg_time" "$std_time" "$avg_rays" "$avg_rps"
echo "=== Results appended to $RESULTS_FILE ==="
