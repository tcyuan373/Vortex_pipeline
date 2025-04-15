#!/bin/bash

# === Parse Arguments ===
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <output_dir> <pid> <bsize>"
    exit 1
fi

LOG_DIR="$1"
PID="$2"
BSIZE="$3"

mkdir -p "$LOG_DIR"
FINAL_GPU_LOG="$LOG_DIR/gpu_util_pid${PID}_bsize${BSIZE}.csv"
INTERVAL=2

declare -a MEMORY_USED_LIST
declare -a UTILIZATION_LIST

STOP_FILE="$LOG_DIR/.stop_gpu_monitor_${PID}_${BSIZE}"

cleanup() {
    echo "Stopping GPU monitoring..."
    echo "Saving logs to $FINAL_GPU_LOG..."
    IFS=',' MEM_LINE="${MEMORY_USED_LIST[*]}"
    IFS=',' UTIL_LINE="${UTILIZATION_LIST[*]}"
    echo "$MEM_LINE" > "$FINAL_GPU_LOG"
    echo "$UTIL_LINE" >> "$FINAL_GPU_LOG"
    rm -f "$STOP_FILE"
    echo "Logs saved to $FINAL_GPU_LOG."
    exit 0
}

trap cleanup SIGINT SIGTERM

echo "GPU monitor running with PID $$"
echo "Monitoring GPU every $INTERVAL seconds until $STOP_FILE is removed."

# Create stop file
touch "$STOP_FILE"

while [[ -f "$STOP_FILE" ]]; do
    mapfile -t output_lines < <(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits)

    for line in "${output_lines[@]}"; do
        IFS=',' read -r UTIL USED_MEM <<< "$line"
        MEMORY_USED_LIST+=("${USED_MEM// /}")
        UTILIZATION_LIST+=("${UTIL// /}")
    done

    sleep $INTERVAL
done

cleanup
