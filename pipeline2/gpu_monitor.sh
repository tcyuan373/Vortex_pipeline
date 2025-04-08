#!/bin/bash

# === Parse Arguments ===
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <output_dir> <pid> <bsize>"
    exit 1
fi

LOG_DIR="$1"
PID="$2"
BSIZE="$3"

# === Ensure log directory exists ===
mkdir -p "$LOG_DIR"

# === Final log filenames ===
FINAL_GPU_LOG="$LOG_DIR/gpu_util_pid${PID}_bsize${BSIZE}.dat"
DCGM_LOG="$LOG_DIR/dcgm_log_pid${PID}_bsize${BSIZE}.dat"
TEMP_GPU_FILE=$(mktemp)

# === Monitoring intervals ===
MEMORY_INTERVAL=5  # seconds
DCGM_INTERVAL=5    # seconds
DCGM_SAMPLING_INTERVAL=$((DCGM_INTERVAL * 1000))

# === Write header for GPU memory log ===
echo "Timestamp, GPU_ID, Total_Memory_MB, Used_Memory_MB, Free_Memory_MB" > "$TEMP_GPU_FILE"

# === Start DCGM monitoring ===
echo "Starting DCGM monitoring (logs to $DCGM_LOG)..."
stdbuf -oL dcgmi dmon -e 203,204,1001,1002,1003,1004,1005,155 -d "$DCGM_SAMPLING_INTERVAL" > "$DCGM_LOG" 2>&1 &
DCGM_PID=$!

# === Cleanup function ===
cleanup() {
    echo "Stopping DCGM monitoring..."
    kill -SIGTERM "$DCGM_PID"
    sleep 1
    sync

    echo "Saving collected GPU memory usage to $FINAL_GPU_LOG..."
    mv "$TEMP_GPU_FILE" "$FINAL_GPU_LOG"

    echo "Logs saved to $FINAL_GPU_LOG and $DCGM_LOG."
    exit 0
}

# === Trap Ctrl+C (SIGINT) to trigger cleanup ===
trap cleanup SIGINT

echo "Logging GPU memory usage every $MEMORY_INTERVAL seconds. DCGM monitoring every $DCGM_INTERVAL seconds."
echo "Press Ctrl+C to stop and save logs."

# === Main loop for GPU memory logging ===
while true; do
    TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")
    nvidia-smi --query-gpu=index,memory.total,memory.used,memory.free --format=csv,noheader,nounits | \
    while IFS=',' read -r GPU_ID TOTAL USED FREE; do
        echo "$TIMESTAMP, $GPU_ID, $TOTAL, $USED, $FREE" >> "$TEMP_GPU_FILE"
    done
    sleep $MEMORY_INTERVAL
done