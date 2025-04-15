#!/bin/bash

set -e

# === Usage info ===
echo "Usage: bash $0 <PID> <MODE1> [MODE2] [MODE3] ..."
echo "  PID:     Identifier used to pick MIG device. Use '000' to skip setting MIG/threads."
echo "  MODEs:   One or more of: A B CD E"
echo "Example:   bash $0 0 A CD E"
echo ""

# === Parse input ===
PID="${1:-000}"
shift
MODES=("$@")

# === 4 MIG UUIDs ===
# MIG_UUIDS=(
#   "MIG-efb9cd1f-5a0f-569b-98b3-d0ea501d8c4e"
#   "MIG-5dfdb424-8448-5ca7-a15e-75706b8d5ab6"
#   "MIG-2c1b2fae-443c-5fa4-8530-ee9471c2d057"
#   "MIG-f845e8ee-9f2a-5a0d-991e-707536011766"
# )

# === 2 MIG setting ===
MIG_UUIDS=(
  "MIG-233913d9-81b0-5d34-b353-559b72f50d7a"
  "MIG-4ee6e6c4-2dc8-5ad9-bbd9-a0425774c477"
)

# === Optional MIG and thread env setup ===
if [[ "$PID" != "000" ]]; then
  IDX=$((10#$PID))
  if [[ $IDX -ge 0 && $IDX -lt ${#MIG_UUIDS[@]} ]]; then
    export CUDA_VISIBLE_DEVICES="${MIG_UUIDS[$IDX]}"
    export OMP_NUM_THREADS=16
    export MKL_NUM_THREADS=16
    export OPENBLAS_NUM_THREADS=16
    export NUMEXPR_NUM_THREADS=16
    export TBB_NUM_THREADS=16
    echo "Set CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  else
    echo "Invalid PID index: $PID"
    exit 1
  fi
else
  echo "PID is 000: Not setting MIG or thread environment variables."
fi

# === Directories ===
mkdir -p micro_stepA micro_stepB micro_stepCD micro_stepE

# === Batch sizes ===
BATCHES_A=(1 2 4 8 12 16 20 24 28 32)
BATCHES_B=(1 2 4 8 12 16 20 24 28 32)
BATCHES_CD=((1 2 4 8 12 16 20 24 28 32)
BATCHES_E=(1 2 4 8 12 16 20 24 28 32)

# === Benchmark runner with GPU monitor ===
run_benchmark() {
  local mode="$1"
  local dir="$2"
  local script="$3"
  local -n batch_sizes=$4

  echo "Running $mode benchmarks..."

  for BSIZE in "${batch_sizes[@]}"; do
    echo "-> Running $script with batch size $BSIZE"
    STOP_FILE="$dir/.stop_gpu_monitor_${PID}_${BSIZE}"

    bash gpu_monitor.sh "$dir" "$PID" "$BSIZE" &
    MONITOR_PID=$!
    sleep 2

    python "$script" -p "$dir" -id "$PID" -b "$BSIZE"

    rm -f "$STOP_FILE"
    wait "$MONITOR_PID" 2>/dev/null
  done
}

# === Mode dispatcher ===
for MODE in "${MODES[@]}"; do
  case "$MODE" in
    A)
      run_benchmark "Step A (text encoder)" "micro_stepA" "step_A_text_encoder.py" BATCHES_A
      ;;
    B)
      run_benchmark "Step B (vision encoder)" "micro_stepB" "step_B_vision_encoder.py" BATCHES_B
      ;;
    CD)
      run_benchmark "Step C+D (cross attention)" "micro_stepCD" "step_CD_cross_atn.py" BATCHES_CD
      ;;
    E)
      run_benchmark "Step E (search)" "micro_stepE" "step_E_search.py" BATCHES_E
      ;;
    *)
      echo "Unknown mode: $MODE"
      exit 1
      ;;
  esac
done

echo "All selected benchmarks completed."