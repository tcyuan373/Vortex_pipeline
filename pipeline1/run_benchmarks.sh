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
MODES=("$@")  # Remaining arguments after PID

# === 4 MIG UUIDs ===
# MIG_UUIDS=(
#   "MIG-efb9cd1f-5a0f-569b-98b3-d0ea501d8c4e"
#   "MIG-5dfdb424-8448-5ca7-a15e-75706b8d5ab6"
#   "MIG-2c1b2fae-443c-5fa4-8530-ee9471c2d057"
#   "MIG-f845e8ee-9f2a-5a0d-991e-707536011766"
# )

# == 2 MIG setting ==
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
BATCHES_COMMON=(1 2 4 8 12 16 20 24 28 32)
BATCHES_HEAVY=(1 2 4 8 16 32 64 128 256 512)
BATCHES_E=(1 2 4 8 16 32 64)

# === Run selected benchmarks ===
for MODE in "${MODES[@]}"; do
  case "$MODE" in
    A)
      echo "Running Step A (text encoder) benchmarks..."
      for BSIZE in "${BATCHES_COMMON[@]}"; do
        echo "-> Running step_A_text_encoder.py with batch size $BSIZE"
        python step_A_text_encoder.py -p micro_stepA -id "$PID" -b "$BSIZE"
      done
      ;;
    B)
      echo "Running Step B (vision encoder) benchmarks..."
      for BSIZE in "${BATCHES_COMMON[@]}"; do
        echo "-> Running step_B_vision_encoder.py with batch size $BSIZE"
        python step_B_vision_encoder.py -p micro_stepB -id "$PID" -b "$BSIZE"
      done
      ;;
    CD)
      echo "Running Step C+D (cross attention) benchmarks..."
      for BSIZE in "${BATCHES_COMMON[@]}"; do
        echo "-> Running step_CD_cross_atn.py with batch size $BSIZE"
        python step_CD_cross_atn.py -p micro_stepCD -id "$PID" -b "$BSIZE"
      done
      ;;
    E)
      echo "Running Step E (search) benchmarks..."
      for BSIZE in "${BATCHES_COMMON[@]}"; do
        echo "-> Running step_E_search.py with batch size $BSIZE"
        python step_E_search.py -p micro_stepE -id "$PID" -b "$BSIZE"
      done
      ;;
    *)
      echo "Unknown mode: $MODE"
      exit 1
      ;;
  esac
done

echo "All selected benchmarks completed."