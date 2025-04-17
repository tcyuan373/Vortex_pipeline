#!/bin/bash

set -e

# === Usage info ===
echo "Usage: bash $0 <PID> <MODE1> [MODE2] [MODE3] ..."
echo "  PID:     Identifier used to pick MIG device. Use '000' to skip setting MIG/threads."
echo "  MODEs:   One or more of: A B CD E streamE"
echo "Example:   bash $0 0 A CD E"
echo ""

# === Parse input ===
PID="${1:-000}"
shift
MODES=("$@")  # Remaining arguments after PID

# === 4 MIG UUIDs ===
#MIG_UUIDS=(
#  "MIG-3357e852-8b6d-5d90-8755-2f9f2e542311"
#  "MIG-88103043-32d5-5304-9384-79008081ec63"
#  "MIG-d7cf6436-b3e7-5230-b9d4-3cff2af2b1a9"
#  "MIG-5a6035a9-6f02-5280-b85a-84d30f245c16"
#)

# == 2 MIG setting ==
MIG_UUIDS=(
    "MIG-a79594fe-f33a-5f37-9148-af16fbb9f1e0"
    "MIG-3d993195-0b7f-5f90-a21c-ebb1caa36fb5"
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
BATCHES_COMMON=(4 8 16 )
BATCHES_HEAVY=(1 2 4 8 16 32 64 128 256 512)
BATCHES_E=(1 4 8 16)

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
      for BSIZE in "${BATCHES_E[@]}"; do
        echo "-> Running step_E_search.py with batch size $BSIZE"
        python step_E_search.py -p micro_stepE -id "$PID" -b "$BSIZE"
      done
      ;;
    streamE)
      echo "Running stream Step E (search) benchmarks..."
      for BSIZE in "${BATCHES_E[@]}"; do
        echo "-> Running stream_stepE.py with batch size $BSIZE"
        python stream_stepE.py -p micro_stepE -id "$PID" -b "$BSIZE" 
      done
      ;;
    *)
      echo "Unknown mode: $MODE"
      exit 1
      ;;
  esac
done

echo "All selected benchmarks completed."
