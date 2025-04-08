#!/bin/bash

set -e

# === Usage info ===
echo "Usage: bash $0 <PID> <MODE1> [MODE2] [MODE3] ..."
echo "  PID:     Identifier used to pick MIG device. Use '000' to skip setting MIG/threads."
echo "  MODEs:   One or more of: tcheck, ivf, audio, encode, lang"
echo "Example:   bash $0 0 tcheck ivf"
echo ""

# === Parse input ===
PID="${1:-000}"
shift
MODES=("$@")  # Remaining arguments after PID

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
mkdir -p ppl2_tcheck ppl2_ivf ppl2_audio ppl2_encode ppl2_lang

# === Batch sizes ===
TEXTCHECK_BATCHES=(1 2 4 8 16 32 64 128 256)
IVFPQ_BATCHES=(1 2 4 8 16 32 64 128 256 512 1024)
AUDIO_BATCHES=(1 2 4 8 16 32 36 40)
ENCODE_BATCHES=(1 2 4 8 16 32 64 128 256 512 1024)
LANG_BATCHES=(1 4 8 12 16 20 24 28 32)

# === Run selected benchmarks ===

for MODE in "${MODES[@]}"; do
  case "$MODE" in
    tcheck)
      echo "Running text check benchmarks..."
      for BSIZE in "${TEXTCHECK_BATCHES[@]}"; do
        echo "-> Running step_text_check.py with batch size $BSIZE"
        bash gpu_monitor.sh ppl2_tcheck "$PID" "$BSIZE" &
        MONITOR_PID=$!
        sleep 2  # optional: allow monitor to start
        python step_text_check.py -p ppl2_tcheck -id "$PID" -b "$BSIZE"
        kill -SIGINT "$MONITOR_PID"
        wait "$MONITOR_PID" 2>/dev/null
      done
      ;;
    ivf)
      echo "Running IVFPQ search benchmarks..."
      for BSIZE in "${IVFPQ_BATCHES[@]}"; do
        echo "-> Running step_ivfpq.py with batch size $BSIZE"
        bash gpu_monitor.sh ppl2_ivf "$PID" "$BSIZE" &
        MONITOR_PID=$!
        sleep 2
        python step_ivfpq.py -p ppl2_ivf -id "$PID" -b "$BSIZE"
        kill -SIGINT "$MONITOR_PID"
        wait "$MONITOR_PID" 2>/dev/null
      done
      ;;
    audio)
      echo "Running audio recognition benchmarks..."
      for BSIZE in "${AUDIO_BATCHES[@]}"; do
        echo "-> Running step_audio_recognition.py with batch size $BSIZE"
        bash gpu_monitor.sh ppl2_audio "$PID" "$BSIZE" &
        MONITOR_PID=$!
        sleep 2
        python step_audio_recognition.py -p ppl2_audio -id "$PID" -b "$BSIZE"
        kill -SIGINT "$MONITOR_PID"
        wait "$MONITOR_PID" 2>/dev/null
      done
      ;;
    encode)
      echo "Running encode benchmarks..."
      for BSIZE in "${ENCODE_BATCHES[@]}"; do
        echo "-> Running step_encode.py with batch size $BSIZE"
        bash gpu_monitor.sh ppl2_encode "$PID" "$BSIZE" &
        MONITOR_PID=$!
        sleep 2
        python step_encode.py -p ppl2_encode -id "$PID" -b "$BSIZE"
        kill -SIGINT "$MONITOR_PID"
        wait "$MONITOR_PID" 2>/dev/null
      done
      ;;
    lang)
      echo "Running language detection benchmarks..."
      for BSIZE in "${LANG_BATCHES[@]}"; do
        echo "-> Running step_lan_det.py with batch size $BSIZE"
        bash gpu_monitor.sh ppl2_lang "$PID" "$BSIZE" &
        MONITOR_PID=$!
        sleep 2
        python step_lan_det.py -p ppl2_lang -id "$PID" -b "$BSIZE"
        kill -SIGINT "$MONITOR_PID"
        wait "$MONITOR_PID" 2>/dev/null
      done
      ;;
    *)
      echo "Unknown mode: $MODE"
      exit 1
      ;;
  esac
done

echo "All selected benchmarks completed."