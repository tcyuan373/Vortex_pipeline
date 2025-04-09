#!/bin/bash

set -e

# === Usage info ===
echo "Usage: bash $0 <PID> <MODE1> [MODE2] [MODE3] ..."
echo "  PID:     Identifier used to pick MIG device. Use '000' to skip setting MIG/threads."
echo "  MODEs:   One or more of: tcheck, ivf, audio, encode, lang, topic, sum, tts"
echo "Example:   bash $0 0 tcheck topic tts"
echo ""

# === Parse input ===
PID="${1:-000}"
shift
MODES=("$@")  # Remaining arguments after PID

# === 4 MIG UUIDs ===
#MIG_UUIDS=(
#  "MIG-efb9cd1f-5a0f-569b-98b3-d0ea501d8c4e"
#  "MIG-5dfdb424-8448-5ca7-a15e-75706b8d5ab6"
#  "MIG-2c1b2fae-443c-5fa4-8530-ee9471c2d057"
#  "MIG-f845e8ee-9f2a-5a0d-991e-707536011766"
#)

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
mkdir -p ppl2_tcheck ppl2_ivf ppl2_audio ppl2_encode ppl2_lang ppl2_topic ppl2_sum ppl2_tts

# === Batch sizes ===
TEXTCHECK_BATCHES=(1 2 4 8 16 32 64 128 256)
IVFPQ_BATCHES=(1 2 4 8 16 32 64 128 256 512 1024)
AUDIO_BATCHES=(20 24 28)
ENCODE_BATCHES=(1 2 4 8 16 32 64 128 256 512 1024)
LANG_BATCHES=(1 4 8 12 16 20 24 28 32)
TOPIC_BATCHES=(1 2 4 8 16 32 64 128)
SUMMARIZE_BATCHES=(1 2 4 8 16 32)
TTS_BATCHES=(1 2 4 8 16 32)

# === Benchmark runner ===
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

    rm -f "$STOP_FILE"  # Signal monitor to stop
    wait "$MONITOR_PID" 2>/dev/null
  done
}

# === Mode dispatcher ===
for MODE in "${MODES[@]}"; do
  case "$MODE" in
    tcheck)
      run_benchmark "text check" "ppl2_tcheck" "step_text_check.py" TEXTCHECK_BATCHES
      ;;
    ivf)
      run_benchmark "IVFPQ search" "ppl2_ivf" "step_ivfpq.py" IVFPQ_BATCHES
      ;;
    audio)
      run_benchmark "audio recognition" "ppl2_audio" "step_audio_recognition.py" AUDIO_BATCHES
      ;;
    encode)
      run_benchmark "encode" "ppl2_encode" "step_encode.py" ENCODE_BATCHES
      ;;
    lang)
      run_benchmark "language detection" "ppl2_lang" "step_lan_det.py" LANG_BATCHES
      ;;
    topic)
      run_benchmark "topic classification" "ppl2_topic" "step_topic_classification.py" TOPIC_BATCHES
      ;;
    sum)
      run_benchmark "summarization" "ppl2_sum" "step_sum.py" SUMMARIZE_BATCHES
      ;;
    tts)
      run_benchmark "TTS synthesis" "ppl2_tts" "step_tts.py" TTS_BATCHES
      ;;
    *)
      echo "Unknown mode: $MODE"
      exit 1
      ;;
  esac
done

echo "All selected benchmarks completed."