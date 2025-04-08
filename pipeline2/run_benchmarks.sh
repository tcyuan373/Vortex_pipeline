#!/bin/bash

set -e

# === Directories for results ===
TEXTCHECK_DIR="ppl2_tcheck"
IVFPQ_DIR="ppl2_ivf"
PID="000"

# Create output directories if they don't exist
mkdir -p "$TEXTCHECK_DIR"
mkdir -p "$IVFPQ_DIR"

# === Batch sizes ===
TEXTCHECK_BATCHES=(1 2 4 8 16 32 64 128 256)
IVFPQ_BATCHES=(1 2 4 8 16 32 64 128 256 1024 2048 4096)

# === Run text classification benchmarks ===
echo "Running text check benchmarks..."
for BSIZE in "${TEXTCHECK_BATCHES[@]}"; do
    echo "-> Running step_text_check.py with batch size $BSIZE"
    python step_text_check.py -p "$TEXTCHECK_DIR" -id "$PID" -b "$BSIZE"
done

# === Run FAISS IVFPQ benchmarks ===
echo "Running IVFPQ search benchmarks..."
for BSIZE in "${IVFPQ_BATCHES[@]}"; do
    echo "-> Running step_ivfpq.py with batch size $BSIZE"
    python step_ivfpq.py -p "$IVFPQ_DIR" -id "$PID" -b "$BSIZE"
done

echo "All benchmarks completed."