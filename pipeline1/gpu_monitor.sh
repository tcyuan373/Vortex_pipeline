#!/bin/bash

mkdir -p gpu_data
cd gpu_data

while true
do
    # Append timestamp and GPU utilization data
    echo "$(date +%s), $(nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv,noheader)" >> gpu_utilization.dat

    # Append timestamp and per-process GPU usage data
    nvidia-smi --query-compute-apps=process_name,pid,used_memory --format=csv,noheader | awk -v ts="$(date +%s), " '{print ts $0}' >> gpu_by_process.dat

    sleep 1
done