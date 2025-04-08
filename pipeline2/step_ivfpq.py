import numpy as np
import faiss
from tqdm import tqdm
import time
import csv
import torch
import os
import argparse
from pynvml import *

# === Global config ===
k = 5           # Top-k results
n_times = 10000  # Number of batches to run

def init_nvml():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    return handle

def query_gpu_metrics(handle):
    mem_info = nvmlDeviceGetMemoryInfo(handle)
    util = nvmlDeviceGetUtilizationRates(handle)
    return mem_info.used // 1024**2, util.gpu  # Memory in MB, GPU util %

def main(output_dir, pid, bsize):
    index_file     = "/mydata/msmarco/msmarco_pq.index"
    query_e_file   = "/mydata/msmarco/ms_macro_1m_queries_embeds.npy"

    # Load index and move to GPU
    cpu_index = faiss.read_index(index_file)
    res = faiss.StandardGpuResources()
    gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
    gpu_index.nprobe = 10

    # Load queries
    query_vector = np.load(query_e_file).astype(np.float32)
    print("Query shape:", query_vector.shape)

    start_list = []
    end_list = []
    memory_checkpoints = []
    utilization_checkpoints = []

    handle = init_nvml()

    start = time.perf_counter()
    for i in tqdm(range(n_times)):
        batch = query_vector[i:(i + bsize), :]

        # Record GPU usage before search
        used_mem_mb, util_percent = query_gpu_metrics(handle)
        memory_checkpoints.append(used_mem_mb)
        utilization_checkpoints.append(util_percent)

        start_list.append(time.perf_counter())
        _ = gpu_index.search(batch, k)
        torch.cuda.synchronize()
        end_list.append(time.perf_counter())
    end = time.perf_counter()

    nvmlShutdown()

    total_queries = n_times * bsize
    throughput = total_queries / (end - start)
    avg_latency_ns = int(np.mean([(e - s) * 1e9 for s, e in zip(start_list, end_list)]))

    print(f"Batch size {bsize}, throughput: {throughput:.2f} queries/sec")
    print(f"Average latency per batch: {avg_latency_ns} ns")

    # === Save latency log ===
    os.makedirs(output_dir, exist_ok=True)
    latency_file = os.path.join(output_dir, f"ivfpq_batch{bsize}_runtime{pid}_tp{throughput:.2f}.csv")
    with open(latency_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([int((e - s) * 1e9) for s, e in zip(start_list, end_list)])

    # === Save GPU metrics log ===
    gpu_metrics_file = os.path.join(output_dir, f"ivfpq_gpu_tp{throughput:.2f}_batch{bsize}_runtime{pid}.csv")
    with open(gpu_metrics_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(memory_checkpoints)
        writer.writerow(utilization_checkpoints)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark FAISS IVFPQ search on GPU with GPU metrics")
    parser.add_argument('-p', '--output_dir', type=str, required=True, help='Directory to store output CSV')
    parser.add_argument('-id', '--pid', type=str, required=True, help='Process ID tag (e.g., 000, MIG1, A100_FULL)')
    parser.add_argument('-b', '--bsize', type=int, required=True, help='Batch size for each search call')
    args = parser.parse_args()

    main(args.output_dir, args.pid, args.bsize)
