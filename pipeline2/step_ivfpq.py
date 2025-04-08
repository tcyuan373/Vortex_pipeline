import numpy as np
import faiss
from tqdm import tqdm
import time
import csv
import torch
import os
import argparse

# === Global config ===
k = 5           # Top-k results
n_times = 10000  # Number of batches to run

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

    start = time.perf_counter()
    for i in tqdm(range(n_times)):
        start_list.append(time.perf_counter())
        _ = gpu_index.search(query_vector[i:(i + bsize), :], k)
        torch.cuda.synchronize()
        end_list.append(time.perf_counter())
    end = time.perf_counter()

    total_queries = n_times * bsize
    throughput = total_queries / (end - start)
    print(f"Batch size {bsize}, throughput: {throughput:.2f} queries/sec")

    runtime_ns = [int((end - start) * 1e9) for start, end in zip(start_list, end_list)]
    avg_latency_ns = int(np.mean(runtime_ns))
    print(f"Average latency per batch: {avg_latency_ns} ns")

    # Save runtimes
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"ivfpq_batch{bsize}_runtime{pid}_tp{throughput:.2f}.csv")
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(runtime_ns)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark FAISS IVFPQ search on GPU")
    parser.add_argument('-p', '--output_dir', type=str, required=True, help='Directory to store output CSV')
    parser.add_argument('-id', '--pid', type=str, required=True, help='Process ID tag (e.g., 0, 001, MIG1, 000 for no MIG)')
    parser.add_argument('-b', '--bsize', type=int, required=True, help='Batch size for each search call')
    args = parser.parse_args()

    main(args.output_dir, args.pid, args.bsize)