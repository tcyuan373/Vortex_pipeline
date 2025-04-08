import numpy as np
import faiss
from tqdm import tqdm
import time
import csv
import torch
import os
import argparse

def main(output_dir, pid):
    d       = 384
    nb      = 8841823
    nlist   = 1000
    m       = 16
    nbits   = 8
    k       = 5
    n_times = 10000
    BS      = 128

    index_file     = "/mydata/msmarco/msmarco_pq.index"
    query_e_file   = "/mydata/msmarco/ms_macro_1m_queries_embeds.npy"

    # Load index and move to GPU
    cpu_index = faiss.read_index(index_file)
    res = faiss.StandardGpuResources()
    gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
    gpu_index.nprobe = 10

    # Load queries
    query_vector = np.load(query_e_file)
    print("Query shape:", query_vector.shape)

    # Benchmark
    start_list = []
    end_list = []

    start = time.perf_counter()
    for i in tqdm(range(n_times)):
        start_list.append(time.perf_counter())
        _, _ = gpu_index.search(query_vector[i:(i+BS), :], k)
        torch.cuda.synchronize()
        end_list.append(time.perf_counter())
    end = time.perf_counter()

    total_queries = n_times * BS
    throughput = total_queries / (end - start)
    print(f"Batch size {BS}, throughput: {throughput} queries/sec")

    runtime_ns = [int((end - start) * 1e9) for start, end in zip(start_list, end_list)]
    avg_latency_ns = int(np.mean(runtime_ns))
    print(f"Average latency per batch: {avg_latency_ns} ns")

    # Save runtimes
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"ivfpq_tp{throughput:.2f}_batch{BS}_runtime{pid}.csv")
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(runtime_ns)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark FAISS IVFPQ search on GPU")
    parser.add_argument('--output_dir', type=str, default='./', help='Directory to store output CSV (default: ./)')
    parser.add_argument('--pid', type=int, default=0, help='Process ID to tag the output file (default: 0)')
    args = parser.parse_args()

    main(args.output_dir, args.pid)