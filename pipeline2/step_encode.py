#!/usr/bin/env python3
import argparse
import csv
import numpy as np
import torch
import os
import warnings
warnings.filterwarnings("ignore")
import tqdm
tqdm.tqdm = lambda *args, **kwargs: iter(args[0])
from FlagEmbedding import FlagModel

# === Global configuration ===
TOTAL_RUNS = 1000
QUERY_PATH = '/mydata/msmarco/msmarco_3_clusters/query.csv'
MODEL_NAME = 'BAAI/bge-small-en-v1.5'
DEVICE = 'cuda:0'


class EncoderUDL:
    def __init__(self):
        self.encoder = FlagModel(MODEL_NAME, DEVICE)
        self.emb_dim = 384

    def encode(self, query_list):
        return self.encoder.encode(query_list)

    def __del__(self):
        pass


def run_benchmark(data, bsize, encoder):
    run_times = []
    j = 0

    for _ in range(TOTAL_RUNS):
        query_list = [data[j % len(data)] for _ in range(bsize)]
        j += bsize

        model_start_event = torch.cuda.Event(enable_timing=True)
        model_end_event = torch.cuda.Event(enable_timing=True)

        model_start_event.record()
        query_embeddings = encoder.encode(query_list)
        model_end_event.record()
        torch.cuda.synchronize()

        run_times.append((model_start_event.elapsed_time(model_end_event)) * 1e6)  # microseconds to nanoseconds

    return run_times, query_embeddings


def main(output_dir, pid, bsize):
    with open(QUERY_PATH, mode='r') as file:
        csv_reader = csv.reader(file)
        data = [' '.join(line) for line in csv_reader]

    encoder = EncoderUDL()
    run_times, last_embeddings = run_benchmark(data, bsize, encoder)

    throughput = (bsize * len(run_times)) / (sum(run_times) / 1e9)
    avg_latency = int(sum(run_times) / len(run_times))

    print(f"Batch size {bsize}, throughput is {throughput:.2f} queries/sec")
    print(f"Average latency per batch: {avg_latency} ns")
    print(f"Finished, final embedding shape: {last_embeddings.shape}")

    os.makedirs(output_dir, exist_ok=True)
    runtimes_file = os.path.join(
        output_dir,
        f'encode_bsize{bsize}_runtime{pid}_tp{throughput:.2f}.csv'
    )

    with open(runtimes_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(run_times)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark BGE FlagModel Encoding")
    parser.add_argument('-p', '--output_dir', type=str, required=True, help='Directory to store output CSV')
    parser.add_argument('-id', '--pid', type=str, required=True, help='String identifier for MIG setup (e.g., 0,1,2,3,000 if no MIG)')
    parser.add_argument('-b', '--bsize', type=int, required=True, help='Batch size for encoding')
    args = parser.parse_args()

    main(args.output_dir, args.pid, args.bsize)
