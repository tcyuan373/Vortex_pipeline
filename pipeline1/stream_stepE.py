#!/usr/bin/env python3
import argparse
import csv
import os
import json
import time
import torch
from flmr import search_custom_collection, create_searcher

# === Config ===
TOTAL_RUNS = 1000
QUERY_PATH = "/mydata/EVQA/queries.json"
EMBED_PATH = "/mydata/EVQA/qembeds.pt"
DEFAULT_NUM_STREAMS = 2


# === Search Wrapper ===
class StepE:
    def __init__(
        self,
        index_root_path='/mydata/EVQA/index/',
        index_experiment_name='EVQA_train_split/',
        index_name='EVQA_PreFLMR_ViT-L',
    ):
        self.searcher = create_searcher(
            index_root_path=index_root_path,
            index_experiment_name=index_experiment_name,
            index_name=index_name,
            nbits=8,
            use_gpu=True,
        )

    def step_E_search(self, queries, query_embeddings):
        result = search_custom_collection(
            searcher=self.searcher,
            queries=queries,
            query_embeddings=query_embeddings,
            num_document_to_retrieve=5,
            centroid_search_batch_size=None,
        )
        return result.todict() if hasattr(result, "todict") else result


# === Benchmarking ===
def run_benchmark_with_streams(queries_dict, query_embeddings, bsize, searcher, num_streams):
    run_times = []

    all_keys = list(queries_dict.keys())
    all_texts = list(queries_dict.values())
    num_queries = len(all_keys)

    stream_list = [torch.cuda.Stream() for _ in range(num_streams)]

    print(f"Running {TOTAL_RUNS} batches using {num_streams} CUDA streams...")

    start_wall_time = time.time()

    for i in range(TOTAL_RUNS):
        start_idx = (i * bsize) % num_queries
        indices = [(start_idx + j) % num_queries for j in range(bsize)]

        batch_keys = [all_keys[idx] for idx in indices]
        batch_texts = [all_texts[idx] for idx in indices]
        batch_embeddings = query_embeddings[indices]
        batch_queries = dict(zip(batch_keys, batch_texts))

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()

        results = [None] * num_streams
        for s in range(num_streams):
            with torch.cuda.stream(stream_list[s]):
                results[s] = searcher.step_E_search(batch_queries, batch_embeddings)

        torch.cuda.synchronize()
        end_event.record()
        torch.cuda.synchronize()

        elapsed = start_event.elapsed_time(end_event) * 1e6  # µs → ns
        run_times.append(elapsed)

    end_wall_time = time.time()
    total_wall_time = end_wall_time - start_wall_time

    return run_times, total_wall_time


# === Main ===
def main(output_dir, pid, bsize, num_streams):
    with open(QUERY_PATH, 'r') as f:
        queries_dict = json.load(f)

    query_embeddings = torch.load(EMBED_PATH)
    if isinstance(query_embeddings, list):
        query_embeddings = torch.cat(query_embeddings, dim=0)

    print(f"Loaded {len(queries_dict)} queries")
    print(f"Query embeddings shape: {query_embeddings.shape}")
    assert query_embeddings.dim() == 3, "Expected [N, T, D] shaped embeddings"

    stepE = StepE()
    run_times, wall_clock_time = run_benchmark_with_streams(queries_dict, query_embeddings, bsize, stepE, num_streams)

    total_queries = bsize * TOTAL_RUNS
    throughput = total_queries / wall_clock_time
    avg_latency_ns = int(sum(run_times) / len(run_times))

    print(f"\n[RESULTS]")
    print(f"Batch size: {bsize}")
    print(f"Streams used: {num_streams}")
    print(f"Total queries processed: {total_queries}")
    print(f"Wall-clock time: {wall_clock_time:.2f} sec")
    print(f"Throughput: {throughput:.2f} queries/sec")
    print(f"Average latency per batch: {avg_latency_ns} ns")

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(
        output_dir,
        f'stepE_bsize{bsize}_streams{num_streams}_runtime{pid}_tp{throughput:.2f}.csv'
    )

    with open(out_file, 'w', newline='') as f:
        csv.writer(f).writerow(run_times)


# === Entry Point ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark StepE Search with CUDA Streams")
    parser.add_argument("-p", "--output_dir", type=str, required=True)
    parser.add_argument("-id", "--pid", type=str, required=True)
    parser.add_argument("-b", "--bsize", type=int, required=True)
    parser.add_argument("-n", "--num_streams", type=int, default=DEFAULT_NUM_STREAMS)
    args = parser.parse_args()

    main(args.output_dir, args.pid, args.bsize, args.num_streams)