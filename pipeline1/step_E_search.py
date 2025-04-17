#!/usr/bin/env python3
import argparse
import csv
import os
import json
import torch
import numpy as np
from flmr import search_custom_collection, create_searcher


TOTAL_RUNS = 1000
QUERY_PATH = "/mydata/EVQA/queries.json"
EMBED_PATH = "/mydata/EVQA/qembeds.pt"


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
        return result.todict()


def run_benchmark(queries_dict, query_embeddings, bsize, searcher):
    run_times = []
    all_keys = list(queries_dict.keys())
    all_texts = list(queries_dict.values())
    num_queries = len(all_keys)

    for i in range(TOTAL_RUNS):
        start_idx = (i * bsize) % num_queries
        indices = [(start_idx + j) % num_queries for j in range(bsize)]
        
        batch_keys = [all_keys[idx] for idx in indices]
        batch_texts = [all_texts[idx] for idx in indices]
        batch_embeddings = query_embeddings[indices]

        queries_batch = dict(zip(batch_keys, batch_texts))
        
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        res = searcher.step_E_search(queries_batch, batch_embeddings)
        ret = []
        for i in res:
            ret.append(res[i])
        end_event.record()
        torch.cuda.synchronize()
        
        elapsed = start_event.elapsed_time(end_event) * 1e6  # microseconds to nanoseconds
        run_times.append(elapsed)

    return run_times


def main(output_dir, pid, bsize):
    with open(QUERY_PATH, 'r') as f:
        queries_dict = json.load(f)

    query_embeddings = torch.load(EMBED_PATH)
    print(f"query embeddings type: {type(query_embeddings)}")

    # If it's a list of tensors, stack into a single tensor
    if isinstance(query_embeddings, list):
        query_embeddings = torch.cat(query_embeddings, dim=0)

    print(f"Final stacked query embeddings shape: {query_embeddings.shape}")
    assert query_embeddings.dim() == 3, "Expected [N, T, D] shaped tensor"

    stepE = StepE()
    run_times = run_benchmark(queries_dict, query_embeddings, bsize, stepE)

    throughput = (bsize * len(run_times)) / (sum(run_times) / 1e9)
    avg_latency = int(sum(run_times) / len(run_times))

    print(f"Batch size {bsize}, throughput: {throughput:.2f} queries/sec")
    print(f"Average latency per batch (StepE): {avg_latency} ns")

    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, f'stepE_bsize{bsize}_runtime{pid}_tp{throughput:.2f}.csv')

    with open(out_file, mode='w', newline='') as f:
        csv.writer(f).writerow(run_times)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark StepE Search (query -> ranking)")
    parser.add_argument("-p", "--output_dir", type=str, required=True)
    parser.add_argument("-id", "--pid", type=str, required=True)
    parser.add_argument("-b", "--bsize", type=int, required=True)
    args = parser.parse_args()

    main(args.output_dir, args.pid, args.bsize)
