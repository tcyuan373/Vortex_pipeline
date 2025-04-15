#!/usr/bin/env python3
import argparse
import csv
import os
import time
import json
import warnings
import numpy as np
import torch
from torch import nn

from flmr import (
    FLMRConfig,
    FLMRQueryEncoderTokenizer,
    FLMRContextEncoderTokenizer,
    FLMRModelForRetrieval,
    FLMRTextModel
)

warnings.filterwarnings("ignore")

TOTAL_RUNS = 1000
QUERY_JSON_PATH = "/mydata/EVQA/queries.json"


class StepAWrapper:
    def __init__(self):
        self.checkpoint_path = 'LinWeizheDragon/PreFLMR_ViT-L'
        self.local_encoder_path = 'models_step_A_query_text_encoder.pt'
        self.local_projection_path = 'models_step_A_query_text_linear.pt'

        self.flmr_config = FLMRConfig.from_pretrained(self.checkpoint_path)
        self.query_tokenizer = FLMRQueryEncoderTokenizer.from_pretrained(
            self.checkpoint_path, text_config=self.flmr_config.text_config, subfolder="query_tokenizer"
        )
        self.context_tokenizer = FLMRContextEncoderTokenizer.from_pretrained(
            self.checkpoint_path, text_config=self.flmr_config.text_config, subfolder="context_tokenizer"
        )

        if not os.path.exists(self.local_encoder_path) or not os.path.exists(self.local_projection_path):
            print('Local model not found, loading full model...')
            full_model = FLMRModelForRetrieval.from_pretrained(
                self.checkpoint_path,
                query_tokenizer=self.query_tokenizer,
                context_tokenizer=self.context_tokenizer,
            )
            self.query_text_encoder = full_model.query_text_encoder
            self.query_text_encoder_linear = full_model.query_text_encoder_linear
            del full_model
        else:
            print('Found local model files. Loading...')
            self.query_text_encoder = FLMRTextModel(self.flmr_config.text_config)
            self.query_text_encoder.load_state_dict(torch.load(self.local_encoder_path, weights_only=True))
            self.query_text_encoder_linear = nn.Linear(self.flmr_config.text_config.hidden_size, self.flmr_config.dim, bias=False)
            self.query_text_encoder_linear.load_state_dict(torch.load(self.local_projection_path, weights_only=True))

        self.query_text_encoder.to('cuda')
        self.query_text_encoder_linear.to('cuda')
        self.query_text_encoder.eval()
        self.query_text_encoder_linear.eval()

    def encode(self, input_text_list):
        encoded_inputs = self.query_tokenizer(input_text_list)
        input_ids = encoded_inputs['input_ids'].to('cuda')
        attention_mask = encoded_inputs['attention_mask'].to('cuda')

        with torch.no_grad():
            output = self.query_text_encoder(input_ids=input_ids, attention_mask=attention_mask)
            hidden_states = output[0]
            embeddings = self.query_text_encoder_linear(hidden_states)

        return embeddings


def run_benchmark(data, bsize, encoder):
    run_times = []
    j = 0

    for _ in range(TOTAL_RUNS):
        query_list = [data[j % len(data)] for _ in range(bsize)]
        j += bsize

        model_start_event = torch.cuda.Event(enable_timing=True)
        model_end_event = torch.cuda.Event(enable_timing=True)

        model_start_event.record()
        _ = encoder.encode(query_list)
        model_end_event.record()
        torch.cuda.synchronize()

        run_times.append(model_start_event.elapsed_time(model_end_event) * 1e6)  # µs to ns

    return run_times


def main(output_dir, pid, bsize):
    with open(QUERY_JSON_PATH, 'r') as f:
        queries_dict = json.load(f)
    data = list(queries_dict.values())

    encoder = StepAWrapper()
    run_times = run_benchmark(data, bsize, encoder)

    throughput = (bsize * len(run_times)) / (sum(run_times) / 1e9)
    avg_latency = int(sum(run_times) / len(run_times))

    print(f"Batch size {bsize}, throughput is {throughput:.2f} queries/sec")
    print(f"Average latency per batch: {avg_latency} ns")

    os.makedirs(output_dir, exist_ok=True)
    runtimes_file = os.path.join(
        output_dir,
        f'stepA_bsize{bsize}_runtime{pid}_tp{throughput:.2f}.csv'
    )

    with open(runtimes_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(run_times)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark StepA FLMR Encoder")
    parser.add_argument('-p', '--output_dir', type=str, required=True, help='Directory to store output CSV')
    parser.add_argument('-id', '--pid', type=str, required=True, help='String identifier for MIG setup')
    parser.add_argument('-b', '--bsize', type=int, required=True, help='Batch size for encoding')
    args = parser.parse_args()

    main(args.output_dir, args.pid, args.bsize)