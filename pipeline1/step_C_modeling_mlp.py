# transformer mapping : Linear + BERT_base + out_linear
# Text Enc: BERT_base + proj(linear)
# Vis Enc: CLIP_ViT_G + proj(linear)
# hardcode configs
# init with default config
# cpp extension
# utilities colbert score, world size, world rank
# handling attention mask

#### late interaction size, 128
#### mapping network prefix len, 32

import copy
import os
import pathlib
import string
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.utils.cpp_extension import load
import time
import csv
import argparse

from transformers import AutoModel, AutoConfig
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from transformers.models.bert.modeling_bert import BertModel
from transformers.models.clip import CLIPVisionModel
# from .configuration_flmr import FLMRConfig, FLMRTextConfig, FLMRVisionConfig
# from .tokenization_flmr import FLMRQueryEncoderTokenizer, FLMRContextEncoderTokenizer
# from .tokenization_flmr_fast import FLMRQueryEncoderTokenizerFast, FLMRContextEncoderTokenizerFast
# from .flmr_utils import (
#     colbert_score,
#     colbert_score_reduce,
#     get_rank,
#     get_world_size,
# )

logger = logging.get_logger(__name__)



class FLMRMultiLayerPerceptron(nn.Module):
    """
    A simple multi-layer perceptron with an activation function. This can be used as the mapping network in the FLMR model.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def __init__(self, sizes, bias=True, act=nn.Tanh):
        super(FLMRMultiLayerPerceptron, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)

# MLP sizes
# (
#     self.vision_encoder_embedding_size, 768
#     (self.late_interaction_embedding_size * self.mapping_network_prefix_length) // 2,
#     self.late_interaction_embedding_size * self.mapping_network_prefix_length,   128*32
# )

def perform_model():
    # Measure transfer time of the input data from host to GPU
    input_data = torch.randn(3,768)
    start_input_transfer_time = time.perf_counter_ns()
    input_data = input_data.cuda() 
    end_input_transfer_time = time.perf_counter_ns()
    input_transfer_elapsed_time_ns = end_input_transfer_time - start_input_transfer_time

    # Measure transfer time of the model from host to GPU
    model = FLMRMultiLayerPerceptron((768, 128*32 //2 , 128*32))
    start_model_transfer_time = time.perf_counter_ns()
    model = model.cuda()
    end_model_transfer_time = time.perf_counter_ns()
    model_transfer_elapsed_time_ns = end_model_transfer_time - start_model_transfer_time

    # Measure memory after loading model
    # before_allocated_memory = torch.cuda.memory_allocated() 
    # before_reserved_memory = torch.cuda.memory_reserved()
    # before_allocated_memory_mb = before_allocated_memory / (1024 ** 2)
    # before_reserved_memory_mb = before_reserved_memory / (1024 ** 2)
    # print(f"After loading model memory allocated: {before_allocated_memory_mb:.2f} MB")
    # print(f"After loading model memory reserved: {before_reserved_memory_mb:.2f} MB")

    # Measure latency of running the model
    start_time = time.perf_counter_ns()
    output = model(input_data)
    end_time = time.perf_counter_ns()
    elapsed_time_ns = end_time - start_time

    # Measure memory after running the model
    # after_allocated_memory = torch.cuda.memory_allocated() 
    # after_reserved_memory = torch.cuda.memory_reserved()
    # after_allocated_memory_mb = after_allocated_memory / (1024 ** 2)
    # after_reserved_memory_mb = after_reserved_memory / (1024 ** 2)
    # print(f"After running model memory allocated: {after_allocated_memory_mb:.2f} MB")
    # print(f"After running model memory reserved: {after_reserved_memory_mb:.2f} MB")

    # Measure time to transfer output from GPU to host
    start_output_transfer_time = time.perf_counter_ns()
    output_cpu = output.cpu()
    end_output_transfer_time = time.perf_counter_ns()
    output_transfer_elapsed_time_ns = end_output_transfer_time - start_output_transfer_time
    return elapsed_time_ns, input_transfer_elapsed_time_ns, model_transfer_elapsed_time_ns, output_transfer_elapsed_time_ns

def benchmark_model(result_file, transfer_time_file, num_times):
    elapsed_times = []
    input_transfer_times = []
    model_transfer_times = []
    output_transfer_times = []
    for run_id in range(num_times):
        elapsed_time_ns, input_transfer_time_ns, model_transfer_time_ns, output_transfer_time_ns = perform_model()
        elapsed_times.append((run_id + 1, elapsed_time_ns))
        input_transfer_times.append((run_id + 1, input_transfer_time_ns))
        model_transfer_times.append((run_id + 1, model_transfer_time_ns))
        output_transfer_times.append((run_id + 1, output_transfer_time_ns))

    with open(result_file, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["run_id", "elapsed_time_ns"])
        writer.writerows(elapsed_times)
    
    with open(transfer_time_file, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["run_id", "input_transfer_time_ns", "model_transfer_time_ns", "output_transfer_time_ns"])
        writer.writerows(elapsed_times)

if __name__ == "__main__": #(B * vector dim)
    parser = argparse.ArgumentParser(description="Benchmark the latency step C and save results to a CSV file.")
    parser.add_argument("--runtime_file", type=str, required=True, help="The name of the CSV file to save the latency results.")
    parser.add_argument("--transfer_time_file", type=str, required=True, help="The name of the CSV file to save the transfer time results.")
    parser.add_argument("--num_times", type=int, required=True, help="The number of times to run the benchmark.")
    args = parser.parse_args()
    benchmark_model(args.result_file, args.transfer_time_file, args.num_times)
