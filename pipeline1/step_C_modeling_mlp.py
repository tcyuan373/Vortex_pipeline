import torch
import torch.distributed as dist
from torch import Tensor, nn
from flmr import FLMRConfig
from transformers import BertConfig

import time
import csv
import argparse

class StepC:
    def __init__(self):
        self.checkpoint_path = 'LinWeizheDragon/PreFLMR_ViT-L'
        self.flmr_config = FLMRConfig.from_pretrained(self.checkpoint_path)
        
        transformer_mapping_config_base = self.flmr_config.transformer_mapping_config_base
        transformer_mapping_config = BertConfig.from_pretrained(transformer_mapping_config_base)
        transformer_mapping_config.num_hidden_layers = self.flmr_config.transformer_mapping_num_hidden_layers
        transformer_mapping_config.is_decoder = True
        transformer_mapping_config.add_cross_attention = True
        
        self.transformer_mapping_input_linear = nn.Linear(
            self.flmr_config.vision_config.hidden_size, transformer_mapping_config.hidden_size
        )
        
    def load_model_cuda(self):
        self.transformer_mapping_input_linear.cuda()
        
    def stepC_output(self, vision_second_last_layer_hidden_states):
        transformer_mapping_input_features = self.transformer_mapping_input_linear(
            vision_second_last_layer_hidden_states
        )
        
        return transformer_mapping_input_features

def perform_model():    
    stepc = StepC()
    # Measure model transfer time
    start_model_transfer_time = time.perf_counter_ns()
    stepc.load_model_cuda()
    end_model_transfer_time = time.perf_counter_ns()
    model_transfer_time = end_model_transfer_time - start_model_transfer_time
    bsize = 16

    # Measure data transfer time
    dummy_hidden_states = torch.randn(bsize, 256, 1024)
    start_data_transfer_time = time.perf_counter_ns()
    dummy_hidden_states = dummy_hidden_states.cuda()
    end_data_transfer_time = time.perf_counter_ns()
    data_transfer_time = end_data_transfer_time - start_data_transfer_time

    # Measure step C latency
    start_time = time.perf_counter_ns()
    output = stepc.stepC_output(dummy_hidden_states)
    end_time = time.perf_counter_ns()
    model_run_time = end_time - start_time

    # Measure output transfer time
    start_output_transfer_time = time.perf_counter_ns()
    output.cpu()
    end_output_transfer_time = time.perf_counter_ns()
    output_transfer_time = end_output_transfer_time - start_output_transfer_time
    # print(f"transformer mapping input feature shape is: {output.shape}")
    return model_run_time, model_transfer_time, data_transfer_time, output_transfer_time

def benchmark_model(runtime_file, transfer_time_file, num_times):
    model_run_times = []
    transfer_times = []
    for run_id in range(num_times):
        model_run_time, model_transfer_time, data_transfer_time, output_transfer_time = perform_model()
        model_run_times.append((run_id + 1, model_run_time))
        transfer_times.append((
                    run_id + 1,
                    model_transfer_time,
                    data_transfer_time,
                    output_transfer_time
        ))

    with open(runtime_file, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["run_id", "model_run_time"])
        writer.writerows(model_run_times)
    
    with open(transfer_time_file, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["run_id", "model_transfer_time", "data_transfer_time", "output_transfer_time"])
        writer.writerows(transfer_times)

if __name__ == "__main__": # Bsize, vision_hidden_size[-2], vision_hidden_size[-1]
    parser = argparse.ArgumentParser(description="Benchmark the latency step C and save results to a CSV file.")
    parser.add_argument("--runtime_file", type=str, required=True, help="The name of the CSV file to save the model latency results.")
    parser.add_argument("--transfer_time_file", type=str, required=True, help="The name of the CSV file to save the transfer time results.")
    parser.add_argument("--num_times", type=int, required=True, help="The number of times to run the benchmark.")
    args = parser.parse_args()
    benchmark_model(args.runtime_file, args.transfer_time_file, args.num_times)