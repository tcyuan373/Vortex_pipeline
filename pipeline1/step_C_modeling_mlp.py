import csv
import time

import torch, os
from torch import nn
from flmr import FLMRConfig
from transformers import BertConfig
from flmr import FLMRConfig, FLMRQueryEncoderTokenizer, FLMRContextEncoderTokenizer, FLMRModelForRetrieval

class StepC:
    def __init__(self):
        self.checkpoint_path = 'LinWeizheDragon/PreFLMR_ViT-L'
        self.flmr_config = FLMRConfig.from_pretrained(self.checkpoint_path)

        transformer_mapping_config_base = self.flmr_config.transformer_mapping_config_base
        transformer_mapping_config = BertConfig.from_pretrained(transformer_mapping_config_base)
        transformer_mapping_config.num_hidden_layers = self.flmr_config.transformer_mapping_num_hidden_layers
        transformer_mapping_config.is_decoder = True
        transformer_mapping_config.add_cross_attention = True

        self.local_model_path = "models_step_C_transformer_mapping_input_linear.pt"

        if not os.path.exists(self.local_model_path):
            print(f'local directory not found, initing from full model...')
            self.query_tokenizer = FLMRQueryEncoderTokenizer.from_pretrained(
                self.checkpoint_path, 
                text_config=self.flmr_config.text_config, 
                subfolder="query_tokenizer")
            self.context_tokenizer = FLMRContextEncoderTokenizer.from_pretrained(
                self.checkpoint_path, 
                text_config=self.flmr_config.text_config, 
                subfolder="context_tokenizer"
            )
            full_model = FLMRModelForRetrieval.from_pretrained(
                self.checkpoint_path,
                query_tokenizer=self.query_tokenizer,
                context_tokenizer=self.context_tokenizer,
            )
            self.transformer_mapping_input_linear = full_model.transformer_mapping_input_linear
            # torch.save(self.transformer_mapping_input_linear.state_dict(), self.local_model_path)

            del full_model

        else:       
            print(f'found local model for step C, now loading...')
            self.transformer_mapping_input_linear = nn.Linear(
                self.flmr_config.vision_config.hidden_size, transformer_mapping_config.hidden_size
            )

            self.transformer_mapping_input_linear.load_state_dict(torch.load(self.local_model_path, weights_only=True))

    def load_model_cuda(self):
        self.transformer_mapping_input_linear.cuda()

    def stepC_output(self, vision_second_last_layer_hidden_states):
        transformer_mapping_input_features = self.transformer_mapping_input_linear(
            vision_second_last_layer_hidden_states
        )
        return transformer_mapping_input_features


if __name__ == "__main__": # Bsize, vision_hidden_size[-2], vision_hidden_size[-1]
    stepc = StepC()
    stepc.load_model_cuda()

    # GPU memory usage after loading model
    print("Allocated memory after loading model:", torch.cuda.memory_allocated())
    print("Reserved memory after loading model:", torch.cuda.memory_reserved())

    load_input_times = []
    run_times = []
    output_to_host_times = []
    bsize = 16

    # CUDA events for accurate profiling
    total_start_event = torch.cuda.Event(enable_timing=True)
    total_end_event = torch.cuda.Event(enable_timing=True)
    # total start time for throughput calculation
    total_start_event.record()

    for i in range(1000):
        # CUDA events for accurate profiling
        mvgpu_start_event = torch.cuda.Event(enable_timing=True)
        mvgpu_end_event = torch.cuda.Event(enable_timing=True)
        model_start_event = torch.cuda.Event(enable_timing=True)
        model_end_event = torch.cuda.Event(enable_timing=True)
        mvcpu_start_event = torch.cuda.Event(enable_timing=True)
        mvcpu_end_event = torch.cuda.Event(enable_timing=True)

        dummy_hidden_states = torch.randn(bsize, 256, 1024)

        # time before put to GPU
        mvgpu_start_event.record()
        dummy_hidden_states = dummy_hidden_states.cuda()
        # time after put to GPU
        mvgpu_end_event.record()
        torch.cuda.synchronize()
        load_input_times.append((mvgpu_start_event.elapsed_time(mvgpu_end_event)) * 1e6)

        # time before running model
        model_start_event.record()
        output = stepc.stepC_output(dummy_hidden_states)
        # time after running model
        model_end_event.record()
        torch.cuda.synchronize()
        run_times.append((model_start_event.elapsed_time(model_end_event)) * 1e6)

        # time before transfer to CPU
        mvcpu_start_event.record()
        output.cpu()
        # time after transfer to CPU
        mvcpu_end_event.record()
        torch.cuda.synchronize()
        output_to_host_times.append((mvcpu_start_event.elapsed_time(mvcpu_end_event)) * 1e6)

    # total end time for throughput calculation
    total_end_event.record()
    torch.cuda.synchronize()
    time_elapsed=(total_start_event.elapsed_time(total_end_event)) * 1e6
    throughput = (1000 * bsize) / (time_elapsed / 1000000000)
    print("Throughput with batch size", bsize, "(queries/s):", throughput)

    runtimes_file = 'step_C_runtime.csv'
    gpu_transfer = 'step_C_transfer_to_gpu.csv'
    cpu_transfer = 'step_C_transfer_to_cpu.csv'

    with open(runtimes_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(run_times)

    with open(gpu_transfer, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(load_input_times)

    with open(cpu_transfer, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(output_to_host_times)

    print(f"transformer mapping input feature shape is: {output.shape}")
