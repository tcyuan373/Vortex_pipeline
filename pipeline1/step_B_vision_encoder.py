import csv
import numpy
import os
import time

import torch
from torch import nn
from flmr import FLMRConfig, FLMRQueryEncoderTokenizer, FLMRContextEncoderTokenizer, FLMRModelForRetrieval, FLMRVisionModel
from transformers import AutoImageProcessor
from PIL import Image


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


class StepB:
    def __init__(self):
        self.checkpoint_path = 'LinWeizheDragon/PreFLMR_ViT-L'
        self.flmr_config = FLMRConfig.from_pretrained(self.checkpoint_path)
        self.local_encoder_path = 'models_step_B_vision_encoder.pt'
        self.local_projection_path = 'models_step_B_vision_projection.pt'
        self.image_processor = AutoImageProcessor.from_pretrained('openai/clip-vit-large-patch14')

        if not os.path.exists(self.local_encoder_path) and not os.path.exists(self.local_projection_path):
            print('local model not found, initing from full model...')
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

            if self.flmr_config.use_vision_encoder:
                self.query_vision_encoder = full_model.query_vision_encoder

            self.query_vision_projection = full_model.query_vision_projection
            # torch.save(self.query_vision_encoder.state_dict(), self.local_encoder_path)
            # torch.save(self.query_vision_projection.state_dict(), self.local_projection_path)
            del full_model

        else:
            print(f'found local model for step B, now loading...')
            self.query_vision_encoder = FLMRVisionModel(self.flmr_config.vision_config)
            self.query_vision_projection = FLMRMultiLayerPerceptron(
                    (
                        self.flmr_config.vision_config.hidden_size,
                        (self.flmr_config.dim * self.flmr_config.mapping_network_prefix_length) // 2,
                        self.flmr_config.dim * self.flmr_config.mapping_network_prefix_length,
                    )
                )
            self.query_vision_encoder.load_state_dict(torch.load(self.local_encoder_path, weights_only=False))
            self.query_vision_projection.load_state_dict(torch.load(self.local_projection_path, weights_only=False))

        self.device = 'cuda'
        
        
    def load_model_cuda(self):
        self.query_vision_projection.cuda()
        self.query_vision_encoder.cuda()

    def StepB_output(self, pixel_values, batch_size, load_input_times):
        # time before put to GPU
        mvgpu_start=time.perf_counter_ns()
        # Forward the vision encoder
        pixel_values = pixel_values.to(self.device).cuda()
        # time after put to GPU
        mvgpu_end=time.perf_counter_ns()
        load_input_times.append(mvgpu_end-mvgpu_start)

        if len(pixel_values.shape) == 5:
            # Multiple ROIs are provided
            # merge the first two dimensions
            pixel_values = pixel_values.reshape(
                -1, pixel_values.shape[2], pixel_values.shape[3], pixel_values.shape[4]
            )

        vision_encoder_outputs = self.query_vision_encoder(pixel_values, output_hidden_states=True)
        vision_embeddings = vision_encoder_outputs.last_hidden_state[:, 0]

        vision_embeddings = self.query_vision_projection(vision_embeddings)
        vision_embeddings = vision_embeddings.view(batch_size, -1, self.flmr_config.dim)

        vision_second_last_layer_hidden_states = vision_encoder_outputs.hidden_states[-2][:, 1:]

        return vision_embeddings, vision_second_last_layer_hidden_states


if __name__=="__main__":
    img_root = './images'
    img_paths = [os.path.join(img_root, item) for item in os.listdir(img_root)]
    list_of_images = []
    for img_path in img_paths:
        image = Image.open(img_path).convert("RGB")
        list_of_images.append(image)

    pixel_values = []
    image_processor = AutoImageProcessor.from_pretrained('openai/clip-vit-large-patch14')
    for img in list_of_images:
        encoded = image_processor(img, return_tensors="pt")
        pixel_values.append(encoded.pixel_values)
    pixel_values = torch.stack(pixel_values, dim=0)
    batch_size = pixel_values.shape[0]

    stepb = StepB()
    stepb.load_model_cuda()

    # GPU memory usage after loading model
    print("Allocated memory after loading model:", torch.cuda.memory_allocated())
    print("Reserved memory after loading model:", torch.cuda.memory_reserved())

    load_input_times = []
    run_times = []
    embeddings_to_host_times = []
    hidden_states_to_host_times = []

    # total start time for throughput calculation
    start=time.perf_counter_ns()

    for i in range(1000):
        # time before running model
        model_start=time.perf_counter_ns()
        vision_embeddings, vision_second_last_layer_hidden_states = stepb.StepB_output(pixel_values, batch_size, load_input_times)
        # time after running model
        model_end=time.perf_counter_ns()
        run_times.append(model_end-model_start)

        # time before transferring vision embeddings to CPU
        embeddings_start=time.perf_counter_ns()
        vision_embeddings.cpu()
        # time after transferring vision embeddings to CPU
        embeddings_end=time.perf_counter_ns()
        embeddings_to_host_times.append(embeddings_end-embeddings_start)

        # time before transferring hidden states to CPU
        hstates_start=time.perf_counter_ns()
        vision_second_last_layer_hidden_states.cpu()
        # time after transferring hidden states to CPU
        hstates_end=time.perf_counter_ns()
        hidden_states_to_host_times.append(hstates_end-hstates_start)

    # total end time for throughput calculation
    end=time.perf_counter_ns()
    time_elapsed=end-start
    throughput = (1000 * batch_size) / (time_elapsed / 1000000000)
    print("Throughput with batch size", batch_size, "(queries/s):", throughput)

    # subtract transfer time from runtime
    run_times = numpy.subtract(run_times, load_input_times)

    runtimes_file = 'step_B_runtime.csv'
    gpu_transfer = 'step_B_transfer_to_gpu.csv'
    embeddings_cpu_transfer = 'step_B_transfer_embeddings_to_cpu.csv'
    hstates_cpu_transfer = 'step_B_transfer_hidden_states_to_cpu.csv'

    with open(runtimes_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(run_times)

    with open(gpu_transfer, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(load_input_times)

    with open(embeddings_cpu_transfer, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(embeddings_to_host_times)

    with open(hstates_cpu_transfer, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(hidden_states_to_host_times)

    print(f"vision_embedding shape is : {vision_embeddings.shape} | vision penultimate shape is: {vision_second_last_layer_hidden_states.shape}")
