#!/usr/bin/env python3
import argparse
import csv
import os
import time
import warnings
from PIL import Image
import torch
from torch import nn
import numpy as np
from transformers import AutoImageProcessor

from flmr import (
    FLMRConfig,
    FLMRQueryEncoderTokenizer,
    FLMRContextEncoderTokenizer,
    FLMRModelForRetrieval,
    FLMRVisionModel
)

warnings.filterwarnings("ignore")

TOTAL_RUNS = 1000
IMAGE_DIR = './images'


class FLMRMultiLayerPerceptron(nn.Module):
    def __init__(self, sizes, bias=True, act=nn.Tanh):
        super().__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class StepBWrapper:
    def __init__(self):
        self.checkpoint_path = 'LinWeizheDragon/PreFLMR_ViT-L'
        self.local_encoder_path = '/mydata/EVQA/models/models_step_B_vision_encoder.pt'
        self.local_projection_path = '/mydata/EVQA/models/models_step_B_vision_projection.pt'
        self.flmr_config = FLMRConfig.from_pretrained(self.checkpoint_path)
        self.image_processor = AutoImageProcessor.from_pretrained('openai/clip-vit-large-patch14')

        if not os.path.exists(self.local_encoder_path) or not os.path.exists(self.local_projection_path):
            print("Local model not found, loading full model...")
            self.query_tokenizer = FLMRQueryEncoderTokenizer.from_pretrained(
                self.checkpoint_path,
                text_config=self.flmr_config.text_config,
                subfolder="query_tokenizer"
            )
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
            self.query_vision_encoder = full_model.query_vision_encoder
            self.query_vision_projection = full_model.query_vision_projection
            del full_model
        else:
            print("Found local model files, loading...")
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

        self.query_vision_encoder.to('cuda').eval()
        self.query_vision_projection.to('cuda').eval()

    def encode(self, images):
        pixel_values = []
        for img in images:
            encoded = self.image_processor(img, return_tensors="pt", padding=True)
            pixel_values.append(encoded.pixel_values)
        pixel_values = torch.stack(pixel_values, dim=0)

        batch_size = pixel_values.shape[0]
        pixel_values = pixel_values.to('cuda')

        if len(pixel_values.shape) == 5:
            pixel_values = pixel_values.reshape(
                -1, pixel_values.shape[2], pixel_values.shape[3], pixel_values.shape[4]
            )

        with torch.no_grad():
            vision_out = self.query_vision_encoder(pixel_values, output_hidden_states=True)
            embeddings = self.query_vision_projection(vision_out.last_hidden_state[:, 0])
            embeddings = embeddings.view(batch_size, -1, self.flmr_config.dim)
        return embeddings


def load_images():
    img_paths = [os.path.join(IMAGE_DIR, fname) for fname in os.listdir(IMAGE_DIR) if fname.lower().endswith(('.jpg', '.png', '.jpeg'))]
    images = [Image.open(p).convert("RGB") for p in img_paths]
    return images


def run_benchmark(images, bsize, encoder):
    run_times = []
    j = 0

    for _ in range(TOTAL_RUNS):
        image_batch = [images[j % len(images)] for _ in range(bsize)]
        j += bsize

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        _ = encoder.encode(image_batch)

        end_event.record()
        torch.cuda.synchronize()
        

        run_times.append(start_event.elapsed_time(end_event) * 1e6)  # microseconds to nanoseconds

    return run_times


def main(output_dir, pid, bsize):
    images = load_images()
    encoder = StepBWrapper()
    run_times = run_benchmark(images, bsize, encoder)

    throughput = (bsize * len(run_times)) / (sum(run_times) / 1e9)
    avg_latency = int(sum(run_times) / len(run_times))

    print(f"Batch size {bsize}, throughput: {throughput:.2f} images/sec")
    print(f"Average latency per batch: {avg_latency} ns")

    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, f'stepB_bsize{bsize}_runtime{pid}_tp{throughput:.2f}.csv')

    with open(out_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(run_times)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark StepB FLMR Vision Encoder")
    parser.add_argument('-p', '--output_dir', type=str, required=True, help='Directory to store output CSV')
    parser.add_argument('-id', '--pid', type=str, required=True, help='String identifier for MIG setup')
    parser.add_argument('-b', '--bsize', type=int, required=True, help='Batch size for image encoding')
    args = parser.parse_args()

    main(args.output_dir, args.pid, args.bsize)
