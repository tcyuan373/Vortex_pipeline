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

from flmr import FLMRConfig, FLMRVisionModel
from transformers import AutoImageProcessor
from torchinfo import summary
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
        self.image_processor = AutoImageProcessor.from_pretrained('openai/clip-vit-large-patch14')
        if self.flmr_config.use_vision_encoder:
            self.query_vision_encoder = FLMRVisionModel(self.flmr_config.vision_config)
            
        self.query_vision_projection = FLMRMultiLayerPerceptron(
                (
                    self.flmr_config.vision_config.hidden_size,
                    (self.flmr_config.dim * self.flmr_config.mapping_network_prefix_length) // 2,
                    self.flmr_config.dim * self.flmr_config.mapping_network_prefix_length,
                )
            )
        self.device = 'cuda'
        
    def load_model_cuda(self):
        self.query_vision_projection.cuda()
        self.query_vision_encoder.cuda()
        
    def StepB_output(self, list_of_images):
        pixel_values = []
        for img in list_of_images:

            encoded = self.image_processor(img, return_tensors="pt")
            pixel_values.append(encoded.pixel_values)
        pixel_values = torch.stack(pixel_values, dim=0)
            
        batch_size = pixel_values.shape[0]
        # Forward the vision encoder
        pixel_values = pixel_values.to(self.device)
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
    stepb = StepB()
    stepb.load_model_cuda()
    vision_embeddings, vision_second_last_layer_hidden_states= stepb.StepB_output(list_of_images)
    vision_embeddings.cpu()
    vision_second_last_layer_hidden_states.cpu()
    print(f"vision_embedding shape is : {vision_embeddings.shape} | vision penultimate shape is: {vision_second_last_layer_hidden_states.shape}")