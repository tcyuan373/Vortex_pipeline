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
        
        
if __name__ == "__main__": #(B * vector dim)
    input_data = torch.randn(3,768).cuda()
    model = FLMRMultiLayerPerceptron((768, 128*32 //2 , 128*32)).cuda()
    output = model(input_data)
    print(output.shape)