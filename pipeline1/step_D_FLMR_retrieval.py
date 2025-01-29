import copy, random
import os
import pathlib
import string
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.utils.cpp_extension import load

from transformers.models.bert.modeling_bert import BertModel
from transformers import BertConfig, BertTokenizer
from transformers.models.bert.modeling_bert import BertEncoder
from transformers import AutoModel
## key idea: replacing the retrival model input with query_text_embed, query_img_embed, context_text_embed, context_img_embed
## through away configs thats redundant



# ENCODER hidden states: 

class step_D_transformer_mapping:
    def __init__(self):
        # super().__init__(FLMRConfig)
        self.context_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.transformer_mapping_cross_attention_length = 32 # each element, modeling how many tokens upfront
        self.skiplist = []
        
        # self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.vision_encoder_embedding_size = 1024
        self.late_interaction_embedding_size = 128
        self.transformer_mapping_config_base = 'bert-base-uncased'
        transformer_mapping_config = BertConfig.from_pretrained(self.transformer_mapping_config_base)
        transformer_mapping_config.is_decoder = True
        transformer_mapping_config.add_cross_attention = True
        transformer_mapping_config.num_hidden_layers = 1
        
        
        self.transformer_mapping_input_linear = nn.Linear(
            self.vision_encoder_embedding_size, transformer_mapping_config.hidden_size
        )
        self.transformer_mapping_network = BertEncoder(transformer_mapping_config)
        self.transformer_mapping_output_linear = nn.Linear(
            transformer_mapping_config.hidden_size, self.late_interaction_embedding_size
        )
        self.mask_instruction = False
        self.dummy_model = AutoModel.from_pretrained('bert-base-uncased')
        
        
    def mask(self, input_ids, skiplist):
        return [[(x not in skiplist) and (x != 0) for x in d] for d in input_ids.cpu().tolist()]
    
    
    def query_mask(self, input_ids, skiplist):
        if not self.mask_instruction:
            return self.mask(input_ids, skiplist)
        
        else:
            raise NotImplementedError("TC: No integration with masking instruction tokens yet!!!")
    
    def load_model_cuda(self,):
        self.transformer_mapping_network.cuda()
        self.transformer_mapping_output_linear.cuda()
    
    def invert_attention_mask(self, encoder_attention_mask: Tensor) -> Tensor:
        """
        Invert an attention mask (e.g., switches 0. and 1.).

        Args:
            encoder_attention_mask (`torch.Tensor`): An attention mask.

        Returns:
            `torch.Tensor`: The inverted attention mask.
        """
        if encoder_attention_mask.dim() == 3:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
        if encoder_attention_mask.dim() == 2:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
        # T5 has a mask that can compare sequence ids, we can simulate this here with this transposition
        # Cf. https://github.com/tensorflow/mesh/blob/8d2465e9bc93129b913b5ccc6a59aa97abd96ec6/mesh_tensorflow
        # /transformer/transformer_layers.py#L270
        # encoder_extended_attention_mask = (encoder_extended_attention_mask ==
        # encoder_extended_attention_mask.transpose(-1, -2))
        encoder_extended_attention_mask = encoder_extended_attention_mask.to(dtype=encoder_attention_mask.dtype)  # fp16 compatibility
        encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * torch.finfo(encoder_attention_mask.dtype).min

        return encoder_extended_attention_mask
        
        
    def cross_attn_embedding(
        self,
        input_ids,                              # from Step A
        text_embeddings,                        # from Step A
        text_encoder_hidden_states,             # from Step A        
        vision_embeddings,                      # from Step B
        transformer_mapping_input_features,     # from Step C
    ):
        #preparing mask
        mask = torch.tensor(self.query_mask(input_ids, skiplist=self.skiplist)).cuda()
        encoder_mask = torch.ones_like(mask).to(mask.device, dtype=mask.dtype)
        if text_encoder_hidden_states.shape[1] > self.transformer_mapping_cross_attention_length:
            text_encoder_hidden_states = text_encoder_hidden_states[:, :self.transformer_mapping_cross_attention_length]
            encoder_mask = encoder_mask[:, :self.transformer_mapping_cross_attention_length]
        # Obtain cross attention mask
        encoder_extended_attention_mask = self.dummy_model.invert_attention_mask(encoder_mask.squeeze(-1))
        # Pass through the transformer mapping
        
        
        # ENCODER hidden states: Encoder_bsize, Encoder_seqLen, _
        # ENCODER attention mask: ones_like(encoder_hidden_states)

        transformer_mapping_outputs = self.transformer_mapping_network(
            transformer_mapping_input_features,
            encoder_hidden_states=text_encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
        )
        transformer_mapping_output_features = transformer_mapping_outputs.last_hidden_state
        # Convert the dimension to FLMR dim
        transformer_mapping_output_features = self.transformer_mapping_output_linear(
            transformer_mapping_output_features
        )
        # Merge with the vision embeddings
        vision_embeddings = torch.cat([vision_embeddings, transformer_mapping_output_features], dim=1)
        
        Q = torch.cat([text_embeddings, vision_embeddings], dim=1)
        query_embeddings = torch.nn.functional.normalize(Q, p=2, dim=2).detach().cpu()
        return query_embeddings



if __name__ =="__main__":
    stepD = step_D_transformer_mapping()
    # step_D_transformer_mapping.from_pretrained('')
    bsize = 8
    mapping_network_prefix_length = 32
    query_length = 32
    late_interaction_size = 128
    text_hidden_size = 768
    # vision_penultimate layer output shape [bsize, 256, 1024] if using CLIP ViT
    vision_penultimate_shape = (bsize, 256, 1024)
    # bag = random.randint(500, 1000)
    dummy_ids = torch.randint(0, 10000, (bsize, query_length)).to(torch.int64).cuda()
    dummy_text_embeddings = torch.randn(bsize, query_length, late_interaction_size).cuda()
    dummy_text_encoder_hidden_states = torch.randn(bsize, query_length, text_hidden_size).cuda()
    dummy_vision_embeddings = torch.randn(bsize, mapping_network_prefix_length, late_interaction_size).cuda()
    dummy_tf_mapping_input_features = torch.randn(bsize, vision_penultimate_shape[1], text_hidden_size).cuda()
    
    stepD.load_model_cuda()
    query_embeddings = stepD.cross_attn_embedding(dummy_ids, dummy_text_embeddings, dummy_text_encoder_hidden_states, dummy_vision_embeddings, dummy_tf_mapping_input_features)
    print(f'query embedding shape is: {query_embeddings.shape}')