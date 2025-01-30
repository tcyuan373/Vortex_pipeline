# pretrained_uncased_BERT with linear projection
# add load project layer weights
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


from flmr import FLMRConfig, FLMRQueryEncoderTokenizer, FLMRTextModel
# input text strings
# output: text_embeddings, input_ids[gen mask], encoder_hidden_states

class StepA:
    def __init__(self,):
        self.checkpoint_path = 'LinWeizheDragon/PreFLMR_ViT-L'
        self.flmr_config = FLMRConfig.from_pretrained(self.checkpoint_path)
        
        self.tokenizer = FLMRQueryEncoderTokenizer.from_pretrained(self.checkpoint_path,
                                                                    text_config=self.flmr_config.text_config,
                                                                    subfolder="query_tokenizer")
        self.context_text_encoder = FLMRTextModel(self.flmr_config.text_config)
        self.context_text_encoder_linear = nn.Linear(self.flmr_config.text_config.hidden_size, self.flmr_config.dim, bias=False)
        self.skiplist = []
        
        if self.flmr_config.mask_instruction_token is not None:
            self.mask_instruction = True
            # obtain the token id of the instruction token
            self.instruction_token_id = self.tokenizer.encode(
                self.flmr_config.mask_instruction_token, add_special_tokens=False
            )[0]
        else:
            self.mask_instruction = False
    
    def load_model_cuda(self):
        self.context_text_encoder_linear.to('cuda')
        self.context_text_encoder.to('cuda')
    
    
    def stepA_output(
        self,
        input_text_sequence,
    ):

        # query sentences: bsize of sentences
        encoded_inputs      = self.tokenizer(input_text_sequence)
        input_ids           = encoded_inputs['input_ids'].to(self.context_text_encoder.device)
        attention_mask      = encoded_inputs['attention_mask'].to(self.context_text_encoder.device)
        
        text_encoder_outputs = self.context_text_encoder(input_ids=input_ids,attention_mask=attention_mask,)
        text_encoder_hidden_states = text_encoder_outputs[0]
        text_embeddings = self.context_text_encoder_linear(text_encoder_hidden_states)

        # note, text_embeddings not masked yet here!!!!
        
        return text_embeddings, input_ids, text_encoder_hidden_states
    
    
if __name__ == '__main__':
        raw_sentences = ['Hello World', 'This is a test text sequence', 'I have a cute puppy', "my puppy's name is GOJI"]
        stepA = StepA()
        stepA.load_model_cuda()
        txt_embed, input_ids, txt_encoder_hs = stepA.stepA_output(raw_sentences)
        print(f'text embedding shape: {txt_embed.shape} | input ids shape: {input_ids.shape} | hidden_states shape: {txt_encoder_hs.shape}')
