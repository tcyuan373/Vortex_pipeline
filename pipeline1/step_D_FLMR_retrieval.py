import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.utils.cpp_extension import load

from transformers.models.bert.modeling_bert import BertModel
from transformers import BertConfig, BertTokenizer
from transformers.models.bert.modeling_bert import BertEncoder
from transformers import AutoModel

from flmr import FLMRConfig, FLMRQueryEncoderTokenizer
## key idea: replacing the retrival model input with query_text_embed, query_img_embed, context_text_embed, context_img_embed
## through away configs thats redundant

import time
import csv
import argparse

# ENCODER hidden states: 

class step_D_transformer_mapping:
    def __init__(self):
        # super().__init__(FLMRConfig)
        self.checkpoint_path = 'LinWeizheDragon/PreFLMR_ViT-L'
        self.flmr_config = FLMRConfig.from_pretrained(self.checkpoint_path)
        self.transformer_mapping_cross_attention_length = 32 # each element, modeling how many tokens upfront
        self.skiplist = []
        self.tokenizer = FLMRQueryEncoderTokenizer.from_pretrained(self.checkpoint_path,
                                                                    text_config=self.flmr_config.text_config,
                                                                    subfolder="query_tokenizer")
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
        
        if self.flmr_config.mask_instruction_token is not None:
            self.mask_instruction = True
            # obtain the token id of the instruction token
            self.instruction_token_id = self.tokenizer.encode(
                self.flmr_config.mask_instruction_token, add_special_tokens=False
            )[0]
        else:
            self.mask_instruction = False
                    
        
    def mask(self, input_ids, skiplist):
        return [[(x not in skiplist) and (x != 0) for x in d] for d in input_ids.cpu().tolist()]
    
    
    def query_mask(self, input_ids, skiplist):
        if not self.mask_instruction:
            return self.mask(input_ids, skiplist)

        # find the position of end of instruction in input_ids
        # mask the tokens before the position
        sep_id = self.instruction_token_id
        sep_positions = torch.argmax((input_ids == sep_id).int(), dim=1).tolist()
        # if any of the positions is lower than 1, set to 1
        for i, x in enumerate(sep_positions):
            if x < 1:
                sep_positions[i] = 1
        mask = [
            [
                (x not in skiplist) and (x != 0) and (index > sep_positions[seq_index] or index < 2)
                for index, x in enumerate(d)
            ]
            for seq_index, d in enumerate(input_ids.cpu().tolist())
        ]
        return mask
    
    
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
        mask = torch.tensor(self.query_mask(input_ids, skiplist=self.skiplist)).unsqueeze(2).float().cuda()
        text_embeddings = text_embeddings * mask
        encoder_mask = torch.ones_like(mask).to(mask.device, dtype=mask.dtype)
        if text_encoder_hidden_states.shape[1] > self.transformer_mapping_cross_attention_length:
            text_encoder_hidden_states = text_encoder_hidden_states[:, :self.transformer_mapping_cross_attention_length]
            encoder_mask = encoder_mask[:, :self.transformer_mapping_cross_attention_length]
        # Obtain cross attention mask
        encoder_extended_attention_mask = self.invert_attention_mask(encoder_mask.squeeze(-1))
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

def perform_model():    
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
    
    # Measure model transfer time
    start_model_transfer_time = time.perf_counter_ns()
    stepD.load_model_cuda()
    end_model_transfer_time = time.perf_counter_ns()
    model_transfer_time = end_model_transfer_time - start_model_transfer_time

    # Measure memory after loading model
    before_allocated_memory = torch.cuda.memory_allocated() 
    before_reserved_memory = torch.cuda.memory_reserved()
    before_allocated_memory_mb = before_allocated_memory / (1024 ** 2)
    before_reserved_memory_mb = before_reserved_memory / (1024 ** 2)
    print(f"After loading model GPU memory allocated: {before_allocated_memory_mb:.2f} MB")
    print(f"After loading model GPU memory reserved: {before_reserved_memory_mb:.2f} MB")

    # Measure step D latency
    start_time = time.perf_counter_ns()
    query_embeddings = stepD.cross_attn_embedding(dummy_ids, dummy_text_embeddings, dummy_text_encoder_hidden_states, dummy_vision_embeddings, dummy_tf_mapping_input_features)
    end_time = time.perf_counter_ns()
    model_run_time = end_time - start_time
    # print(f'query embedding shape is: {query_embeddings.shape}')
    
    # Measure memory after running the model
    after_allocated_memory = torch.cuda.memory_allocated() 
    after_reserved_memory = torch.cuda.memory_reserved()
    after_allocated_memory_mb = after_allocated_memory / (1024 ** 2)
    after_reserved_memory_mb = after_reserved_memory / (1024 ** 2)
    print(f"After running model GPU memory allocated: {after_allocated_memory_mb:.2f} MB")
    print(f"After running model GPU memory reserved: {after_reserved_memory_mb:.2f} MB")

    # Measure output transfer time
    start_output_transfer_time = time.perf_counter_ns()
    query_embeddings.cpu()
    end_output_transfer_time = time.perf_counter_ns()
    output_transfer_time = end_output_transfer_time - start_output_transfer_time
    return model_run_time, model_transfer_time, output_transfer_time

def benchmark_model(runtime_file, transfer_time_file, num_times):
    model_run_times = []
    transfer_times = []
    for run_id in range(num_times):
        model_run_time, model_transfer_time, output_transfer_time = perform_model()
        model_run_times.append((run_id + 1, model_run_time))
        transfer_times.append((
                    run_id + 1,
                    model_transfer_time,
                    output_transfer_time
        ))

    with open(runtime_file, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["run_id", "model_run_time(ns)"])
        writer.writerows(model_run_times)
    
    with open(transfer_time_file, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["run_id", "model_transfer_time(ns)", "output_transfer_time(ns)"])
        writer.writerows(transfer_times)

if __name__ == "__main__": # Bsize, vision_hidden_size[-2], vision_hidden_size[-1]
    parser = argparse.ArgumentParser(description="Benchmark the latency step C and save results to a CSV file.")
    parser.add_argument("--runtime_file", type=str, required=True, help="The name of the CSV file to save the model latency results.")
    parser.add_argument("--transfer_time_file", type=str, required=True, help="The name of the CSV file to save the transfer time results.")
    parser.add_argument("--num_times", type=int, required=True, help="The number of times to run the benchmark.")
    args = parser.parse_args()
    benchmark_model(args.runtime_file, args.transfer_time_file, args.num_times)