import csv
import numpy
import time

import torch, os
from torch import Tensor, nn
from transformers import BertConfig
from transformers.models.bert.modeling_bert import BertEncoder
from flmr import FLMRConfig, FLMRQueryEncoderTokenizer, FLMRContextEncoderTokenizer, FLMRModelForRetrieval
## key idea: replacing the retrival model input with query_text_embed, query_img_embed, context_text_embed, context_img_embed
## through away configs thats redundant


# ENCODER hidden states: 

class step_D_transformer_mapping:
    def __init__(self):
        # super().__init__(FLMRConfig)
        self.checkpoint_path = 'LinWeizheDragon/PreFLMR_ViT-L'
        self.flmr_config = FLMRConfig.from_pretrained(self.checkpoint_path)
        self.transformer_mapping_cross_attention_length = 32 # each element, modeling how many tokens upfront
        self.skiplist = []
        self.local_tf_mapping_path = 'models_step_D_transformer_mapping.pt'
        self.local_tf_mapping_output_path = 'models_step_D_transformer_mapping_output.pt'
        self.query_tokenizer = FLMRQueryEncoderTokenizer.from_pretrained(
                    self.checkpoint_path, 
                    text_config=self.flmr_config.text_config, 
                    subfolder="query_tokenizer")


        if not os.path.exists(self.local_tf_mapping_path) and not os.path.exists(self.local_tf_mapping_output_path):
            print(f'local directory not found, initing from full model...')
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
            self.transformer_mapping_network = full_model.transformer_mapping_network
            self.transformer_mapping_output_linear = full_model.transformer_mapping_output_linear
            # torch.save(self.transformer_mapping_network.state_dict(), self.local_tf_mapping_path)
            # torch.save(self.transformer_mapping_output_linear.state_dict(), self.local_tf_mapping_output_path)
        else:
            print(f'found local model for step D, now loading...')            
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
            self.transformer_mapping_network.load_state_dict(torch.load(self.local_tf_mapping_path, weights_only=True))
            self.transformer_mapping_output_linear = nn.Linear(
                transformer_mapping_config.hidden_size, self.late_interaction_embedding_size
            )
            self.transformer_mapping_output_linear.load_state_dict(torch.load(self.local_tf_mapping_output_path, weights_only=True))


        if self.flmr_config.mask_instruction_token is not None:
            self.mask_instruction = True
            # obtain the token id of the instruction token
            self.instruction_token_id = self.query_tokenizer.encode(
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
        output_to_host_times,
        run_id
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

        if run_id==100:
            print("Allocated memory when running model:", torch.cuda.memory_allocated())
            print("Reserved memory when running model:", torch.cuda.memory_reserved())

        # time before transfer to CPU
        mvcpu_start=time.perf_counter_ns()
        query_embeddings = torch.nn.functional.normalize(Q, p=2, dim=2).detach().cpu()
        # time after transfer to CPU
        mvcpu_end=time.perf_counter_ns()
        output_to_host_times.append(mvcpu_end-mvcpu_start)

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

    stepD.load_model_cuda()

    # GPU memory usage after loading model
    print("Allocated memory after loading model:", torch.cuda.memory_allocated())
    print("Reserved memory after loading model:", torch.cuda.memory_reserved())

    load_input_times = []
    run_times = []
    output_to_host_times = []

    # total start time for throughput calculation
    start=time.perf_counter_ns()

    for i in range(1000):
        # time before put to GPU
        mvgpu_start=time.perf_counter_ns()
        dummy_ids = torch.randint(0, 10000, (bsize, query_length)).to(torch.int64).cuda()
        dummy_text_embeddings = torch.randn(bsize, query_length, late_interaction_size).cuda()
        dummy_text_encoder_hidden_states = torch.randn(bsize, query_length, text_hidden_size).cuda()
        dummy_vision_embeddings = torch.randn(bsize, mapping_network_prefix_length, late_interaction_size).cuda()
        dummy_tf_mapping_input_features = torch.randn(bsize, vision_penultimate_shape[1], text_hidden_size).cuda()
        # time after put to GPU
        mvgpu_end=time.perf_counter_ns()
        load_input_times.append(mvgpu_end-mvgpu_start)

        # time before running model
        model_start=time.perf_counter_ns()
        query_embeddings = stepD.cross_attn_embedding(dummy_ids, dummy_text_embeddings, dummy_text_encoder_hidden_states, dummy_vision_embeddings, dummy_tf_mapping_input_features, output_to_host_times, i)
        # time after running model
        model_end=time.perf_counter_ns()
        run_times.append(model_end-model_start)

    # total end time for throughput calculation
    end=time.perf_counter_ns()
    time_elapsed=end-start
    throughput = (1000 * bsize) / (time_elapsed / 1000000000)
    print("Throughput with batch size", bsize, "(queries/s):", throughput)

    # subtract transfer time from runtime
    run_times = numpy.subtract(run_times, output_to_host_times)

    runtimes_file = 'step_D_runtime.csv'
    gpu_transfer = 'step_D_transfer_to_gpu.csv'
    cpu_transfer = 'step_D_transfer_to_cpu.csv'

    with open(runtimes_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(run_times)

    with open(gpu_transfer, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(load_input_times)

    with open(cpu_transfer, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(output_to_host_times)

    print(f'query embedding shape is: {query_embeddings.shape}')
