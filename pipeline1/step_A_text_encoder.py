# pretrained_uncased_BERT with linear projection
# add load project layer weights
import csv
import numpy
import os
import time

import torch
from torch import Tensor, nn
from flmr import FLMRConfig, FLMRQueryEncoderTokenizer, FLMRContextEncoderTokenizer, FLMRModelForRetrieval, FLMRTextModel
# input text strings
# output: text_embeddings, input_ids[gen mask], encoder_hidden_states

class StepA:
    def __init__(self,):
        self.checkpoint_path = 'LinWeizheDragon/PreFLMR_ViT-L'
        self.flmr_config = FLMRConfig.from_pretrained(self.checkpoint_path)
        self.local_encoder_path = 'models_step_A_query_text_encoder.pt'
        self.local_projection_path = 'models_step_A_query_text_linear.pt'
        self.query_tokenizer = FLMRQueryEncoderTokenizer.from_pretrained(
                self.checkpoint_path, 
                text_config=self.flmr_config.text_config, 
                subfolder="query_tokenizer")
        self.context_tokenizer = FLMRContextEncoderTokenizer.from_pretrained(
            self.checkpoint_path, 
            text_config=self.flmr_config.text_config, 
            subfolder="context_tokenizer"
        )

        if not os.path.exists(self.local_encoder_path) and not os.path.exists(self.local_projection_path):
            print('local model not found, initing from full model...')

            full_model = FLMRModelForRetrieval.from_pretrained(
                self.checkpoint_path,
                query_tokenizer=self.query_tokenizer,
                context_tokenizer=self.context_tokenizer,
            )
            self.query_text_encoder = full_model.query_text_encoder
            self.query_text_encoder_linear = full_model.query_text_encoder_linear
            # torch.save(self.query_text_encoder.state_dict(), self.local_encoder_path)
            # torch.save(self.query_text_encoder_linear.state_dict(), self.local_projection_path)
            del full_model
        else:
            print(f'found local model for step A, now loading...')

            self.query_text_encoder = FLMRTextModel(self.flmr_config.text_config)
            self.query_text_encoder.load_state_dict(torch.load(self.local_encoder_path, weights_only=True))
            self.query_text_encoder_linear = nn.Linear(self.flmr_config.text_config.hidden_size, self.flmr_config.dim, bias=False)
            self.query_text_encoder_linear.load_state_dict(torch.load(self.local_projection_path, weights_only=True))

        self.skiplist = []

        if self.flmr_config.mask_instruction_token is not None:
            self.mask_instruction = True
            # obtain the token id of the instruction token
            self.instruction_token_id = self.query_tokenizer.encode(
                self.flmr_config.mask_instruction_token, add_special_tokens=False
            )[0]
        else:
            self.mask_instruction = False


    def load_model_cuda(self):
        self.query_text_encoder_linear.to('cuda')
        self.query_text_encoder.to('cuda')


    def stepA_output(
        self,
        input_text_sequence,
        load_input_times
    ):
        # query sentences: bsize of sentences
        encoded_inputs      = self.query_tokenizer(input_text_sequence)

        # time before put to GPU
        mvgpu_start=time.perf_counter_ns()
        input_ids           = encoded_inputs['input_ids'].to(self.query_text_encoder.device).cuda()
        attention_mask      = encoded_inputs['attention_mask'].to(self.query_text_encoder.device).cuda()
        # time after put to GPU
        mvgpu_end=time.perf_counter_ns()
        load_input_times.append(mvgpu_end-mvgpu_start)

        text_encoder_outputs = self.query_text_encoder(input_ids=input_ids,attention_mask=attention_mask,)
        text_encoder_hidden_states = text_encoder_outputs[0]
        text_embeddings = self.query_text_encoder_linear(text_encoder_hidden_states)

        # note, text_embeddings not masked yet here!!!!
        return text_embeddings, input_ids, text_encoder_hidden_states
    
    
if __name__ == '__main__':
    raw_sentences = ['Hello World', 'This is a test text sequence', 'I have a cute puppy', "my puppy's name is GOJI"]
    stepA = StepA()
    stepA.load_model_cuda()

    # GPU memory usage after loading model
    print("Allocated memory after loading model:", torch.cuda.memory_allocated())
    print("Reserved memory after loading model:", torch.cuda.memory_reserved())

    load_input_times = []
    run_times = []
    output_to_host_times = []

    # total start time for throughput calculation
    start=time.perf_counter_ns()

    for i in range(1000):
        # time before running model
        model_start=time.perf_counter_ns()
        txt_embed, input_ids, txt_encoder_hs = stepA.stepA_output(raw_sentences, load_input_times)
        # time after running model
        model_end=time.perf_counter_ns()
        run_times.append(model_end-model_start)

        # time before transfer to CPU
        mvcpu_start=time.perf_counter_ns()
        txt_embed.cpu()
        input_ids.cpu()
        txt_encoder_hs.cpu()
        # time after transfer to CPU
        mvcpu_end=time.perf_counter_ns()
        output_to_host_times.append(mvcpu_end-mvcpu_start)

    # total end time for throughput calculation
    end=time.perf_counter_ns()
    time_elapsed=end-start
    throughput = (1000 * len(raw_sentences)) / (time_elapsed / 1000000000)
    print("Throughput with batch size", len(raw_sentences), "(queries/s):", throughput)

    # subtract transfer time from runtime
    run_times = numpy.subtract(run_times, load_input_times)

    runtimes_file = 'step_A_runtime.csv'
    gpu_transfer = 'step_A_transfer_to_gpu.csv'
    cpu_transfer = 'step_A_transfer_to_cpu.csv'

    with open(runtimes_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(run_times)

    with open(gpu_transfer, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(load_input_times)

    with open(cpu_transfer, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(output_to_host_times)

    print(f'text embedding shape: {txt_embed.shape} | input ids shape: {input_ids.shape} | hidden_states shape: {txt_encoder_hs.shape}')
    print(stepA.query_text_encoder_linear.weight.cpu().detach().numpy())
