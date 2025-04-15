#!/usr/bin/env python3
import torch
import argparse
import os
import csv
import numpy as np
from torch import nn
from flmr import FLMRConfig, FLMRQueryEncoderTokenizer, FLMRContextEncoderTokenizer, FLMRModelForRetrieval
from transformers import BertConfig
from transformers.models.bert.modeling_bert import BertEncoder

TOTAL_RUNS = 1


class StepC:
    def __init__(self, flmr_config):
        self.flmr_config = flmr_config
        transformer_mapping_config = BertConfig.from_pretrained(
            self.flmr_config.transformer_mapping_config_base
        )
        transformer_mapping_config.num_hidden_layers = self.flmr_config.transformer_mapping_num_hidden_layers
        transformer_mapping_config.is_decoder = True
        transformer_mapping_config.add_cross_attention = True

        self.local_model_path = "models_step_C_transformer_mapping_input_linear.pt"

        if not os.path.exists(self.local_model_path):
            full_model = FLMRModelForRetrieval.from_pretrained(
                'LinWeizheDragon/PreFLMR_ViT-L',
                query_tokenizer=FLMRQueryEncoderTokenizer.from_pretrained(
                    'LinWeizheDragon/PreFLMR_ViT-L',
                    text_config=self.flmr_config.text_config,
                    subfolder="query_tokenizer"
                ),
                context_tokenizer=FLMRContextEncoderTokenizer.from_pretrained(
                    'LinWeizheDragon/PreFLMR_ViT-L',
                    text_config=self.flmr_config.text_config,
                    subfolder="context_tokenizer"
                )
            )
            self.transformer_mapping_input_linear = full_model.transformer_mapping_input_linear
            del full_model
        else:
            self.transformer_mapping_input_linear = nn.Linear(
                self.flmr_config.vision_config.hidden_size,
                transformer_mapping_config.hidden_size
            )
            self.transformer_mapping_input_linear.load_state_dict(torch.load(self.local_model_path, weights_only=True))

    def load_model_cuda(self):
        self.transformer_mapping_input_linear.cuda()

    def stepC_output(self, vision_second_last_layer_hidden_states):
        return self.transformer_mapping_input_linear(vision_second_last_layer_hidden_states)


class StepD:
    def __init__(self, flmr_config):
        self.flmr_config = flmr_config
        self.query_tokenizer = FLMRQueryEncoderTokenizer.from_pretrained(
            'LinWeizheDragon/PreFLMR_ViT-L',
            text_config=self.flmr_config.text_config,
            subfolder="query_tokenizer"
        )
        self.skiplist = []
        self.cross_attn_len = 32
        self.local_tf_mapping_path = '/mydata/EVQA/models/models_step_D_transformer_mapping.pt'
        self.local_tf_mapping_output_path = '/mydata/EVQA/models/models_step_D_transformer_mapping_output.pt'

        if not os.path.exists(self.local_tf_mapping_path) or not os.path.exists(self.local_tf_mapping_output_path):
            full_model = FLMRModelForRetrieval.from_pretrained(
                'LinWeizheDragon/PreFLMR_ViT-L',
                query_tokenizer=self.query_tokenizer,
                context_tokenizer=FLMRContextEncoderTokenizer.from_pretrained(
                    'LinWeizheDragon/PreFLMR_ViT-L',
                    text_config=self.flmr_config.text_config,
                    subfolder="context_tokenizer"
                )
            )
            self.transformer_mapping_network = full_model.transformer_mapping_network
            self.transformer_mapping_output_linear = full_model.transformer_mapping_output_linear
            del full_model
        else:
            tfm_cfg = BertConfig.from_pretrained(self.flmr_config.transformer_mapping_config_base)
            tfm_cfg.is_decoder = True
            tfm_cfg.add_cross_attention = True
            tfm_cfg.num_hidden_layers = 1

            self.transformer_mapping_network = BertEncoder(tfm_cfg)
            self.transformer_mapping_network.load_state_dict(torch.load(self.local_tf_mapping_path, weights_only=True))
            self.transformer_mapping_output_linear = nn.Linear(
                tfm_cfg.hidden_size, self.flmr_config.dim
            )
            self.transformer_mapping_output_linear.load_state_dict(torch.load(self.local_tf_mapping_output_path, weights_only=True))

        if self.flmr_config.mask_instruction_token:
            self.mask_instruction = True
            self.instruction_token_id = self.query_tokenizer.encode(
                self.flmr_config.mask_instruction_token, add_special_tokens=False
            )[0]
        else:
            self.mask_instruction = False

    def load_model_cuda(self):
        self.transformer_mapping_network.cuda()
        self.transformer_mapping_output_linear.cuda()

    def invert_attention_mask(self, encoder_attention_mask: torch.Tensor) -> torch.Tensor:
        if encoder_attention_mask.dim() == 2:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
        else:
            raise ValueError("Only 2D attention masks are supported here")
        encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * torch.finfo(encoder_attention_mask.dtype).min
        return encoder_extended_attention_mask

    def query_mask(self, input_ids):
        if not self.mask_instruction:
            return (input_ids != 0).float()

        sep_positions = torch.argmax((input_ids == self.instruction_token_id).int(), dim=1).tolist()
        sep_positions = [max(1, pos) for pos in sep_positions]
        mask = torch.zeros_like(input_ids).float()
        for i, pos in enumerate(sep_positions):
            mask[i, pos:] = 1.0
        return mask

    def cross_attn_embedding(
        self,
        input_ids,
        text_embeddings,
        text_encoder_hidden_states,
        vision_embeddings,
        transformer_mapping_input_features,
    ):
        mask = self.query_mask(input_ids).unsqueeze(-1).cuda()
        text_embeddings = text_embeddings * mask
        encoder_mask = torch.ones_like(mask)

        if text_encoder_hidden_states.shape[1] > self.cross_attn_len:
            text_encoder_hidden_states = text_encoder_hidden_states[:, :self.cross_attn_len]
            encoder_mask = encoder_mask[:, :self.cross_attn_len]

        attn_mask = self.invert_attention_mask(encoder_mask.squeeze(-1))

        outputs = self.transformer_mapping_network(
            transformer_mapping_input_features,
            encoder_hidden_states=text_encoder_hidden_states,
            encoder_attention_mask=attn_mask
        )
        x = self.transformer_mapping_output_linear(outputs.last_hidden_state)
        vision_embeddings = torch.cat([vision_embeddings, x], dim=1)
        Q = torch.cat([text_embeddings, vision_embeddings], dim=1)
        return torch.nn.functional.normalize(Q, p=2, dim=2).detach().cpu()


def run_benchmark(flmr_config, bsize, output_dir, pid):
    step_c = StepC(flmr_config)
    step_d = StepD(flmr_config)
    step_c.load_model_cuda()
    step_d.load_model_cuda()

    query_len = 32
    mapping_prefix_len = 32
    late_dim = 128
    hidden = 768
    vision_hidden = 1024

    run_times = []

    for _ in range(TOTAL_RUNS):
        ids = torch.randint(0, 10000, (bsize, query_len), dtype=torch.int64).cuda()
        txt_embed = torch.randn(bsize, query_len, late_dim).cuda()
        txt_enc_hs = torch.randn(bsize, query_len, hidden).cuda()
        vis_embed = torch.randn(bsize, mapping_prefix_len, late_dim).cuda()
        vis_hs = torch.randn(bsize, 256, vision_hidden).cuda()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        tf_input = step_c.stepC_output(vis_hs)
        _ = step_d.cross_attn_embedding(ids, txt_embed, txt_enc_hs, vis_embed, tf_input)
        
        torch.cuda.synchronize()
        end.record()

        elapsed = start.elapsed_time(end) * 1e6  # ns
        run_times.append(elapsed)

    avg_latency = sum(run_times) / len(run_times)
    throughput = (bsize * len(run_times)) / (sum(run_times) / 1e9)

    print(f"Batch size {bsize}, throughput: {throughput:.2f} queries/sec")
    print(f"Avg latency per batch (StepC+StepD): {avg_latency:.0f} ns")

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f'stepCD_bsize{bsize}_runtime{pid}_tp{throughput:.2f}.csv')
    with open(out_path, mode='w', newline='') as f:
        csv.writer(f).writerow(run_times)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark StepC + StepD")
    parser.add_argument("-p", "--output_dir", type=str, required=True)
    parser.add_argument("-id", "--pid", type=str, required=True)
    parser.add_argument("-b", "--bsize", type=int, required=True)
    args = parser.parse_args()

    flmr_cfg = FLMRConfig.from_pretrained("LinWeizheDragon/PreFLMR_ViT-L")
    run_benchmark(flmr_cfg, args.bsize, args.output_dir, args.pid)