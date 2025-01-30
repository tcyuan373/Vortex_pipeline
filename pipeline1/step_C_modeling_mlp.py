import torch
import torch.distributed as dist
from torch import Tensor, nn
from flmr import FLMRConfig
from transformers import BertConfig

class StepC:
    def __init__(self):
        self.checkpoint_path = 'LinWeizheDragon/PreFLMR_ViT-L'
        self.flmr_config = FLMRConfig.from_pretrained(self.checkpoint_path)
        
        transformer_mapping_config_base = self.flmr_config.transformer_mapping_config_base
        transformer_mapping_config = BertConfig.from_pretrained(transformer_mapping_config_base)
        transformer_mapping_config.num_hidden_layers = self.flmr_config.transformer_mapping_num_hidden_layers
        transformer_mapping_config.is_decoder = True
        transformer_mapping_config.add_cross_attention = True
        
        self.transformer_mapping_input_linear = nn.Linear(
            self.flmr_config.vision_config.hidden_size, transformer_mapping_config.hidden_size
        )
        
    def load_model_cuda(self):
        self.transformer_mapping_input_linear.cuda()
        
    def stepC_output(self, vision_second_last_layer_hidden_states):
        transformer_mapping_input_features = self.transformer_mapping_input_linear(
            vision_second_last_layer_hidden_states
        )
        
        return transformer_mapping_input_features
if __name__ == "__main__": # Bsize, vision_hidden_size[-2], vision_hidden_size[-1]
    stepc = StepC()
    stepc.load_model_cuda()
    bsize = 16
    dummy_hidden_states = torch.randn(bsize, 256, 1024).cuda()
    output = stepc.stepC_output(dummy_hidden_states)
    output.cpu()
    print(f"transformer mapping input feature shape is: {output.shape}")