import threading
import time
import torch
import torch.nn as nn
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
from pipeline1.configuration_flmr import FLMRConfig, FLMRTextConfig, FLMRVisionConfig
from pipeline1.modeling_pretrained_model import FLMRPreTrainedModel


FLMR_TEXT_ENCODERS_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. To match pretraining, FLMR input sequence should be
            formatted with [CLS] and [SEP] tokens as follows:

            (a) For sequence pairs (for a pair title+text for example):

            ```
            tokens:         [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
            token_type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
            ```

            (b) For single sequences (for a question for example):

            ```
            tokens:         [CLS] the dog is hairy . [SEP]
            token_type_ids:   0   0   0   0  0     0   0
            ```

            FLMR is a model with absolute position embeddings so it's usually advised to pad the inputs on the right
            rather than the left.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        token_type_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,
            1]`:

            - 0 corresponds to a *sentence A* token,
            - 1 corresponds to a *sentence B* token.

            [What are token type IDs?](../glossary#token-type-ids)
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


class FLMRTextModel(FLMRPreTrainedModel):
    base_model_prefix = "flmr_text_model"
    config_class = FLMRTextConfig

    def __init__(self, config: FLMRTextConfig, *args, **kwargs):
        super().__init__(config)
        if config.text_encoder_base_model == "bert-base-uncased":
            self.bert_model = BertModel(config, add_pooling_layer=True)
        else:
            self.bert_model = AutoModel.from_pretrained(config.text_encoder_base_model, *args, **kwargs)
        if self.bert_model.config.hidden_size <= 0:
            raise ValueError("Encoder hidden_size can't be zero")
        self.projection_dim = config.projection_dim
        if self.projection_dim > 0:
            self.encode_proj = nn.Linear(self.bert_model.config.hidden_size, config.projection_dim)
        # Initialize weights and apply final processing
        self.post_init()
        self.text_model = self.bert_model

    def cuda(self):
        self.text_model.to('cuda')
        if self.projection_dim > 0:
            self.encode_proj.to('cuda')
    
    @add_start_docstrings_to_model_forward(FLMR_TEXT_ENCODERS_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=FLMRTextConfig)
    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        output_attentions: bool = None,
        output_hidden_states: bool = None,
        return_dict: bool = None,
    ) -> Union[BaseModelOutputWithPooling, Tuple[Tensor, ...]]:
        r"""
        Returns:

        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        pooled_output = sequence_output[:, 0, :]

        if self.projection_dim > 0:
            pooled_output = self.encode_proj(pooled_output)

        if not return_dict:
            return (sequence_output, pooled_output) + outputs[2:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @property
    def embeddings_size(self) -> int:
        if self.projection_dim > 0:
            return self.encode_proj.out_features
        return self.text_model.config.hidden_size
    
    
def run_model_inference():
    model = FLMRTextModel(FLMRTextConfig())
    model.cuda()
    print("Starting model inference...")

    input_data = torch.ones(2, 512).to(torch.int).cuda()  # Random input tensor
    
    # Perform inference
    for i in range(5):
        print(f"begin model @ {time.perf_counter()}")
        start_time = time.time()
        with torch.no_grad():
            output = model(input_data)  # Forward pass
        end_time = time.time()
    
    print(f"Model inference completed in {end_time - start_time:.3f} seconds.")
    

def print_hello_world():
    start_time = time.time()
    while True:
        print(f"Hello, World! @ {time.perf_counter()}")
        time.sleep(0.001)  # 1 millisecond
        if time.time() - start_time > 240:
            break

# Main execution
if __name__ == "__main__":
    # Create threads
    inference_thread = threading.Thread(target=run_model_inference)
    hello_world_thread = threading.Thread(target=print_hello_world)
    
    # Start threads
    hello_world_thread.start()  # Start the Hello, World! thread
    inference_thread.start()    # Start the model inference thread

    # Wait for inference to complete
    inference_thread.join()
    hello_world_thread.join()
    # The hello_world_thread will continue running as it is a daemon thread.
    print("Main thread completed.")
