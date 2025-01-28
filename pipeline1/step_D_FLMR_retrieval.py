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

# convert the configuration
from configuration_flmr import FLMRConfig, FLMRTextConfig
from modeling_pretrained_model import PreTrainedModel
from transformers import FLMRQueryEncoderTokenizer, FLMRContextEncoderTokenizer, FLMRModelForRetrieval
from flmr_utils import (
    colbert_score,
    colbert_score_reduce,
    get_rank,
    get_world_size,
)



## key idea: replacing the retrival model input with query_text_embed, query_img_embed, context_text_embed, context_img_embed
## through away configs thats redundant


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "FLMRConfig"
_CHECKPOINT_FOR_DOC = "LinWeizheDragon/PreFLMR_ViT-L"


FLMR_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "LinWeizheDragon/PreFLMR_ViT-L",
    "LinWeizheDragon/FLMR",
    # See all FLMR models at https://huggingface.co/models?filter=flmr
]

FLMR_START_DOCSTRING = r"""

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`FLMRConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
        query_tokenizer ([`FLMRQueryEncoderTokenizer`], *optional*): The tokenizer used for tokenizing the query.
            The query tokenizer can be initialized with `FLMRQueryEncoderTokenizer.from_pretrained(pretrained_model_name_or_path)`.
        context_tokenizer ([`FLMRContextEncoderTokenizer`], *optional*): The tokenizer used for tokenizing the context.
            The context tokenizer can be initialized with `FLMRContextEncoderTokenizer.from_pretrained(pretrained_model_name_or_path)`.
"""


FLMR_MODEL_INPUTS_DOCSTRING = r"""
    Args:
        query_input_ids (`torch.LongTensor` of shape `(batch_size, query_length)`):
            Indices of input query tokens in the vocabulary. To match pretraining, FLMR input sequence should be
            formatted with [CLS] and Q marker tokens as follows:
            [CLS] [unused0] using the provided image, obtain documents that address the subsequent question : what is the capital of france? [SEP] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] ...

            FLMR is a model with absolute position embeddings so it's usually advised to pad the inputs on the right
            rather than the left.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        query_attention_mask (`torch.FloatTensor` of shape `(batch_size, query_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        query_pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`, *optional*):
            Pixel values. Pixel values can be obtained using
            [`AutoImageProcessor`]. See [`CLIPImageProcessor.__call__`] for details.
        query_image_features (`torch.FloatTensor` of shape `(batch_size, vision_encoder_hidden_size)`, *optional*):
            Image features are required when `query_pixel_values` is not provided. In this case, vision encoder outputs are pre-extracted to speed up training and inference by skipping the vision encoder forward pass and the extract image features are directly given to the FLMR model. Image features can be obtained
            using [`CLIPVisionModel`]. See [`CLIPVisionModel.__call__`] for details.
        context_input_ids (`torch.LongTensor` of shape `(batch_size * (1 + num_negative_examples), context_length)`):
            Indices of input context tokens in the vocabulary. To match pretraining, FLMR input sequence should be
            formatted with [CLS] and D marker tokens as follows:
            [CLS] [unused1] paris is the capital of france. [SEP] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] ...

            FLMR is a model with absolute position embeddings so it's usually advised to pad the inputs on the right
            rather than the left.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)

            The input batch size of this tensor is `batch_size * (1 + num_negative_examples)`. Check the following argument `num_negative_examples` for details.

        context_attention_mask (`torch.FloatTensor` of shape `(batch_size * (1 + num_negative_examples), context_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            The input batch size of this tensor is `batch_size * (1 + num_negative_examples)`. Check the following argument `num_negative_examples` for details.
        context_pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`, *optional*):
            Pixel values. Pixel values can be obtained using
            [`AutoImageProcessor`]. See [`CLIPImageProcessor.__call__`] for details.
        context_image_features (`torch.FloatTensor` of shape `(batch_size, vision_encoder_hidden_size)`, *optional*):
            Image features are required when `context_pixel_values` is not provided. In this case, vision encoder outputs are pre-extracted to speed up training and inference by skipping the vision encoder forward pass and the extract image features are directly given to the FLMR model. Image features can be obtained
            using [`CLIPVisionModel`]. See [`CLIPVisionModel.__call__`] for details.
        use_in_batch_negatives (`bool`, *optional*):
            Whether or not to use in-batch negatives. If `True`, the contrastive loss includes in-batch negatives. For each positive example, all other examples in the batch except itself are considered negative examples in computing the contrastive loss. This improves ultimate performance in practice. This input is to be used in model training.
        in_batch_negatives_from_all_gpus (`bool`, *optional*):
            Whether or not to use in-batch negatives from all GPUs. If `True`, the contrastive loss includes in-batch negatives from all GPUs. This input is to be used in model training.
        num_negative_examples (`int`, *optional*):
            The number of negative examples in the batch. For example, if `num_negative_examples` is 4, the batch size of `context_input_ids` and `context_attention_mask` is `batch_size * 5`.
        query_concat_output_from_vision_encoder (`bool`, *optional*):
            Whether or not to concatenate the output from the vision encoder to the final query late-interaction representations. If `True`, the output from the vision encoder is concatenated to the query representations. When using a pretrained model, this will be read from the model configuration. It should be set to `True` for FLMR and PreFLMR -style models.
        query_concat_output_from_text_encoder (`bool`, *optional*):
            Whether or not to concatenate the output from the text encoder to the final query late-interaction representations. If `True`, the output from the text encoder is concatenated to the query representations. When using a pretrained model, this will be read from the model configuration. It should be set to `True` for FLMR and PreFLMR -style models.

            This argument can be set to `False` when performing mapping network pretraining as in FLMR and PreFLMR, in which case the output from the text encoder is not concatenated to the final query representations.
        context_concat_output_from_vision_encoder (`bool`, *optional*):
            Whether or not to concatenate the output from the vision encoder to the final context late-interaction representations. If `True`, the output from the vision encoder is concatenated to the context representations. When using a pretrained model, this will be read from the model configuration. It should be set to `False` for FLMR and PreFLMR -style models since the context vision encoder is not used.

            This can be set to `True` to additionally encode the context images with the vision encoder when context images are provided.
        context_concat_output_from_text_encoder (`bool`, *optional*):
            Whether or not to concatenate the output from the text encoder to the final context late-interaction representations. If `True`, the output from the text encoder is concatenated to the context representations. When using a pretrained model, this will be read from the model configuration. It should be set to `True` for FLMR and PreFLMR -style models.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `*_attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `*_hidden_states` under returned tensors for more detail.
"""


FLMR_MODEL_QUERY_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, query_length)`):
            Indices of input query tokens in the vocabulary. To match pretraining, FLMR input sequence should be
            formatted with [CLS] and Q marker tokens as follows:
            [CLS] [unused0] using the provided image, obtain documents that address the subsequent question : what is the capital of france? [SEP] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] ...

            FLMR is a model with absolute position embeddings so it's usually advised to pad the inputs on the right
            rather than the left.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.FloatTensor` of shape `(batch_size, query_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`, *optional*):
            Pixel values. Pixel values can be obtained using
            [`AutoImageProcessor`]. See [`CLIPImageProcessor.__call__`] for details.
        image_features (`torch.FloatTensor` of shape `(batch_size, vision_encoder_hidden_size)`, *optional*):
            Image features are required when `pixel_values` is not provided. In this case, vision encoder outputs are pre-extracted to speed up training and inference by skipping the vision encoder forward pass and the extract image features are directly given to the FLMR model. Image features can be obtained
            using [`CLIPVisionModel`]. See [`CLIPVisionModel.__call__`] for details.
        concat_output_from_vision_encoder (`bool`, *optional*):
            Whether or not to concatenate the output from the vision encoder to the final query late-interaction representations. If `True`, the output from the vision encoder is concatenated to the query representations. When using a pretrained model, this will be read from the model configuration. It should be set to `True` for FLMR and PreFLMR -style models.
        concat_output_from_text_encoder (`bool`, *optional*):
            Whether or not to concatenate the output from the text encoder to the final query late-interaction representations. If `True`, the output from the text encoder is concatenated to the query representations. When using a pretrained model, this will be read from the model configuration. It should be set to `True` for FLMR and PreFLMR -style models.

            This argument can be set to `False` when performing mapping network pretraining as in FLMR and PreFLMR, in which case the output from the text encoder is not concatenated to the final query representations.
"""


FLMR_MODEL_CONTEXT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size * (1 + num_negative_examples), context_length)`):
            Indices of input context tokens in the vocabulary. To match pretraining, FLMR input sequence should be
            formatted with [CLS] and D marker tokens as follows:
            [CLS] [unused1] paris is the capital of france. [SEP] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] ...

            FLMR is a model with absolute position embeddings so it's usually advised to pad the inputs on the right
            rather than the left.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)

            The input batch size of this tensor is `batch_size * (1 + num_negative_examples)`. Check the following argument `num_negative_examples` for details.
        attention_mask (`torch.FloatTensor` of shape `(batch_size * (1 + num_negative_examples), context_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            The input batch size of this tensor is `batch_size * (1 + num_negative_examples)`. Check the following argument `num_negative_examples` for details.
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`, *optional*):
            Pixel values. Pixel values can be obtained using
            [`AutoImageProcessor`]. See [`CLIPImageProcessor.__call__`] for details.
        image_features (`torch.FloatTensor` of shape `(batch_size, vision_encoder_hidden_size)`, *optional*):
            Image features are required when `pixel_values` is not provided. In this case, vision encoder outputs are pre-extracted to speed up training and inference by skipping the vision encoder forward pass and the extract image features are directly given to the FLMR model. Image features can be obtained
            using [`CLIPVisionModel`]. See [`CLIPVisionModel
            .__call__`] for details.
        concat_output_from_vision_encoder (`bool`, *optional*):
            Whether or not to concatenate the output from the vision encoder to the final context late-interaction representations. If `True`, the output from the vision encoder is concatenated to the context representations. When using a pretrained model, this will be read from the model configuration. It should be set to `False` for FLMR and PreFLMR -style models since the context vision encoder is not used.

            This can be set to `True` to additionally encode the context images with the vision encoder when context images are provided.
        concat_output_from_text_encoder (`bool`, *optional*):
            Whether or not to concatenate the output from the text encoder to the final context late-interaction representations. If `True`, the output from the text encoder is concatenated to the context representations. When using a pretrained model, this will be read from the model configuration. It should be set to `True` for FLMR and PreFLMR -style models.
        keep_dims (`bool`, *optional*):
            Whether or not to keep the dimensions of the output. If `True`, the output is returned with the same dimensions as the input. If `False`, the output is returned with the batch size of the input and the context length. This input is to be used in model training.
        return_mask (`bool`, *optional*):
            Whether or not to return the mask of the context representation. If `True`, the mask of the context representation is returned. This input is to be used in model training.
"""


FLMR_TEXT_ENCODERS_START_DOCSTRING = r"""

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`FLMRTextConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


# Modified from transformers.models.dpr.modeling_dpr with DPR -> FLMR
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

FLMR_VISION_ENCODERS_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`FLMRVisionConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# Modified from transformers.models.clip.modeling_clip with CLIP -> FLMR
FLMR_VISION_ENCODERS_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Padding will be ignored by default should you provide it. Pixel values can be obtained using
            [`AutoImageProcessor`]. See [`CLIPImageProcessor.__call__`] for details.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


#   output classes, might not be necessary
@dataclass
class FLMRContextEncoderOutput(ModelOutput):
    """
    Class for outputs of the `doc()` function of [`FLMRModelForRetrieval`].

    Args:
        pooler_output (`torch.FloatTensor` of shape `(batch_size, embeddings_size)`):
            The FLMR encoder outputs the *pooler_output* that corresponds to the embedding of the first token of the context representation.
            This output can be used to embed questions for nearest neighbors queries with query embeddings.
        late_interaction_output (`torch.FloatTensor` of shape `(batch_size, context_embedding_length, embeddings_size)`):
            The FLMR encoder outputs the *late_interaction_output* that corresponds to the question representation. The embeddings of all tokens are included for late interaction retrieval.
            This output is to be used to embed contexts for late-interaction retrieval with query embeddings.
        context_mask (`torch.FloatTensor` of shape `(batch_size, context_embedding_length)`):
            The FLMR encoder outputs the *context_mask* that corresponds to the mask of the context representation.
        text_encoder_attentions (`Tuple[torch.FloatTensor]`, *optional*):
            Tuple of elements containing the attention weights of the text encoder's layers. Each element is a
            tensor of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
        text_encoder_hidden_states (`Tuple[torch.FloatTensor]`, *optional*):
            Tuple of elements containing the hidden states of the text encoder at each layer plus the initial embedding
            outputs. Each tensor has a shape of `(batch_size, sequence_length, hidden_size)`.
        vision_encoder_attentions (`Tuple[torch.FloatTensor]`, *optional*):
            Tuple of elements containing the attention weights of the vision encoder's layers. Each element is a
            tensor of shape `(batch_size, num_heads, vision_sequence_length, vision_sequence_length)`.
        vision_encoder_hidden_states (`Tuple[torch.FloatTensor]`, *optional*):
            Tuple of elements containing the hidden states of the vision encoder at each layer plus the initial embedding
            outputs. Each tensor has a shape of `(batch_size, vision_sequence_length, hidden_size)`.
        transformer_mapping_network_attentions (`Tuple[torch.FloatTensor]`, *optional*):
            Tuple of elements containing the attention weights of the transformer mapping network's layers. Each element
            is a tensor of shape `(batch_size, num_heads, mapping_sequence_length, mapping_sequence_length)`.
        transformer_mapping_network_hidden_states (`Tuple[torch.FloatTensor]`, *optional*):
            Tuple of elements containing the hidden states of the transformer mapping network at each layer plus the
            initial embedding outputs. Each tensor has a shape of `(batch_size, mapping_sequence_length, hidden_size)`.
    """

    pooler_output: torch.FloatTensor
    late_interaction_output: torch.FloatTensor = None
    context_mask: torch.FloatTensor = None
    text_encoder_attentions: Optional[Tuple[Tensor]] = None
    text_encoder_hidden_states: Optional[Tuple[Tensor]] = None
    vision_encoder_attentions: Optional[Tuple[Tensor]] = None
    vision_encoder_hidden_states: Optional[Tuple[Tensor]] = None
    transformer_mapping_network_attentions: Optional[Tuple[Tensor]] = None
    transformer_mapping_network_hidden_states: Optional[Tuple[Tensor]] = None


@dataclass
class FLMRQueryEncoderOutput(ModelOutput):
    """
    Class for outputs of the `query()` function of [`FLMRModelForRetrieval.query()`].

    Args:
        pooler_output (`torch.FloatTensor` of shape `(batch_size, embeddings_size)`):
            The FLMR encoder outputs the *pooler_output* that corresponds to the embedding of the first token of the query representation.
            This output can be used to embed questions for nearest neighbors queries with context embeddings.
        late_interaction_output (`torch.FloatTensor` of shape `(batch_size, query_embedding_length, embeddings_size)`):
            The FLMR encoder outputs the *late_interaction_output* that corresponds to the question representation. The embeddings of all tokens are included for late interaction retrieval.
            This output is to be used to embed questions for late-interaction retrieval with context embeddings.
        text_encoder_attentions (`Tuple[torch.FloatTensor]`, *optional*):
            Tuple of elements containing the attention weights of the text encoder's layers. Each element is a
            tensor of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
        text_encoder_hidden_states (`Tuple[torch.FloatTensor]`, *optional*):
            Tuple of elements containing the hidden states of the text encoder at each layer plus the initial embedding
            outputs. Each tensor has a shape of `(batch_size, sequence_length, hidden_size)`.
        vision_encoder_attentions (`Tuple[torch.FloatTensor]`, *optional*):
            Tuple of elements containing the attention weights of the vision encoder's layers. Each element is a
            tensor of shape `(batch_size, num_heads, vision_sequence_length, vision_sequence_length)`.
        vision_encoder_hidden_states (`Tuple[torch.FloatTensor]`, *optional*):
            Tuple of elements containing the hidden states of the vision encoder at each layer plus the initial embedding
            outputs. Each tensor has a shape of `(batch_size, vision_sequence_length, hidden_size)`.
        transformer_mapping_network_attentions (`Tuple[torch.FloatTensor]`, *optional*):
            Tuple of elements containing the attention weights of the transformer mapping network's layers. Each element
            is a tensor of shape `(batch_size, num_heads, mapping_sequence_length, mapping_sequence_length)`.
        transformer_mapping_network_hidden_states (`Tuple[torch.FloatTensor]`, *optional*):
            Tuple of elements containing the hidden states of the transformer mapping network at each layer plus the
            initial embedding outputs. Each tensor has a shape of `(batch_size, mapping_sequence_length, hidden_size)`.
    """

    pooler_output: torch.FloatTensor
    late_interaction_output: torch.FloatTensor = None
    text_encoder_attentions: Optional[Tuple[Tensor]] = None
    text_encoder_hidden_states: Optional[Tuple[Tensor]] = None
    vision_encoder_attentions: Optional[Tuple[Tensor]] = None
    vision_encoder_hidden_states: Optional[Tuple[Tensor]] = None
    transformer_mapping_network_attentions: Optional[Tuple[Tensor]] = None
    transformer_mapping_network_hidden_states: Optional[Tuple[Tensor]] = None

    
    
# modeling beyond this point

class FLMRPreTrainedModel(PreTrainedModel):
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
            
            
            
class FLMRPretrainedModelForRetrieval(FLMRPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = FLMRConfig
    load_tf_weights = None
    base_model_prefix = "flmr"
    
    
    
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
        
        
    
class FLMRModelForRetrieval(FLMRPretrainedModelForRetrieval):
    _keys_to_ignore_on_load_unexpected = [r"cls"]
    main_input_name = "query_input_ids"
    _tied_weights_keys = []  # Added dynamically at initialization depending on the architecture

    def __init__(self, config: FLMRConfig, query_tokenizer=None, context_tokenizer=None):
        super().__init__(config)
        self.config = config
        self.vision_model_version = config.vision_model_version
        self.context_text_encoder_linear = nn.Linear(config.text_config.hidden_size, config.dim, bias=False)

        self.query_tokenizer = FLMRQueryEncoderTokenizer.from_pretrained("bert-base-uncased")
        self.context_tokenizer = context_tokenizer

        if self.query_tokenizer is None:
            logger.warning(
                "query_tokenizer is not provided. A tokenizer is initialized from `bert-base-uncased`. Please pass in an FLMRQueryEncoderTokenizer instance if you need to extend the vocabulary beyond the existing ones in the bert tokenizer."
            )

            # initialize a FLMRQueryEncoderTokenizer
            self.query_tokenizer = FLMRQueryEncoderTokenizer.from_pretrained("bert-base-uncased")

        if self.context_tokenizer is None:
            logger.warning(
                "context_tokenizer is not provided. A tokenizer is initialized from `bert-base-uncased`. Please pass in an FLMRContextEncoderTokenizer instance if you need to extend the vocabulary beyond the existing ones in the bert tokenizer."
            )

            # initialize a FLMRContextEncoderTokenizer
            self.context_tokenizer = FLMRContextEncoderTokenizer.from_pretrained("bert-base-uncased")

        self.mapping_network_prefix_length = self.config.mapping_network_prefix_length
        self.vision_encoder_embedding_size = self.config.vision_config.hidden_size
        self.text_encoder_embedding_size = self.config.text_config.hidden_size
        self.late_interaction_embedding_size = self.config.dim



        if self.config.use_vision_encoder:
            # self.context_vision_encoder = FLMRVisionModel(config.vision_config)
            # replacing the line above with this
            # self.context_vision_encoder = self.context_vision_embeds
            # self.ctx_vision_embeds = context_vision_embeds

            if self.config.use_transformer_mapping_network:
                # This is a PreFLMR style model
                transformer_mapping_config_base = self.config.transformer_mapping_config_base
                try:
                    from transformers import BertConfig
                    from transformers.models.bert.modeling_bert import BertEncoder
                except Exception as e:
                    raise ImportError(f"Failed to import BertConfig and BertEncoder from transformers. {e}")

                transformer_mapping_config = BertConfig.from_pretrained(transformer_mapping_config_base)

                assert (
                    self.config.text_config.hidden_size == transformer_mapping_config.hidden_size
                ), f"hidden_size {self.config.text_config.hidden_size} != transformer_mapping_config.hidden_size {transformer_mapping_config.hidden_size}. To use cross attention, the dimensions must match."
                # shallow transformer
                transformer_mapping_config.num_hidden_layers = self.config.transformer_mapping_num_hidden_layers
                # add cross attention
                transformer_mapping_config.is_decoder = True
                transformer_mapping_config.add_cross_attention = True

                # The linear layer from vision encoder to transformer input
                self.transformer_mapping_input_linear = nn.Linear(
                    self.vision_encoder_embedding_size, transformer_mapping_config.hidden_size
                )

                # The transformer encoder
                self.transformer_mapping_network = BertEncoder(transformer_mapping_config)

                # The linear layer from transformer output to FLMR dim
                self.transformer_mapping_output_linear = nn.Linear(
                    transformer_mapping_config.hidden_size, self.late_interaction_embedding_size
                )

        if self.config.load_cpu_extension:
            try:
                FLMRModelForRetrieval.try_load_torch_extensions()
            except Exception as e:
                raise(f"Unable to load `segmented_maxsim.cpp`. hf-hub does not download this file automatically. Please download it manually from `https://huggingface.co/LinWeizheDragon/PreFLMR_ViT-L/blob/main/segmented_maxsim.cpp` and put it under the same folder as the model file.\n {e}")

        if self.config.mask_punctuation:
            self.skiplist = {
                w: True
                for symbol in string.punctuation
                for w in [symbol, self.context_tokenizer.encode(symbol, add_special_tokens=False)[0]]
            }

        if self.config.mask_instruction_token is not None:
            self.mask_instruction = True
            # obtain the token id of the instruction token
            self.instruction_token_id = self.query_tokenizer.encode(
                self.config.mask_instruction_token, add_special_tokens=False
            )[0]
        else:
            self.mask_instruction = False

        self.loss_fn = torch.nn.CrossEntropyLoss()

        # Initialize weights and apply final processing
        self.post_init()

    @property
    def use_gpu(self):
        return self.device.type == "cuda"

    @classmethod
    def from_pretrained(self, name_or_path, **kwargs):
        obj = super().from_pretrained(name_or_path, **kwargs)
        return obj

    @classmethod
    def try_load_torch_extensions(cls):
        if hasattr(cls, "loaded_extensions"):
            return

        logger.info(
            "Loading segmented_maxsim_cpp extension (set COLBERT_LOAD_TORCH_EXTENSION_VERBOSE=True for more info)..."
        )
        segmented_maxsim_cpp = load(
            name="segmented_maxsim_cpp",
            sources=[
                os.path.join(pathlib.Path(__file__).parent.resolve(), "segmented_maxsim.cpp"),
            ],
            extra_cflags=["-O3"],
            verbose=os.getenv("COLBERT_LOAD_TORCH_EXTENSION_VERBOSE", "False") == "True",
        )
        cls.segmented_maxsim = segmented_maxsim_cpp.segmented_maxsim_cpp

        cls.loaded_extensions = True

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
                logger.error(f"can not find the separator in the input_ids: {input_ids[i].tolist()}")
        mask = [
            [
                (x not in skiplist) and (x != 0) and (index > sep_positions[seq_index] or index < 2)
                for index, x in enumerate(d)
            ]
            for seq_index, d in enumerate(input_ids.cpu().tolist())
        ]
        return mask

    @add_start_docstrings_to_model_forward(FLMR_MODEL_INPUTS_DOCSTRING)
    @replace_return_docstrings(config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        query_input_ids: Optional[torch.Tensor] = None,
        query_attention_mask: Optional[torch.Tensor] = None,
        query_pixel_values: Optional[torch.Tensor] = None,
        query_image_features: Optional[torch.Tensor] = None,
        context_input_ids: Optional[torch.Tensor] = None,
        context_attention_mask: Optional[torch.Tensor] = None,
        context_pixel_values: Optional[torch.Tensor] = None,
        context_image_features: Optional[torch.Tensor] = None,
        use_in_batch_negatives: bool = True,
        in_batch_negatives_from_all_gpus: bool = False,
        num_negative_examples: int = 1,
        query_concat_output_from_vision_encoder: Optional[Union[bool, list]] = None,
        query_concat_output_from_text_encoder: Optional[Union[bool, list]] = None,
        context_concat_output_from_vision_encoder: Optional[Union[bool, list]] = None,
        context_concat_output_from_text_encoder: Optional[Union[bool, list]] = None,
        return_dict: bool = None,
        output_attentions: bool = None,
        output_hidden_states: bool = None,
    ) -> Union[Tuple[Tensor, ...]]:
        r"""
          Return:

          Examples:

          ```python
          >>> import torch
          >>> from transformers import FLMRQueryEncoderTokenizer, FLMRContextEncoderTokenizer, FLMRModelForRetrieval, AutoImageProcessor

          >>> checkpoint_path = "LinWeizheDragon/PreFLMR_ViT-L"
          >>> image_processor_name = "openai/clip-vit-large-patch14"
          >>> query_tokenizer = FLMRQueryEncoderTokenizer.from_pretrained(checkpoint_path, subfolder="query_tokenizer")
          >>> context_tokenizer = FLMRContextEncoderTokenizer.from_pretrained(checkpoint_path, subfolder="context_tokenizer")

          >>> model = FLMRModelForRetrieval.from_pretrained(checkpoint_path,
                                                          query_tokenizer=query_tokenizer,
                                                          context_tokenizer=context_tokenizer,
                                                          )
          >>> image_processor = AutoImageProcessor.from_pretrained(image_processor_name)

          >>> Q_encoding = query_tokenizer(["Using the provided image, obtain documents that address the subsequent question: What is the capital of France?", "Extract documents linked to the question provided in conjunction with the image: What is the capital of China?"])
          >>> D_encoding = context_tokenizer(["Paris is the capital of France.", "Beijing is the capital of China.",
                                      "Paris is the capital of France.", "Beijing is the capital of China."])
          >>> Q_pixel_values = torch.zeros(2, 3, 224, 224)
          >>> inputs = dict(
                  query_input_ids=Q_encoding['input_ids'],
                  query_attention_mask=Q_encoding['attention_mask'],
                  query_pixel_values=Q_pixel_values,
                  context_input_ids=D_encoding['input_ids'],
                  context_attention_mask=D_encoding['attention_mask'],
                  use_in_batch_negatives=True,
              )

          >>> model.forward(**inputs)
          FLMRModelForRetrievalOutput(loss=tensor(4.5000, device='cuda:0', dtype=torch.float16,
        grad_fn=<NllLossBackward0>), scores=tensor([[44.2188, 40.6562],
         [39.4375, 48.4062]], device='cuda:0', dtype=torch.float16,
         grad_fn=<ViewBackward0>), in_batch_negative_loss=tensor(5.1994, device='cuda:0', grad_fn=<NllLossBackward0>), query_late_interaction_output=tensor(...), context_late_interaction_output=tensor(...)
          ```
        """

        if query_concat_output_from_vision_encoder is None:
            query_concat_output_from_vision_encoder = self.config.query_concat_output_from_vision_encoder

        if query_concat_output_from_text_encoder is None:
            query_concat_output_from_text_encoder = self.config.query_concat_output_from_text_encoder

        if context_concat_output_from_vision_encoder is None:
            context_concat_output_from_vision_encoder = self.config.context_concat_output_from_vision_encoder

        if context_concat_output_from_text_encoder is None:
            context_concat_output_from_text_encoder = self.config.context_concat_output_from_text_encoder

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        query_outputs = self.query(
            input_ids=query_input_ids,
            attention_mask=query_attention_mask,
            pixel_values=query_pixel_values,
            image_features=query_image_features,
            concat_output_from_vision_encoder=query_concat_output_from_vision_encoder,
            concat_output_from_text_encoder=query_concat_output_from_text_encoder,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        Q = query_outputs.late_interaction_output

        context_outputs = self.doc(
            input_ids=context_input_ids,
            attention_mask=context_attention_mask,
            pixel_values=context_pixel_values,
            image_features=context_image_features,
            concat_output_from_vision_encoder=context_concat_output_from_vision_encoder,
            concat_output_from_text_encoder=context_concat_output_from_text_encoder,
            keep_dims=True,
            return_mask=True,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        D, D_mask = context_outputs.late_interaction_output, context_outputs.context_mask

        # Gather tensors from other GPUs
        if in_batch_negatives_from_all_gpus:
            Q, D, D_mask = self.gather_tensors_from_other_gpus(Q, D, D_mask)
        # Repeat each query encoding for every corresponding document.
        Q_duplicated = Q.repeat_interleave(num_negative_examples + 1, dim=0).contiguous()

        scores = self.score(Q_duplicated, D, D_mask)

        # Use contrastive learning
        batch_size = query_input_ids.shape[0]
        scores = scores.view(-1, num_negative_examples + 1)
        labels = torch.zeros(batch_size, dtype=torch.long, device=self.device)
    
        return (scores,
                query_outputs.late_interaction_output,
                context_outputs.late_interaction_output,
                )


    def gather_tensors_from_other_gpus(self, query_embeddings, item_embeddings, item_mask):
        # print("get rank", get_rank())
        # print("get world size", get_world_size())
        # Gather embeddings from other GPUs
        n_nodes = get_world_size()
        if n_nodes == 1:
            return query_embeddings, item_embeddings, item_mask
        # Create placeholder to hold embeddings passed from other ranks
        global_query_embeddings_placeholder = [
            torch.zeros(*query_embeddings.shape, dtype=query_embeddings.dtype).to(query_embeddings.device)
            for _ in range(n_nodes)
        ]
        global_item_embeddings_placeholder = [
            torch.zeros(*item_embeddings.shape, dtype=item_embeddings.dtype).to(item_embeddings.device)
            for _ in range(n_nodes)
        ]
        global_item_mask_placeholder = [
            torch.zeros(*item_mask.shape, dtype=item_mask.dtype).to(item_mask.device) for _ in range(n_nodes)
        ]
        dist.all_gather(global_query_embeddings_placeholder, query_embeddings.detach())
        dist.all_gather(global_item_embeddings_placeholder, item_embeddings.detach())
        dist.all_gather(global_item_mask_placeholder, item_mask.detach())

        global_query_embeddings = []
        global_item_embeddings = []
        global_item_mask = []
        # print(f"rank {get_rank()} global_query_embeddings", global_query_embeddings)
        # print(f"rank {get_rank()} global_item_embeddings", global_item_embeddings)
        # input()
        current_rank = get_rank()
        for rank_index, remote_q_embeddings in enumerate(global_query_embeddings_placeholder):
            # We append the embeddings from other GPUs if this embedding does not require gradients
            if rank_index != current_rank:
                global_query_embeddings.append(remote_q_embeddings)
            else:
                global_query_embeddings.append(query_embeddings)

        for rank_index, remote_item_embeddings in enumerate(global_item_embeddings_placeholder):
            # We append the embeddings from other GPUs if this embedding does not require gradients
            if rank_index != current_rank:
                global_item_embeddings.append(remote_item_embeddings)
            else:
                global_item_embeddings.append(item_embeddings)

        for rank_index, remote_item_mask in enumerate(global_item_mask_placeholder):
            # We append the embeddings from other GPUs if this embedding does not require gradients
            if rank_index != current_rank:
                global_item_mask.append(remote_item_mask)
            else:
                global_item_mask.append(item_mask)

        # Replace the previous variables with gathered tensors
        query_embeddings = torch.cat(global_query_embeddings)
        item_embeddings = torch.cat(global_item_embeddings)
        item_mask = torch.cat(global_item_mask)

        return query_embeddings, item_embeddings, item_mask

    @add_start_docstrings_to_model_forward(FLMR_MODEL_QUERY_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=FLMRQueryEncoderOutput, config_class=_CONFIG_FOR_DOC)
    def query(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        image_features: Optional[torch.Tensor] = None,
        concat_output_from_vision_encoder: Optional[Union[bool, list]] = None,
        concat_output_from_text_encoder: Optional[Union[bool, list]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ):
        r"""
        Returns:

        """

        if concat_output_from_vision_encoder is None:
            concat_output_from_vision_encoder = self.config.query_concat_output_from_vision_encoder

        if concat_output_from_text_encoder is None:
            concat_output_from_text_encoder = self.config.query_concat_output_from_text_encoder

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        input_modality = []
        if pixel_values is not None or image_features is not None:
            input_modality.append("image")
        if input_ids is not None and attention_mask is not None:
            input_modality.append("text")

        text_encoder_outputs = None
        vision_encoder_outputs = None
        transformer_mapping_outputs = None

        if "image" in input_modality:
            assert (
                pixel_values is not None or image_features is not None
            ), "pixel_values or image_features must be provided if image modality is used"
            assert (
                pixel_values is None or image_features is None
            ), "pixel_values and image_features cannot be provided at the same time"

        if "text" in input_modality:
            assert (
                input_ids is not None and attention_mask is not None
            ), "input_ids and attention_mask must be provided if text modality is used"
            # Forward the text encoder
            input_ids, attention_mask = input_ids.to(self.device), attention_mask.to(self.device)
            text_encoder_outputs = self.query_text_encoder(input_ids, attention_mask=attention_mask)
            text_encoder_hidden_states = text_encoder_outputs[0]
            text_embeddings = self.query_text_encoder_linear(text_encoder_hidden_states)
            mask = torch.tensor(self.query_mask(input_ids, skiplist=self.config.query_mask_input_ids_skip_list), device=self.device).unsqueeze(2).float()

            text_embeddings = text_embeddings * mask

        if "image" in input_modality:
            if pixel_values is not None:
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

            if image_features is not None:
                batch_size = image_features.shape[0]
                vision_embeddings = image_features.to(self.device)

            # Forward the vision projection / mapping network
            vision_embeddings = self.query_vision_projection(vision_embeddings)
            vision_embeddings = vision_embeddings.view(batch_size, -1, self.late_interaction_embedding_size)

            if self.config.use_transformer_mapping_network:
                # select the second last layer
                vision_second_last_layer_hidden_states = vision_encoder_outputs.hidden_states[-2][:, 1:]
                # transformer_mapping
                transformer_mapping_input_features = self.transformer_mapping_input_linear(
                    vision_second_last_layer_hidden_states
                )

                # Cross attention only attends to the first 32 tokens
                encoder_mask = torch.ones_like(mask).to(mask.device, dtype=mask.dtype)
                if len(self.config.query_mask_input_ids_skip_list) > 0:
                    encoder_mask[torch.isin(input_ids, torch.tensor(self.config.query_mask_input_ids_skip_list))] = 0
                cross_attention_length = self.config.transformer_mapping_cross_attention_length
                if text_encoder_hidden_states.shape[1] > cross_attention_length:
                    text_encoder_hidden_states = text_encoder_hidden_states[:, :cross_attention_length]
                    encoder_mask = encoder_mask[:, :cross_attention_length]

                # Obtain cross attention mask
                encoder_extended_attention_mask = self.invert_attention_mask(encoder_mask.squeeze(-1))
                # Pass through the transformer mapping
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

        if concat_output_from_vision_encoder and concat_output_from_text_encoder:
            Q = torch.cat([text_embeddings, vision_embeddings], dim=1)
            if isinstance(concat_output_from_vision_encoder, list) or isinstance(concat_output_from_text_encoder, list):
                # When lists are passed in, mask the output accordingly
                assert isinstance(concat_output_from_vision_encoder, list) and isinstance(concat_output_from_text_encoder, list), "concat_output_from_vision_encoder and concat_output_from_text_encoder must be of the same type."
                # obtain the size of each output
                text_size = text_embeddings.shape[1]
                vision_size = vision_embeddings.shape[1]

                # Prepare the mask
                concat_output_mask = torch.zeros_like(Q).to(Q.device)

                # Mask the late interaction outputs
                concat_output_mask[:, :text_size] = torch.tensor(concat_output_from_text_encoder).bool().unsqueeze(-1).unsqueeze(-1)
                concat_output_mask[:, text_size:] = torch.tensor(concat_output_from_vision_encoder).bool().unsqueeze(-1).unsqueeze(-1)

                Q = Q * concat_output_mask

        elif concat_output_from_vision_encoder:
            Q = vision_embeddings
        elif concat_output_from_text_encoder:
            Q = text_embeddings

        vision_encoder_attentions = (
            vision_encoder_outputs.attentions
            if vision_encoder_outputs is not None
            and hasattr(vision_encoder_outputs, "attentions")
            and output_attentions
            else None
        )
        vision_encoder_hidden_states = (
            vision_encoder_outputs.hidden_states
            if vision_encoder_outputs is not None
            and hasattr(vision_encoder_outputs, "hidden_states")
            and output_hidden_states
            else None
        )
        text_encoder_attentions = (
            text_encoder_outputs.attentions
            if text_encoder_outputs is not None and hasattr(text_encoder_outputs, "attentions") and output_attentions
            else None
        )
        text_encoder_hidden_states = (
            text_encoder_outputs.hidden_states
            if text_encoder_outputs is not None
            and hasattr(text_encoder_outputs, "hidden_states")
            and output_hidden_states
            else None
        )
        transformer_mapping_network_attentions = (
            transformer_mapping_outputs.attentions
            if transformer_mapping_outputs is not None
            and hasattr(transformer_mapping_outputs, "attentions")
            and output_attentions
            else None
        )
        transformer_mapping_network_hidden_states = (
            transformer_mapping_outputs.hidden_states
            if transformer_mapping_outputs is not None
            and hasattr(transformer_mapping_outputs, "hidden_states")
            and output_hidden_states
            else None
        )

        return FLMRQueryEncoderOutput(
            pooler_output=Q[:, 0, :],
            late_interaction_output=torch.nn.functional.normalize(Q, p=2, dim=2),
            vision_encoder_attentions=vision_encoder_attentions,
            vision_encoder_hidden_states=vision_encoder_hidden_states,
            text_encoder_attentions=text_encoder_attentions,
            text_encoder_hidden_states=text_encoder_hidden_states,
            transformer_mapping_network_attentions=transformer_mapping_network_attentions,
            transformer_mapping_network_hidden_states=transformer_mapping_network_hidden_states,
        )

    @add_start_docstrings_to_model_forward(FLMR_MODEL_CONTEXT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=FLMRContextEncoderOutput, config_class=_CONFIG_FOR_DOC)
    def doc(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        image_features: Optional[torch.Tensor] = None,
        concat_output_from_vision_encoder: Optional[bool] = None,
        concat_output_from_text_encoder: Optional[bool] = None,
        keep_dims: Optional[bool] = True,
        return_mask: Optional[bool] = True,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ):
        r"""
        Returns:

        """
        assert keep_dims in [True, False]

        if concat_output_from_vision_encoder is None:
            concat_output_from_vision_encoder = self.config.context_concat_output_from_vision_encoder

        if concat_output_from_text_encoder is None:
            concat_output_from_text_encoder = self.config.context_concat_output_from_text_encoder

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        input_modality = []
        if pixel_values is not None or image_features is not None:
            input_modality.append("image")
        if input_ids is not None and attention_mask is not None:
            input_modality.append("text")

        text_encoder_outputs = None
        vision_encoder_outputs = None

        if "image" in input_modality:
            assert (
                pixel_values is not None or image_features is not None
            ), "pixel_values or image_features must be provided if image modality is used"
            assert (
                pixel_values is None or image_features is None
            ), "pixel_values and image_features cannot be provided at the same time"

        if "text" in input_modality:
            assert (
                input_ids is not None and attention_mask is not None
            ), "input_ids and attention_mask must be provided if text modality is used"
            # Forward the text encoder
            input_ids, attention_mask = input_ids.to(self.device), attention_mask.to(self.device)
            text_encoder_outputs = self.context_text_encoder(input_ids, attention_mask=attention_mask)
            text_embeddings = text_encoder_outputs[0]
            text_embeddings = self.context_text_encoder_linear(text_embeddings)

            mask = torch.tensor(self.mask(input_ids, skiplist=self.skiplist), device=self.device).unsqueeze(2).float()
            text_embeddings = text_embeddings * mask

        if "image" in input_modality:
            if pixel_values is not None:
                # Forward the vision encoder
                pixel_values = pixel_values.to(self.device)
                vision_encoder_outputs = self.context_vision_encoder(pixel_values)
                vision_embeddings = vision_encoder_outputs.last_hidden_state[:, 0]

            if image_features is not None:
                vision_embeddings = image_features.to(self.device)

            batch_size = vision_embeddings.shape[0]

            # Forward the vision projection / mapping network
            vision_embeddings = self.context_vision_projection(vision_embeddings)
            vision_embeddings = vision_embeddings.view(
                -1, self.mapping_network_prefix_length, self.late_interaction_embedding_size
            )

            image_mask = torch.ones(batch_size, vision_embeddings.shape[1], 1).to(self.device)

        if concat_output_from_vision_encoder and concat_output_from_text_encoder:
            # Note: vision embeddings must be in the front since the ColBERT engine only indexes embeddings up to number of 1's in the mask
            # TODO: fix the engine to support masks with discontinuous 0 and 1.
            D = torch.cat([vision_embeddings, text_embeddings], dim=1)
            # concatenate the mask
            mask = torch.cat([image_mask, mask], dim=1)
        elif concat_output_from_vision_encoder:
            D = vision_embeddings
            mask = image_mask
        elif concat_output_from_text_encoder:
            D = text_embeddings
            mask = mask

        D = torch.nn.functional.normalize(D, p=2, dim=2)

        if self.use_gpu:
            D = D.half()

        if keep_dims is False:
            D, mask = D.cpu(), mask.bool().cpu().squeeze(-1)
            D = [d[mask[idx]] for idx, d in enumerate(D)]

        vision_encoder_attentions = (
            vision_encoder_outputs.attentions
            if vision_encoder_outputs is not None
            and hasattr(vision_encoder_outputs, "attentions")
            and output_attentions
            else None
        )
        vision_encoder_hidden_states = (
            vision_encoder_outputs.hidden_states
            if vision_encoder_outputs is not None
            and hasattr(vision_encoder_outputs, "hidden_states")
            and output_hidden_states
            else None
        )
        text_encoder_attentions = (
            text_encoder_outputs.attentions
            if text_encoder_outputs is not None and hasattr(text_encoder_outputs, "attentions") and output_attentions
            else None
        )
        text_encoder_hidden_states = (
            text_encoder_outputs.hidden_states
            if text_encoder_outputs is not None
            and hasattr(text_encoder_outputs, "hidden_states")
            and output_hidden_states
            else None
        )

        return FLMRContextEncoderOutput(
            pooler_output=D[:, 0, :],
            late_interaction_output=D,
            context_mask=mask.bool() if return_mask else None,
            vision_encoder_attentions=vision_encoder_attentions,
            vision_encoder_hidden_states=vision_encoder_hidden_states,
            text_encoder_attentions=text_encoder_attentions,
            text_encoder_hidden_states=text_encoder_hidden_states,
        )

    def score(self, Q, D_padded, D_mask):
        # assert self.colbert_config.similarity == 'cosine'
        # if self.colbert_config.similarity == 'l2':
        #     assert self.colbert_config.interaction == 'colbert'
        #     return (-1.0 * ((Q.unsqueeze(2) - D_padded.unsqueeze(1))**2).sum(-1)).max(-1).values.sum(-1)
        return colbert_score(Q, D_padded, D_mask, use_gpu=self.use_gpu)

    def mask(self, input_ids, skiplist):
        mask = [[(x not in skiplist) and (x != 0) for x in d] for d in input_ids.cpu().tolist()]
        return mask

    
    
if __name__ =="__main__":
    
    pass