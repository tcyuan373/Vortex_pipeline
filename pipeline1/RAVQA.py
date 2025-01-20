from transformers import AutoConfig, AutoModel, AutoImageProcessor, AutoTokenizer
import torch

checkpoint_path = "LinWeizheDragon/PreFLMR_ViT-L"
image_processor_name = "openai/clip-vit-large-patch14"
query_tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, subfolder="query_tokenizer", trust_remote_code=True)
context_tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, subfolder="context_tokenizer", trust_remote_code=True)

model = AutoModel.from_pretrained(checkpoint_path,
                                query_tokenizer=query_tokenizer,
                                context_tokenizer=context_tokenizer,
                                trust_remote_code=True,
                                )
image_processor = AutoImageProcessor.from_pretrained(image_processor_name)
