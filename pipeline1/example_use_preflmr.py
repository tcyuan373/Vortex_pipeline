import os
import json
from collections import defaultdict

import numpy as np
import torch
from colbert import Indexer, Searcher
from colbert.data import Queries
from colbert.infra import ColBERTConfig, Run, RunConfig
from easydict import EasyDict
from PIL import Image

from transformers import (
    AutoImageProcessor,
    AutoModel,
    AutoTokenizer,
)
from flmr import (
    FLMRModelForRetrieval,
    FLMRQueryEncoderTokenizer,
    FLMRContextEncoderTokenizer,
    FLMRConfig,
)
from flmr import index_custom_collection
from flmr import create_searcher, search_custom_collection
import time
from torch.utils.data import DataLoader




def main():
    from datasets import load_dataset, load_from_disk
    from datasets import DatasetDict
    
    # all args now in here
    ds_dir              = "/mydata/EVQA/EVQA_data"  
    image_root_dir      = '/mydata/EVQA'
    index_root_path     = '/mydata/EVQA/index'
    index_name          = 'EVQA_PreFLMR_ViT-L'
    checkpoint_path     = 'LinWeizheDragon/PreFLMR_ViT-L'
    experiment_name     = 'EVQA_train_split'
    image_processor_name= 'openai/clip-vit-large-patch14'
    use_gpu             = True
    nbits               = 8
    query_batch_size    = 8
    
    ds = load_dataset('parquet', data_files ={  'train' :ds_dir + '/train-00000-of-00001.parquet',
                                                'test'  : ds_dir + '/test-00000-of-00001-2.parquet'})
    print("========= Loading dataset =========")
    print(ds)

    def add_path_prefix_in_img_path(example, prefix):
        if example["img_path"] != None:
            example["img_path"] = os.path.join(prefix, example["img_path"])
        return example

    ds = ds.map(add_path_prefix_in_img_path, fn_kwargs={"prefix": image_root_dir})

    use_split = 'train'

    ds = ds[use_split].select([i for i in range(8)])
    print("========= Data Summary =========")
    print("Number of examples:", len(ds))


    print("========= Loading pretrained model =========")
    flmr_config = FLMRConfig.from_pretrained(checkpoint_path)
    query_tokenizer = FLMRQueryEncoderTokenizer.from_pretrained(checkpoint_path,
                                                                    text_config=flmr_config.text_config,
                                                                    subfolder="query_tokenizer")
    context_tokenizer = FLMRContextEncoderTokenizer.from_pretrained(checkpoint_path,
                                                                    text_config=flmr_config.text_config,
                                                                    subfolder="context_tokenizer")

    flmr_model = FLMRModelForRetrieval.from_pretrained(
        checkpoint_path,
        query_tokenizer=query_tokenizer,
        context_tokenizer=context_tokenizer,
    )
    flmr_model = flmr_model.to("cuda")
    image_processor = AutoImageProcessor.from_pretrained(image_processor_name)

    print("========= Preparing query input =========")
    
    def prepare_inputs(sample):
        sample = EasyDict(sample)

        module = EasyDict(
            {"type": "QuestionInput", "option": "default", "separation_tokens": {"start": "", "end": ""}}
        )

        instruction = sample.instruction.strip()
        if instruction[-1] != ":":
            instruction = instruction + ":"
        instruction = instruction.replace(":", flmr_config.mask_instruction_token)
        #random_instruction = random.choice(instructions)
        text_sequence = " ".join(
            [instruction]
            + [module.separation_tokens.start]
            + [sample.question]
            + [module.separation_tokens.end]
        )

        sample["text_sequence"] = text_sequence

        return sample
    

    # Prepare inputs using the same configuration as in the original FLMR paper

    def tokenize_inputs(examples, query_tokenizer, image_processor):
        encoding = query_tokenizer(examples["text_sequence"])
        examples["input_ids"] = encoding["input_ids"]
        examples["attention_mask"] = encoding["attention_mask"]

        pixel_values = []
        for img_path in examples["img_path"]:

            if img_path is None:
                image = Image.new("RGB", (336, 336), color='black')
            else:
                image = Image.open(img_path).convert("RGB")
            
            encoded = image_processor(image, return_tensors="pt")
            pixel_values.append(encoded.pixel_values)

        pixel_values = torch.stack(pixel_values, dim=0)
        examples["pixel_values"] = pixel_values
        return examples

    # Tokenize and prepare image pixels for input
    ds = ds.map(
        tokenize_inputs,
        fn_kwargs={"query_tokenizer": query_tokenizer, "image_processor": image_processor},
        batched=True,
        batch_size=8,
        num_proc=16,
    )

    BS = 1
    num_batches = 1000
    
    # using torch loader
    ds = ds[use_split].select([i for i in range(0, BS*num_batches, 1)])
    ds.set_format(
        type="torch", 
        columns=["input_ids", "attention_mask", "pixel_values", "text_sequence", "question_id", "question"]
    )
    loader = DataLoader(
        ds, 
        batch_size=BS, 
        shuffle=False, 
        num_workers=16,      # Use multiple workers to prefetch batches in parallel
        prefetch_factor=2,   # How many batches each worker preloads (can adjust based on your system)
        pin_memory=True      # Optionally, if you are transferring to GPU later
    )
    
    searcher = create_searcher(
        index_root_path=index_root_path,
        index_experiment_name=experiment_name,
        index_name=index_name,
        nbits=nbits, # number of bits in compression
        use_gpu=use_gpu, # whether to enable GPU searching
    )
    
    start = time.perf_counter()
    
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= num_batches:
            break
        # prepare input for FLMR
        input_ids = torch.LongTensor(batch["input_ids"]).to("cuda")
        attention_mask = torch.LongTensor(batch["attention_mask"]).to("cuda")
        pixel_values = torch.FloatTensor(batch["pixel_values"]).to("cuda")
        query_input = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
        }
        query_embeddings = flmr_model.query(**query_input).late_interaction_output
        query_embeddings = query_embeddings.detach().cpu()

        # search
        custom_quries = {
            question_id: question for question_id, question in zip(batch["question_id"], batch["question"])
        }
        ranking = search_custom_collection(
            searcher=searcher,
            queries=custom_quries,
            query_embeddings=query_embeddings,
            num_document_to_retrieve=5, # how many documents to retrieve for each query
            remove_zero_tensors=True,  # For PreFLMR, this is needed
            centroid_search_batch_size=None,
        )

        ranking_dict = ranking.todict()

        
        



if __name__ == "__main__":

    main()

