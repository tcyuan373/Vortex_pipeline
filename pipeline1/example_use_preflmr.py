"""
This script is an example of how to use the pretrained FLMR model for retrieval.
Author: Weizhe Lin, Jingbiao Mei
Date: 31/01/2024
For more information, please refer to the official repository of FLMR:
https://github.com/LinWeizheDragon/Retrieval-Augmented-Visual-Question-Answering
"""

"""
Date: 04/06/2024
Comparing to V1, this version differs in the following aspects:
1. Take the instruction from the HF dataset
2. Save retrieval results to local files
3. Report both Recall@K and PseudoRecall@K
"""


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

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
                              np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def index_corpus(checkpoint_path, 
                 index_root_path, 
                 experiment_name,
                 index_name,
                 nbits,
                 use_gpu,
                 indexing_batch_size,
                 num_gpus,
                 custom_collection):
    # Launch indexer
    index_path = index_custom_collection(
        custom_collection=custom_collection,
        model=checkpoint_path,
        index_root_path=index_root_path,
        index_experiment_name=experiment_name,
        index_name=index_name,
        nbits=nbits, # number of bits in compression
        doc_maxlen=512, # maximum allowed document length
        overwrite=False, # whether to overwrite existing indices
        use_gpu=use_gpu, # whether to enable GPU indexing
        indexing_batch_size=indexing_batch_size,
        model_temp_folder="tmp",
        nranks=num_gpus, # number of GPUs used in indexing
    )
    return index_path


def query_index(index_root_path, 
                experiment_name, 
                index_name, 
                use_gpu, 
                nbits, 
                ds, 
                passage_contents, 
                passage_ids, 
                centroid_search_batch_size, 
                query_batch_size, 
                Ks, 
                flmr_model: FLMRModelForRetrieval):
    # Search documents
    # initiate a searcher
    searcher = create_searcher(
        index_root_path=index_root_path,
        index_experiment_name=experiment_name,
        index_name=index_name,
        nbits=nbits, # number of bits in compression
        use_gpu=use_gpu, # whether to enable GPU searching
    )

    def encode_and_search_batch(batch, Ks):
        # encode queries
        input_ids = torch.LongTensor(batch["input_ids"]).to("cuda")
        # print(query_tokenizer.batch_decode(input_ids, skip_special_tokens=False))
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
            num_document_to_retrieve=max(Ks), # how many documents to retrieve for each query
            remove_zero_tensors=True,  # For PreFLMR, this is needed
            centroid_search_batch_size=centroid_search_batch_size,
        )

        ranking_dict = ranking.todict()

        # Process ranking data and obtain recall scores
        # Psuedo Recall@K to be computed by matching the answer in the retrieved documents
        # Positive ids Recall@K to be computed by matching the sample positive id with the retrieved documents ids
        recall_dict = defaultdict(list)
        result_dict = defaultdict(list)
        for i, (question_id, pos_ids) in enumerate(zip(batch["question_id"], batch["pos_item_ids"])):
            retrieved_docs = ranking_dict[question_id]
            retrieved_doc_scores = [doc[2] for doc in retrieved_docs]
            retrieved_docs = [doc[0] for doc in retrieved_docs]
            retrieved_doc_texts = [passage_contents[doc_idx] for doc_idx in retrieved_docs]
            retrieved_doc_ids = [passage_ids[doc_idx] for doc_idx in retrieved_docs]
            retrieved_doc_list = [
                {
                    "passage_id": doc_id,
                    "score": score,
                } for doc_id, score in zip(retrieved_doc_ids, retrieved_doc_scores)
            ]
            result_dict["retrieved_passage"].append(retrieved_doc_list)
            
            if True:
                # Psuedo Recall@K
                hit_list = []
                # Get answers
                answers = batch["answers"][i]
                for retrieved_doc_text in retrieved_doc_texts:
                    found = False
                    for answer in answers:
                        if answer.strip().lower() in retrieved_doc_text.lower():
                            found = True
                    if found:
                        hit_list.append(1)
                    else:
                        hit_list.append(0)

                # print(hit_list)
                # input()
                for K in Ks:
                    recall = float(np.max(np.array(hit_list[:K])))
                    recall_dict[f"Pseudo Recall@{K}"].append(recall)
            
            # Positive ids Recall@K    
            retrieved_doc_ids = [passage_ids[doc_idx] for doc_idx in retrieved_docs] 
            hit_list = []
            for retrieved_doc_id in retrieved_doc_ids:
                found = False
                for pos_id in pos_ids:
                    if pos_id == retrieved_doc_id:
                        found = True
                if found:
                    hit_list.append(1)
                else:
                    hit_list.append(0)
            for K in Ks:
                recall = float(np.max(np.array(hit_list[:K])))
                recall_dict[f"Recall@{K}"].append(recall)
        batch.update(recall_dict)
        batch.update(result_dict)
        return batch

    flmr_model = flmr_model.to("cuda")
    print("Starting encoding...")
    Ks = Ks
    # ds = ds.select(range(2000, 2100))
    ds = ds.map(
        encode_and_search_batch,
        fn_kwargs={"Ks": Ks},
        batched=True,
        batch_size=query_batch_size,
        load_from_cache_file=False,
        new_fingerprint="avoid_cache",
    )

    return ds


def main():
    from datasets import load_dataset, load_from_disk
    from datasets import DatasetDict
    
    # all args now in here
    ds_dir              = "/home/ty373/workspace/EVQA_data"  
    passage_dir         = "/home/ty373/workspace/EVQA_passages"
    image_root_dir      = '/share/desa/nfs02/ty373/HF_CACHE/datasets/'
    index_root_path     = '.'
    index_name          = 'EVQA_PreFLMR_ViT-L'
    
    checkpoint_path     = 'LinWeizheDragon/PreFLMR_ViT-L'
    experiment_name     = 'EVQA_test_split'
    indexing_batch_size = 64
    image_processor_name= 'openai/clip-vit-large-patch14'
    Ks                  = [1, 3, 5]
    use_gpu             = True
    nbits               = 8
    query_batch_size    = 8
    
    ds = load_dataset('parquet', data_files ={  'train' :ds_dir + '/train-00000-of-00001.parquet',
                                                'test'  : ds_dir + '/test-00000-of-00001-2.parquet'})
    passage_ds = load_dataset('parquet', data_files = { 'train_passages':passage_dir + '/train_passages-00000-of-00001.parquet',
                                                        'test_passages' : passage_dir + '/test_passages-00000-of-00001.parquet'})
    
    print("========= Loading dataset =========")
    print(ds)
    print(passage_ds)

    def add_path_prefix_in_img_path(example, prefix):
        if example["img_path"] != None:
            example["img_path"] = os.path.join(prefix, example["img_path"])
        return example

    ds = ds.map(add_path_prefix_in_img_path, fn_kwargs={"prefix": image_root_dir})

    use_split = 'test'

    ds = ds[use_split].select([i for i in range(64)])
    passage_ds = passage_ds[f"{use_split}_passages"].select([i for i in range(64)])
    print("========= Data Summary =========")
    print("Number of examples:", len(ds))
    print("Number of passages:", len(passage_ds))

    print("========= Indexing =========")
    # Run indexing on passages
    passage_contents = passage_ds["passage_content"]
    passage_ids = passage_ds["passage_id"]

    # passage_contents =['<BOK> ' + passage + ' <EOK>' for passage in passage_contents]

    # if args.run_indexing:
    #     ## Call ColBERT indexing to index passages
    #     index_corpus(args, passage_contents)
    # else:
    #     print("args.run_indexing is False, skipping indexing...")

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
    ds = ds.map(prepare_inputs)

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

    print("========= Querying =========")
    ds = query_index(index_root_path=index_root_path, 
                    experiment_name=experiment_name, 
                    index_name=index_name, 
                    use_gpu=use_gpu, 
                    nbits=nbits, 
                    ds=ds, 
                    passage_contents=passage_contents, 
                    passage_ids=passage_ids, 
                    centroid_search_batch_size=None, 
                    query_batch_size=query_batch_size, 
                    Ks=Ks, 
                    flmr_model=flmr_model)
    # Compute final recall
    print("=============================")
    print("Inference summary:")
    print("=============================")
    # print(args.dataset, checkpoint_path)
    print(f"Total number of questions: {len(ds)}")

    if True:
        for K in Ks:
            recall = np.mean(np.array(ds[f"Pseudo Recall@{K}"]))
            print(f"Pseudo Recall@{K}:\t", recall)
    for K in Ks:
        recall = np.mean(np.array(ds[f"Recall@{K}"]))
        print(f"Recall@{K}:\t", recall)
    print("=============================")
    report_path = os.path.join('.', f"nbits_{nbits}_{index_name}.json")
    # print(f"Saving reports to {report_path}...")
    
    all_columns = ds.column_names
    all_columns = [column for column in all_columns if ('Recall@' in column or column in ["question_id", "retrieved_passage"]) ]
    ds_to_record = ds.select_columns(all_columns)
    dict_to_record = ds_to_record.to_pandas().set_index("question_id").to_dict(orient="index")
    with open(report_path, 'w') as f:
        json.dump(
            dict_to_record, f, indent=4, cls=NumpyEncoder
        )
    print("Done! Program exiting...")


if __name__ == "__main__":
    # Initialize arg parser
    main()

