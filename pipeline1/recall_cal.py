import torch
import numpy as np
from datasets import load_dataset
import json
import os, sys

# use train passage ds
if __name__ == "__main__":
    use_split = "train"
    passage_ds_dir = "/mnt/nvme0/vortex_pipeline1/EVQA_passages/"
    ds_dir = "/mnt/nvme0/vortex_pipeline1/EVQA_data/"
    passage_ds = load_dataset('parquet', data_files ={  
                                                'train' : passage_ds_dir + 'train_passages-00000-of-00001.parquet',
                                                'test'  : passage_ds_dir + 'test_passages-00000-of-00001.parquet',
                                                })[use_split]

    ds = load_dataset('parquet', data_files ={  
                                                'train' : ds_dir + 'train-00000-of-00001.parquet',
                                                'test'  : ds_dir + 'test-00000-of-00001-2.parquet',
                                                })[use_split].select(i for i in range(0, 167000, 1)) 

    passage_contents = passage_ds["passage_content"]
    passage_ids = passage_ds["passage_id"]
    # load local result dic
    with open("search_result_dict.json", "r") as f:
        ranking_dict = json.load(f)

    qid_list = list(ranking_dict)
    ds_answers = ds["answers"]
    pos_id_list = ds["pos_item_ids"]

    print(ds_answers[1])
    # print(qid_list)

    for i, (question_id, pos_ids) in enumerate(zip(qid_list, pos_id_list)):
        retrieved_docs = ranking_dict[question_id]
        retrieved_doc_scores = [doc[2] for doc in retrieved_docs]
        retrieved_docs = [doc[0] for doc in retrieved_docs]
        retrieved_doc_ids = [passage_ids[doc_idx] for doc_idx in retrieved_docs]
        
        retrieved_doc_list = [
            {
                "passage_id": doc_id,
                "score": score,
            } for doc_id, score in zip(retrieved_doc_ids, retrieved_doc_scores)
        ]
        hit_list = []
        # Get answers
        
        answers = ds_answers[i]
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