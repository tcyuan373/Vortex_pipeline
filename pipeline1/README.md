# This repo contains the scripts for running invididual steps
# For Pipeline 1
We provide a script for running pipeline1 the preFLMR workflow End-2-End, excluding the step of building indexing.  For the arguments,
* >***index_root_path***:         
the location to find pre-built index
* >***index_experiment_name***:  
experiment name to find the pre-built index
* >***index_name***:             
also part of the pre-built index root, and altogether the index is located in ***root/exp_name/indexes/index_name***
* >***img_root***:               
image root to search for query images, must be the same number of elements as ***raw_sentences***


# Original PreFLMR implementation in one file
```
python example_use_preflmr.py \
        --use_gpu \
        --index_root_path "." \
        --index_name EVQA_PreFLMR_ViT-L \
        --experiment_name EVQA \
        --indexing_batch_size 64 \
        --image_root_dir ./EVQA/eval_image/ \
        --local_data_hf ./BByrneLab/EVQA_PreFLMR_preprocessed_data \
        --local_passages_hf ./BByrneLab/multi_task_multi_modal_knowledge_retrieval_benchmark_M2KR \
        --dataset EVQA \
        --use_split test \
        --nbits 8 \
        --Ks 1 5 10 20 50 100 500 \
        --checkpoint_path LinWeizheDragon/PreFLMR_ViT-L \
        --image_processor_name openai/clip-vit-large-patch14 \
        --query_batch_size 8 
```


Datasets download link is provided here
* > [local_data_hf](https://huggingface.co/datasets/BByrneLab/EVQA_PreFLMR_preprocessed_data/tree/main): 
Contains info regarding the relative **image_path**, **image_id**, **question_id**, **question**, **answers**, etc.
* > [local_passages_hf](https://huggingface.co/datasets/BByrneLab/EVQA_PreFLMR_preprocessed_passages/tree/main): One supporting doc for corresponding question.
* > [image_root_dir](https://huggingface.co/datasets/BByrneLab/M2KR_Images/tree/main/EVQA) make sure to download and unzip both zip files (INAT and Google-landmarks) for EVQA tasks, the script will use the **image_path** from ***local_data*** to read the images