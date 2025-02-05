# This repo contains the scripts for running invididual steps
# For Pipeline 1
Best download the model pt and unzip the file before running the script
```
wget https://vortexstorage7348269.blob.core.windows.net/flmrmodels/models_pipeline1.zip
``` 
then unzip the pt files and put it under **pipeline1/**


We provide a script for running pipeline1 the preFLMR workflow End-2-End, excluding the step of building indexing.  For the arguments,
* >***index_root_path***:         
the location to find pre-built index
* >***index_experiment_name***:  
experiment name to find the pre-built index
* >***index_name***:             
also part of the pre-built index root, and altogether the index is located in ***root/exp_name/indexes/index_name***
* >***img_root***:               
image root to search for query images, must be the same number of elements as ***raw_sentences***


## Original PreFLMR implementation in one file
This is the original usage script for PreFLMR model.  If using **test** split,
```
python example_use_preflmr.py 
```
All arguments are hardcoded inside our **main()** function.


## Dataset Preparaions
We can directly wget from our Azure storage.
```
wget https://vortexstorage7348269.blob.core.windows.net/flmrdata/EVQA_data.zip

wget https://vortexstorage7348269.blob.core.windows.net/flmrdata/EVQA_passages.zip
```
Be sure to unzip the files afterwards, then specify the **ds_dir**, **passage_dir** in our **main()**.


Datasets download link is provided here
* > [local_data_hf](https://huggingface.co/datasets/BByrneLab/EVQA_PreFLMR_preprocessed_data/tree/main): 
Contains info regarding the relative **image_path**, **image_id**, **question_id**, **question**, **answers**, etc.
* > [local_passages_hf](https://huggingface.co/datasets/BByrneLab/EVQA_PreFLMR_preprocessed_passages/tree/main): One supporting doc for corresponding question.
* > [image_root_dir](https://huggingface.co/datasets/BByrneLab/M2KR_Images/tree/main/EVQA): make sure to download and unzip both zip files (INAT and Google-landmarks) for EVQA tasks, the script will use the **image_path** from ***local_data*** to read the images

## Prebuilt Indices
Prebuilt indicies can be found via this [link](https://docs.google.com/document/d/1KuWGWZrxURkVxDjFRy1Qnwsy7jDQb-RhlbUzm_A-tOs/edit?tab=t.0#heading=h.9y4g2wp666ho)

Or we just wget the files via
```
wget https://vortexstorage7348269.blob.core.windows.net/flmrdata/EVQA_test_split.tar.gz
wget https://vortexstorage7348269.blob.core.windows.net/flmrdata/EVQA_train_split.tar.gz
```