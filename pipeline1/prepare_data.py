from datasets import load_dataset, load_from_disk
from datasets import DatasetDict
import os
# Login using e.g. `huggingface-cli login` to access this dataset

ds = load_dataset('arrow', data_files={'train':'/home/jovyan/workspace/BByrneLab___evqa_pre_flmr_preprocessed_passages/evqa_pre_flmr_preprocessed_passages-train_passages.arrow',
                                       'test': '/home/jovyan/workspace/BByrneLab___evqa_pre_flmr_preprocessed_passages/evqa_pre_flmr_preprocessed_passages-test_passages.arrow',
                                       'valid': '/home/jovyan/workspace/BByrneLab___evqa_pre_flmr_preprocessed_passages/evqa_pre_flmr_preprocessed_passages-valid_passages.arrow'})

# train_ds = ds['train']
# print(train_ds[0])

# DatasetDict.load_from_disk
# ds = load_dataset('BByrneLab/EVQA_PreFLMR_preprocessed_passages')