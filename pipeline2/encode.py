#!/usr/bin/env python3
from collections import defaultdict
import csv
import io
import numpy as np
import json
import re
import time
import torch
import warnings
warnings.filterwarnings("ignore")

from FlagEmbedding import BGEM3FlagModel, FlagModel


class EncoderUDL():
     def __init__(self):
          '''
          Constructor
          '''
          # self.encoder = BGEM3FlagModel(
          #       model_name_or_path='BAAI/bge-small-en-v1.5',
          #       device='cuda',
          #       use_fp16=False,
          #   )
          self.encoder = FlagModel(
               'BAAI/bge-small-en-v1.5',
               'cuda:0',
          )
          self.centroids_embeddings = np.array([])
          self.emb_dim = 384
     
     def encode(self,query_list):
          query_embeddings = self.encoder.encode(query_list)
          return query_embeddings
        
     def __del__(self):
          pass


if __name__ == "__main__":
     udl = EncoderUDL()
     query_list = ["What is the capital of France?", "What is the capital of USA?"]

     run_times = []
     for i in range(1000):
          model_start_event = torch.cuda.Event(enable_timing=True)
          model_end_event = torch.cuda.Event(enable_timing=True)

          # time before running model
          model_start_event.record()
          query_embeddings = udl.encode(query_list)
          # time after running model
          model_end_event.record()
          torch.cuda.synchronize()
          run_times.append((model_start_event.elapsed_time(model_end_event)) * 1e6)

     runtimes_file = 'encode_runtime.csv'

     with open(runtimes_file, mode='w', newline='') as file:
          writer = csv.writer(file)
          writer.writerow(run_times)

     print(f"finished shape:{query_embeddings.shape}")
     del udl
