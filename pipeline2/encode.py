#!/usr/bin/env python3
from collections import defaultdict
import io
import numpy as np
import json
import re
import time
import warnings
warnings.filterwarnings("ignore")

from FlagEmbedding import BGEM3FlagModel




class EncoderUDL():
     def __init__(self):
          '''
          Constructor
          '''
          self.encoder = BGEM3FlagModel("BAAI/bge-m3", use_fp16=False, device="cpu")
          self.centroids_embeddings = np.array([])
          self.emb_dim = 1024
     

     def encode(self,query_list):
          encode_result = self.encoder.encode(
               query_list, return_dense=True, return_sparse=False, return_colbert_vecs=False
          )
          query_embeddings = encode_result['dense_vecs']
          query_embeddings = query_embeddings[:, :self.emb_dim]  
          return query_embeddings
        
     def __del__(self):
          pass


if __name__ == "__main__":
     udl = EncoderUDL()
     query_list = ["What is the capital of France?", "What is the capital of USA?"]
     query_embeddings = udl.encode(query_list)
     print(query_embeddings.shape)
     print(query_embeddings)
     del udl