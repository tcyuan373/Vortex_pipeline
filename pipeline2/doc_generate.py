#!/usr/bin/env python3
import io
import numpy as np
import json
import pickle
import re
import time
import torch
from collections import defaultdict
from collections import OrderedDict
import transformers



class DocGenerateUDL():
     def load_llm(self,):
          model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
          self.pipeline = transformers.pipeline(
               "text-generation",
               model=model_id,
               model_kwargs={"torch_dtype": torch.float16},
               device_map="auto",
          )
          self.terminators = [
               self.pipeline.tokenizer.eos_token_id,
               self.pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
          ]
          

     def __init__(self):
          '''
          Constructor
          '''
          self.doc_file_name = './miniset/doc_list.pkl'
          self.doc_list = None
          self.pipeline = None
          self.terminators = None
          self.doc_data = None
          self.load_llm()
          

     def llm_generate(self, query_text, doc_list ):
          messages = [
               {"role": "system", "content": "Answer the user query based on this list of documents:"+" ".join(doc_list)},
               {"role": "user", "content": query_text},
          ]
          
          tmp_res = self.pipeline(
               messages,
               max_new_tokens=256,
               # eos_token_id=self.terminators,
               do_sample=True,
               temperature=0.6,
               top_p=0.9,
          )
          raw_text = tmp_res[0]["generated_text"][-1]['content']

          print(f"for query:{query_text}")
          print(f"the llm generated response: {raw_text}")
          return raw_text

     # Retrieve Documents Based on FAISS Indices
     def get_documents(self,top_k_idxs):          
          doc_list = []
          if self.doc_data is None:
               with open(self.doc_file_name, "rb") as f:
                    self.doc_data = pickle.load(f)  
          
          for idx in top_k_idxs:
               if 0 <= idx < len(self.doc_data):
                    doc_text = self.doc_data[idx]
                    doc_list.append(doc_text)
               else:
                    print(f"Warning: FAISS index {idx} is out of range in doc_list.pkl.")
          return doc_list
               
     
     
     def generate(self, query_text, doc_ids):
          doc_list = self.get_documents(doc_ids)
          
          llm_res = self.llm_generate(query_text, doc_list)

          return llm_res

          

     def __del__(self):
          pass


if __name__ == "__main__":
     udl = DocGenerateUDL()
     query_text = "What is the capital of France?"
     doc_ids = [0, 1, 2, 3, 4]
     result = udl.generate(query_text, doc_ids)
     print(f"finished generating response: {result}")
     del udl