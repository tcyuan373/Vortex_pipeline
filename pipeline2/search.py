import faiss
import numpy as np
import pandas as pd
import pickle
import os
import torch
import csv

class SearchUDL():

     def __init__(self, cluster_dir = "/mydata/msmarco/msmarco_3_clusters", index_type='Flat', nprobe=1):
          self.index_type = index_type
          self.nprobe = nprobe
          self.index = None
          self.load_cluster_embeddings(cluster_dir)
          self.build_ivf_index()

     def load_cluster_embeddings(self, cluster_dir):
          self.cluster_embeddings = []
          for file in os.listdir(cluster_dir):
               if file.startswith("cluster_") and file.endswith(".pkl"):
                    file_path = os.path.join(cluster_dir, file)
                    with open(file_path, "rb") as f:
                         emb = pickle.load(f)
                         self.cluster_embeddings.append(emb)
          self.cluster_embeddings = np.vstack(self.cluster_embeddings).astype(np.float32)  

     # Note that in this implementation, use a undistributed IVF Flat search
     def build_ivf_index(self, nlist=3):
          index = faiss.read_index('/mydata/msmarco/msmarco.index')
          gpu_res = faiss.StandardGpuResources()
          self.index = faiss.index_cpu_to_gpu(gpu_res, 0, index)
          # dim = self.cluster_embeddings.shape[1]  
          # res = faiss.StandardGpuResources()  

          # self.index = faiss.GpuIndexIVFFlat(
          #      res,  
          #      dim,  
          #      nlist,  
          #      faiss.METRIC_L2, 
          #      faiss.GpuIndexIVFFlatConfig()
          # )

          # self.index.train(self.cluster_embeddings)
          # self.index.add(self.cluster_embeddings) 
     
     def search_queries(self, query_embeddings, top_k=5):
          distances, indices = self.index.search(query_embeddings, top_k)
          return distances, indices


def load_query_embeddings(file_path):
    df = pd.read_csv(file_path)
    print(df.shape)
    return df.values.astype(np.float32)  


if __name__ == "__main__":
     dataset_dir = "/mydata/msmarco/msmarco_3_clusters"
     query_emb_file = os.path.join(dataset_dir, "query_emb.csv")
     cluster_dir = dataset_dir 
     searcher = SearchUDL(cluster_dir = cluster_dir)
     query_embeddings = load_query_embeddings(query_emb_file)
     num_queries = query_embeddings.shape[0]

     run_times = []
     bsize = 1
     k = 0
     for i in range(10):
          batch = []
          for j in range(bsize):
               batch.append(query_embeddings[k % num_queries])
               k+=1

          batch_df = pd.DataFrame(batch)

          model_start_event = torch.cuda.Event(enable_timing=True)
          model_end_event = torch.cuda.Event(enable_timing=True)

          # time before running model
          model_start_event.record()

          distances, indices = searcher.search_queries(batch_df, top_k=5)
          # print(f"distances.shape: {distances.shape}")
          # print(f"indices.shape: {indices.shape}")

          # time after running model
          model_end_event.record()
          torch.cuda.synchronize()
          run_times.append((model_start_event.elapsed_time(model_end_event)) * 1e6)

     runtimes_file = 'search_runtime.csv'

     with open(runtimes_file, mode='w', newline='') as file:
          writer = csv.writer(file)
          writer.writerow(run_times)
