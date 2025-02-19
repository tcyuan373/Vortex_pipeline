import faiss
import numpy as np
import pandas as pd
import pickle
import os

class SearchUDL():

     def __init__(self, cluster_dir = "./miniset", index_type='Flat', nprobe=1):
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
     def build_ivf_index(self, nlist=10):
          dim = self.cluster_embeddings.shape[1]  
          quantizer = faiss.IndexFlatL2(dim) 
          self.index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_L2)
          self.index.train(self.cluster_embeddings)  
          self.index.add(self.cluster_embeddings)      

          res = faiss.StandardGpuResources()  
          quantizer = faiss.IndexFlatL2(dim)  
          quantizer = faiss.index_cpu_to_gpu(res, 0, quantizer)  

          self.index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_L2)
          self.index = faiss.index_cpu_to_gpu(res, 0, self.index)  # Move full index to GPU

          # Train and add embeddings
          self.index.train(self.cluster_embeddings)
          self.index.add(self.cluster_embeddings) 
     
     def search_queries(self, query_embeddings, top_k=5):
          distances, indices = self.index.search(query_embeddings, top_k)
          return distances, indices


def load_query_embeddings(file_path):
    df = pd.read_csv(file_path)
    return df.values.astype(np.float32)  




if __name__ == "__main__":
     dataset_dir = "miniset"
     query_emb_file = os.path.join(dataset_dir, "query_emb.csv")
     cluster_dir = dataset_dir 
     searcher = SearchUDL(cluster_dir = cluster_dir)
     query_embeddings = load_query_embeddings(query_emb_file)
     distances, indices = searcher.search_queries(query_embeddings, top_k=5)
     print(f"distances.shape: {distances.shape}")
     print(f"indices.shape: {indices.shape}")



