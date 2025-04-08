import numpy as np
import faiss
from tqdm import tqdm
import time,csv
import torch

def train_and_save_index_gpu(
                    d, 
                    nlist, 
                    m, 
                    nbits, 
                    embeddings, 
                    ):
    quantizer = faiss.IndexFlatL2(d) # build a flat (CPU) index
    cpu_index = faiss.IndexIVFPQ(quantizer, d, nlist, m, nbits)
    res = faiss.StandardGpuResources()
    gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
    gpu_index.train(embeddings)
    gpu_index.add(embeddings)
    
    print(f"Finished indexing embeddings of number: {gpu_index.ntotal}")
    cpu_index = faiss.index_gpu_to_cpu(gpu_index)
    faiss.write_index(cpu_index, "./msmarco_full_passages.index")
    return gpu_index





if __name__ == "__main__":
    d       = 384           # Dimension of each embedding
    nb      = 8841823       # Number of embeddings (1m)
    nlist   = 1000          # Number of clusters for the coarse quantizer
    m       = 16            # Number of subquantizers (must divide d evenly)
    nbits   = 8             # Bits per sub-vector
    k       = 5
    n_times = 10000
    BS      = 128
    
    
    index_file = "/mydata/EVQA/MSMARCO/msmarco_pq.index"
    query_e_file = "/mydata/EVQA/MSMARCO/ms_macro_1m_queries_embeds.npy"
    # embeddings_file = "/mydata/EVQA/ms_macro_full_passages_embeds.npy"


    #### Hiden this section if not training ####
    # embeddings = np.load(embeddings_file)
    # assert embeddings.shape == (nb, d)
    # train_and_save_index_gpu(d, nlist, m, nbits, embeddings)

    #### this section for searching with gpu index #####
    cpu_index = faiss.read_index(index_file)
    res = faiss.StandardGpuResources()
    gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
    gpu_index.nprobe = 10
    
    
    query_vector = np.load(query_e_file)
    print(query_vector.shape)

    start_list = []
    end_list = []
    start = time.perf_counter()
    for i in tqdm(range(n_times)):
        start_list.append(time.perf_counter())
        res1, res2 = gpu_index.search(query_vector[i:(i+BS),:], k)
        torch.cuda.synchronize()
        end_list.append(time.perf_counter())
        
    end = time.perf_counter()
    

    print(f"Throughput is: {(n_times * BS) / (end - start)}")
    
    assert len(start_list) == len(end_list)
    runtime = [int((x - y)*10**9) for x, y in zip(end_list, start_list)]
    print(f"Average runtime per batch: {np.array(runtime).mean()}")    
    
    with open("ivfpq_runtime.csv", mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(runtime)
