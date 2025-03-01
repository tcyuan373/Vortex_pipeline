import csv
import json
import time

from flmr import search_custom_collection, create_searcher
import torch


late_interaction_size = 128


class StepE:
    def __init__(self, 
                 index_root_path='/mydata/EVQA/index',
                 index_experiment_name='EVQA_train_split',
                 index_name='EVQA_PreFLMR_ViT-L',
                 ):
        self.searcher = create_searcher(
            index_root_path=index_root_path,
            index_experiment_name=index_experiment_name,
            index_name=index_name,
            nbits=8, # number of bits in compression
            use_gpu=True, # break if set to False, see doc: https://docs.google.com/document/d/1KuWGWZrxURkVxDjFRy1Qnwsy7jDQb-RhlbUzm_A-tOs/edit?tab=t.0
        )

    # input: question, question_id, image_features, 
    def step_E_search(self, batch, query_embeddings):
        custom_quries = {
            question_id: question for question_id, question in zip(batch["question_id"], batch["question"])
        }
        ranking = search_custom_collection(
            searcher=self.searcher,
            queries=custom_quries,
            query_embeddings=query_embeddings,
            num_document_to_retrieve=5, # how many documents to retrieve for each query
            centroid_search_batch_size=None,
        )
        return ranking.todict()


if __name__=='__main__':
    stepE = StepE()

    # GPU memory usage after loading model
    print("Allocated memory after loading model:", torch.cuda.memory_allocated())
    print("Reserved memory after loading model:", torch.cuda.memory_reserved())

    load_input_times = []
    run_times = []

    embeddings = torch.load("qembeds.pt")
    with open("queries.json", "r") as f:
        queries = json.load(f)

    # CUDA events for accurate profiling
    # total_start_event = torch.cuda.Event(enable_timing=True)
    # total_end_event = torch.cuda.Event(enable_timing=True)
    # total start time for throughput calculation
    total_start_event = time.perf_counter_ns()

    total_runs = 100
    batch_size = 16
    i = 0
    keys = list(queries.keys())

    num_keys = len(keys)

    for _ in range(0, total_runs):
        query_embed_list = []
        query = {
            'question_id': [],
            'question': [],
        }
        for j in range(batch_size):
            cur_key = keys[i%num_keys]
            query['question_id'].append(cur_key)
            query['question'].append(queries[cur_key])
            query_embed_list.append(embeddings[i%num_keys])
            i+=1

        # CUDA events for accurate profiling
        # mvgpu_start_event = torch.cuda.Event(enable_timing=True)
        # mvgpu_end_event = torch.cuda.Event(enable_timing=True)
        # model_start_event = torch.cuda.Event(enable_timing=True)
        # model_end_event = torch.cuda.Event(enable_timing=True)

        query_embed = torch.stack(query_embed_list, dim=0)

        # time before put to GPU
        # mvgpu_start_event.record()
        query_embed = query_embed.cuda()
        # time after put to GPU
        # mvgpu_end_event.record()
        # torch.cuda.synchronize()
        # load_input_times.append((mvgpu_start_event.elapsed_time(mvgpu_end_event)) * 1e6)

        # time before running model
        model_start_event = time.perf_counter_ns()
        ranking_dict = stepE.step_E_search(query, query_embed)
        # time after running model
        model_end_event = time.perf_counter_ns()
        # torch.cuda.synchronize()
        run_times.append(model_end_event - model_start_event)

    # total end time for throughput calculation
    total_end_event = time.perf_counter_ns()
    # torch.cuda.synchronize()
    time_elapsed=(total_end_event - total_start_event)
    throughput = (total_runs * batch_size) / (time_elapsed / 1000000000)
    print("Throughput with batch size", batch_size, "(queries/s):", throughput)

    runtimes_file = 'step_E_runtime.csv'
    # gpu_transfer = 'step_E_transfer_to_gpu.csv'

    with open(runtimes_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(run_times)

    # with open(gpu_transfer, mode='w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(load_input_times)

    # print(ranking_dict)
