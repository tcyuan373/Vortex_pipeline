import csv
import time

from flmr import search_custom_collection, create_searcher
import torch
import random


bsize = 8
late_interaction_size = 128


class StepE:
    def __init__(self, 
                 index_root_path='.',
                 index_experiment_name='EVQA_test_split',
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
            num_document_to_retrieve=2, # how many documents to retrieve for each query
            centroid_search_batch_size=bsize,
        )
        return ranking.todict()


if __name__=='__main__':
    dummy_dict = {
        'question_id': [0],
        'question': ["test sentence test sentece, this this, 100"],
    }

    stepE = StepE()

    # GPU memory usage after loading model
    print("Allocated memory after loading model:", torch.cuda.memory_allocated())
    print("Reserved memory after loading model:", torch.cuda.memory_reserved())

    load_input_times = []
    run_times = []

    # CUDA events for accurate profiling
    total_start_event = torch.cuda.Event(enable_timing=True)
    total_end_event = torch.cuda.Event(enable_timing=True)
    # total start time for throughput calculation
    total_start_event.record()

    for i in range(100):
        # CUDA events for accurate profiling
        mvgpu_start_event = torch.cuda.Event(enable_timing=True)
        mvgpu_end_event = torch.cuda.Event(enable_timing=True)
        model_start_event = torch.cuda.Event(enable_timing=True)
        model_end_event = torch.cuda.Event(enable_timing=True)

        dummy_query_embed = torch.randn(bsize, random.randint(500, 1000), late_interaction_size)

        # time before put to GPU
        mvgpu_start_event.record()
        dummy_query_embed = dummy_query_embed.cuda()
        # time after put to GPU
        mvgpu_end_event.record()
        torch.cuda.synchronize()
        load_input_times.append((mvgpu_start_event.elapsed_time(mvgpu_end_event)) * 1e6)

        # time before running model
        model_start_event.record()
        ranking_dict = stepE.step_E_search(dummy_dict, dummy_query_embed)
        # time after running model
        model_end_event.record()
        torch.cuda.synchronize()
        run_times.append((model_start_event.elapsed_time(model_end_event)) * 1e6)

    # total end time for throughput calculation
    total_end_event.record()
    torch.cuda.synchronize()
    time_elapsed=(total_start_event.elapsed_time(total_end_event)) * 1e6
    throughput = (100 * bsize) / (time_elapsed / 1000000000)
    print("Throughput with batch size", bsize, "(queries/s):", throughput)

    runtimes_file = 'step_E_runtime.csv'
    gpu_transfer = 'step_E_transfer_to_gpu.csv'

    with open(runtimes_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(run_times)

    with open(gpu_transfer, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(load_input_times)

    print(ranking_dict)
