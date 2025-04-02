from flmr import search_custom_collection, create_searcher, FLMRConfig, FLMRQueryEncoderTokenizer
import torch
import random
from transformers import AutoImageProcessor

bsize = 32
late_interaction_size = 128


class StepE:
    def __init__(self, 
                 index_root_path='.',
                 index_experiment_name='test_experiment',
                 index_name='test_index',
                 ):
        self.searcher = create_searcher(
            index_root_path=index_root_path,
            index_experiment_name=index_experiment_name,
            index_name=index_name,
            nbits=8, # number of bits in compression
            use_gpu=True, # break if set to False, see doc: https://docs.google.com/document/d/1KuWGWZrxURkVxDjFRy1Qnwsy7jDQb-RhlbUzm_A-tOs/edit?tab=t.0
        )

    # input: question, question_id, image_features, 
    def step_E_search(self, queries, query_embeddings):
        ranking = search_custom_collection(
            searcher=self.searcher,
            queries=queries,
            query_embeddings=query_embeddings,
            num_document_to_retrieve=2, # how many documents to retrieve for each query
            centroid_search_batch_size=1,
        )
        return ranking.todict()



if __name__=='__main__':
    
    # dummy_dict = {f
    #     'question_id': [0],
    #     'question': ["test sentence test sentece, this this, 100"],
    # }
    # question_ids = [0, 1, 2]
    # dummy_dict = {
    #     'question_id': question_ids,
    #     'question': ["test sentence test sentece, this this, 100", "GOJI", "I love puppies"],
    #     'embedding': [torch.randn(3, 320, 128) for i in range(len(question_ids))]
    # }
    num_queries = 2

    dummy_q_embeds = torch.randn(num_queries, 320, 128)
    query_instructions = [f"instruction {i}" for i in range(num_queries)]
    query_texts = [f"{query_instructions[i]} : query {i}" for i in range(num_queries)]
    queries = {i: query_texts[i] for i in range(num_queries)}

    stepE = StepE()
    ranking_dict = stepE.step_E_search(queries, dummy_q_embeds)
    print(ranking_dict)