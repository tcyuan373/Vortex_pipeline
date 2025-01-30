from flmr import search_custom_collection, create_searcher
import torch
import random


bsize = 32
late_interaction_size = 128


class StepE:
    def __init__(self):
            
        
        self.searcher = create_searcher(
            index_root_path='.',
            index_experiment_name='test_experiment',
            index_name='test_index',
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
            centroid_search_batch_size=1,
        )
        return ranking.todict()





if __name__=='__main__':
    
    dummy_dict = {
        'question_id': [0],
        'question': ["test sentence test sentece, this this, 100"],
        
    }
    dummy_query_embed = torch.randn(bsize, random.randint(500, 1000), late_interaction_size)
    stepE = StepE()
    ranking_dict = stepE.step_E_search(dummy_dict, dummy_query_embed)
    print(ranking_dict)