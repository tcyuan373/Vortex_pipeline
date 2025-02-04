from step_A_text_encoder import *
from step_B_vision_encoder import *
from step_C_modeling_mlp import *
from step_D_FLMR_retrieval import *
from step_E_search import *



if __name__ == "__main__":
    # preparing raw input texts
    raw_sentences = ['This is a test text sequence', "my puppy's name is GOJI"]
    # preparing raw input images
    img_root = './images'
    img_paths = [os.path.join(img_root, item) for item in os.listdir(img_root)]
    index_root = '.'
    list_of_images = []
    for img_path in img_paths:
        image = Image.open(img_path).convert("RGB")
        list_of_images.append(image)
    
    # proceed with the steps
    stepA = StepA()
    stepA.load_model_cuda()
    txt_embed, input_ids, txt_encoder_hs = stepA.stepA_output(raw_sentences)
    
    stepb = StepB()
    stepb.load_model_cuda()
    vision_embeddings, vision_second_last_layer_hidden_states= stepb.StepB_output(list_of_images)
    
    stepc = StepC()
    stepc.load_model_cuda()
    mapping_input_features = stepc.stepC_output(vision_second_last_layer_hidden_states)
    
    stepD = step_D_transformer_mapping()
    stepD.load_model_cuda()
    query_embeddings = stepD.cross_attn_embedding(
        input_ids, 
        txt_embed, 
        txt_encoder_hs, 
        vision_embeddings, 
        mapping_input_features
        )
    print(query_embeddings.shape)
    
    dummy_dict = {
        'question_id': [0, 1],
        'question': ["GOJIGOJIGOJI", 'test test test test']
    }
    
    stepE = StepE()
    ranking_dict = stepE.step_E_search(dummy_dict, query_embeddings)
    print(ranking_dict)