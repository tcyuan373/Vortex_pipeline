# pretrained_uncased_BERT with linear projection
# add load project layer weights
import os
import torch
from torch import Tensor, nn
from flmr import FLMRConfig, FLMRQueryEncoderTokenizer, FLMRContextEncoderTokenizer, FLMRModelForRetrieval, FLMRTextModel
# input text strings
# output: text_embeddings, input_ids[gen mask], encoder_hidden_states

class StepA:
    def __init__(self,):
        self.checkpoint_path = 'LinWeizheDragon/PreFLMR_ViT-L'
        self.flmr_config = FLMRConfig.from_pretrained(self.checkpoint_path)
        self.local_encoder_path = 'models_step_A_query_text_encoder.pt'
        self.local_projection_path = 'models_step_A_query_text_linear.pt'
        self.query_tokenizer = FLMRQueryEncoderTokenizer.from_pretrained(
                self.checkpoint_path, 
                text_config=self.flmr_config.text_config, 
                subfolder="query_tokenizer")
        self.context_tokenizer = FLMRContextEncoderTokenizer.from_pretrained(
            self.checkpoint_path, 
            text_config=self.flmr_config.text_config, 
            subfolder="context_tokenizer"
        )
        
        if not os.path.exists(self.local_encoder_path) and not os.path.exists(self.local_projection_path):
            print('local model not found, initing from full model...')
            
            full_model = FLMRModelForRetrieval.from_pretrained(
                self.checkpoint_path,
                query_tokenizer=self.query_tokenizer,
                context_tokenizer=self.context_tokenizer,
            )
            self.query_text_encoder = full_model.query_text_encoder
            self.query_text_encoder_linear = full_model.query_text_encoder_linear
            # torch.save(self.query_text_encoder.state_dict(), self.local_encoder_path)
            # torch.save(self.query_text_encoder_linear.state_dict(), self.local_projection_path)
            del full_model
        else:
            print(f'found local model for step A, now loading...')

            self.query_text_encoder = FLMRTextModel(self.flmr_config.text_config)
            self.query_text_encoder.load_state_dict(torch.load(self.local_encoder_path, weights_only=True))
            self.query_text_encoder_linear = nn.Linear(self.flmr_config.text_config.hidden_size, self.flmr_config.dim, bias=False)
            self.query_text_encoder_linear.load_state_dict(torch.load(self.local_projection_path, weights_only=True))
            
            
        self.skiplist = []
        
        if self.flmr_config.mask_instruction_token is not None:
            self.mask_instruction = True
            # obtain the token id of the instruction token
            self.instruction_token_id = self.query_tokenizer.encode(
                self.flmr_config.mask_instruction_token, add_special_tokens=False
            )[0]
        else:
            self.mask_instruction = False
        
    
    def load_model_cuda(self):
        self.query_text_encoder_linear.to('cuda')
        self.query_text_encoder.to('cuda')
    
    
    def stepA_output(
        self,
        input_text_sequence,
    ):

        # query sentences: bsize of sentences
        encoded_inputs      = self.query_tokenizer(input_text_sequence)
        input_ids           = encoded_inputs['input_ids'].to(self.query_text_encoder.device)
        attention_mask      = encoded_inputs['attention_mask'].to(self.query_text_encoder.device)
        
        text_encoder_outputs = self.query_text_encoder(input_ids=input_ids,attention_mask=attention_mask,)
        text_encoder_hidden_states = text_encoder_outputs[0]
        text_embeddings = self.query_text_encoder_linear(text_encoder_hidden_states)

        # note, text_embeddings not masked yet here!!!!
        
        return text_embeddings, input_ids, text_encoder_hidden_states
    
    
if __name__ == '__main__':
        raw_sentences = ['Hello World', 'This is a test text sequence', 'I have a cute puppy', "my puppy's name is GOJI"]
        stepA = StepA()
        stepA.load_model_cuda()
        txt_embed, input_ids, txt_encoder_hs = stepA.stepA_output(raw_sentences)
        print(f'text embedding shape: {txt_embed.shape} | input ids shape: {input_ids.shape} | hidden_states shape: {txt_encoder_hs.shape}')
        print(stepA.query_text_encoder_linear.weight.cpu().detach().numpy())
