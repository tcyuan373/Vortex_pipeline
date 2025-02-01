from transformers import BertConfig, BertModel, BertTokenizer
from transformers.models.bert.modeling_bert import BertEncoder
from transformers.modeling_outputs import BaseModelOutput
import torch

bsize = 8
seqlen = 10
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
configuration = BertConfig.from_pretrained('bert-base-uncased')

raw_sentences = [f"this is a raw text for the {i} input, appending more means much longer ctx length" for i in range(bsize)]
# Initializing a model (with random weights) from the google-bert/bert-base-uncased style configuration
configuration.num_hidden_layers = 1
configuration.is_decoder = True
configuration.add_cross_attention = True

full_model = BertModel(configuration)
Enc_model = BertEncoder(configuration)

if __name__ == "__main__":
    encoded_input = tokenizer(raw_sentences, padding=True, truncation=True, return_tensors='pt')
    print(encoded_input)
    for key in encoded_input.keys():
        print(f'the input data type for {key} is:{encoded_input[key].dtype} \t | plus shape: {encoded_input[key].shape}')
    output = full_model(**encoded_input)
    print(output.keys())
    for key in output.keys():
        print(f'the output data type for {key} is:{output[key][0][0].dtype} \t | plus shape: {output[key][0][0].shape}')
        
    # dummy_enc_input_features = torch.randn([bsize, 768])
    # enc_output = Enc_model(dummy_enc_input_features, encoder_hidden_states= )
    # Enc_model()