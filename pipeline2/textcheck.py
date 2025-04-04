# from transformers import BartForSequenceClassification, BartTokenizer, pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import time
import torch

def textcheck(raw_texts):
     model_path = "badmatr11x/distilroberta-base-offensive-hateful-speech-text-multiclassification"
     device = torch.device('cuda')
     tokenizer = AutoTokenizer.from_pretrained(model_path)
     model = AutoModelForSequenceClassification.from_pretrained(model_path).to('cuda')

     encoded_inputs = tokenizer(raw_texts, padding=True, truncation=True, return_tensors='pt').to(device)
     logits = model(**encoded_inputs)[0]
     probs = logits.softmax(dim=1)

     full_probs = probs.detach().cpu()
     list_of_ids = torch.argmax(full_probs, dim=1).tolist() 
     list_of_labels = [model.config.id2label[int(idx)] for idx in list_of_ids]
     print(f"the obtained labels are: {list_of_labels} ")
     return list_of_labels


if __name__ == "__main__":
     # Example usage
     # for i in range(100):
     list_of_texts = ["I want to kill Asians!",\
                    "This is a hateful statement against a particular group of people.", \
                    "The weather is nice today.", \
                    "Fuck Fuck Fuck, shit shit shit", \
                    "I think this is a great initiative to support education."]
     textcheck(list_of_texts)




