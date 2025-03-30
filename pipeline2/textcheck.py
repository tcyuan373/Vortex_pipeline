# from transformers import BartForSequenceClassification, BartTokenizer, pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import time
import torch

def textcheck(premise):
     device = torch.device('cuda')
     tokenizer = AutoTokenizer.from_pretrained('SamLowe/roberta-base-go_emotions')
     model = AutoModelForSequenceClassification.from_pretrained('SamLowe/roberta-base-go_emotions').to('cuda')

     # pose sequence as a NLI premise and label (politics) as a hypothesis
     # premise = 'I hate Asians!'
     # premise = 'A new model offers an explanation for how the Galilean satellites formed around the solar system’s largest world.'
     hypothesis = 'harmful.'

     # run through model pre-trained on MNLI
     input_ids = tokenizer.encode(premise, hypothesis, return_tensors='pt').to(device)
     logits = model(input_ids)[0]

     # we throw away "neutral" (dim 1) and take the probability of
     # "entailment" (2) as the probability of the label being true 
     entail_contradiction_logits = logits[:,[0,2]]
     probs = entail_contradiction_logits.softmax(dim=1)
     true_prob = probs[:,1].item() * 100
     print(f'Probability that the label is true: {true_prob:0.2f}%')
     return true_prob


if __name__ == "__main__":
     # Example usage
     for i in range(100):
          premise = "I love programming in Python!"
          textcheck(premise)




