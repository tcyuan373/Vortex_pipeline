from transformers import BartForSequenceClassification, BartTokenizer, pipeline
import csv
import time
import torch
import pickle

def textcheck(premise, run_times):
     # pose sequence as a NLI premise and label (politics) as a hypothesis
     # premise = 'I hate Asians!'
     # premise = 'A new model offers an explanation for how the Galilean satellites formed around the solar system’s largest world.'
     hypothesis = 'harmful.'

     # run through model pre-trained on MNLI
     input_ids = tokenizer.encode(premise, hypothesis, return_tensors='pt').to('cuda')

     model_start_event = torch.cuda.Event(enable_timing=True)
     model_end_event = torch.cuda.Event(enable_timing=True)

     # time before running model
     model_start_event.record()
     result = model(input_ids)
     # time after running model
     model_end_event.record()
     torch.cuda.synchronize()
     run_times.append((model_start_event.elapsed_time(model_end_event)) * 1e6)

     # print(f"result shape: {len(result)}")
     logits = result[0]

     # we throw away "neutral" (dim 1) and take the probability of
     # "entailment" (2) as the probability of the label being true 
     entail_contradiction_logits = logits[:,[0,2]]
     probs = entail_contradiction_logits.softmax(dim=1)
     true_prob = probs[:,1].item() * 100
     # print(f"true_prob shape: {probs.shape}")
     # print(f'Probability that the label is true: {true_prob:0.2f}%')
     return true_prob

file_path = '/mydata/msmarco/msmarco_3_clusters/doc_list.pkl'

with open(file_path, 'rb') as file:
     data = pickle.load(file)

iterator = iter(data)

device = torch.device('cuda')
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-mnli', device_map = device)
model = BartForSequenceClassification.from_pretrained('facebook/bart-large-mnli').to('cuda')

bsize = 8
run_times = []

for i in range(1000):
     text = []
     for j in range(bsize):
          text.append(next(iterator))

     textcheck(text, run_times)

runtimes_file = 'textcheck_runtime.csv'

with open(runtimes_file, mode='w', newline='') as file:
     writer = csv.writer(file)
     writer.writerow(run_times)
