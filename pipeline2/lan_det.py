import torch, time
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import pickle
import csv

file_path = '/mydata/msmarco/msmarco_3_clusters/doc_list.pkl'

with open(file_path, 'rb') as file:
    data = pickle.load(file)

iterator = iter(data)

model_ckpt = "papluca/xlm-roberta-base-language-detection"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = AutoModelForSequenceClassification.from_pretrained(model_ckpt).to('cuda')

#model_path = hf_hub_download(repo_id="facebook/fasttext-language-identification", filename="model.bin")


#model = fasttext.load_model(model_path)
bsize = 4
run_times = []

total_start_event = torch.cuda.Event(enable_timing=True)
total_end_event = torch.cuda.Event(enable_timing=True)

total_start_event.record()

nruns= 500
for i in range(nruns):
    text = []
    for j in range(bsize):
        text.append(next(iterator))

    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt").to('cuda')

    model_start_event = torch.cuda.Event(enable_timing=True)
    model_end_event = torch.cuda.Event(enable_timing=True)

    model_start_event.record()
    with torch.no_grad():
        logits = model(**inputs).logits
    model_end_event.record()
    torch.cuda.synchronize()
    run_times.append((model_start_event.elapsed_time(model_end_event)) * 1e6)

    preds = torch.softmax(logits, dim=-1)

total_end_event.record()
torch.cuda.synchronize()
time_elapsed=(total_start_event.elapsed_time(total_end_event)) * 1e6
#throughput = (1000 * bsize) / (time_elapsed / 1000000000)
#print("Throughput with batch size", bsize, "(queries/s):", throughput)

# Map raw predictions to languages
id2lang = model.config.id2label
vals, idxs = torch.max(preds, dim=1)
#print({id2lang[k.item()]: v.item() for k, v in zip(idxs, vals)})

throughput = (bsize * len(run_times)) / (sum(run_times) / 1e9)
print(f"batch size {bsize}, throughput is {throughput}")
avg_latency = int(sum(run_times) / len(run_times))
print(f"avg latency is {avg_latency} ns")

runtimes_file = 'rt_lan_det_tp' + str(throughput) + 'bsize' + str(bsize)+ '_runtime.csv'

with open(runtimes_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(run_times)
