from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import csv
import pickle

def textcheck(batch_premise, run_times):
    # hypothesis = 'harmful.' # not used by this model

    # Tokenize all premises as a batch
    inputs = tokenizer(batch_premise,
                       return_tensors='pt', padding=True, truncation=True).to(device)

    model_start_event = torch.cuda.Event(enable_timing=True)
    model_end_event = torch.cuda.Event(enable_timing=True)

    # time before running model
    model_start_event.record()

    # Forward pass
    with torch.no_grad():
        result = model(**inputs)

    # time after running model
    model_end_event.record()
    torch.cuda.synchronize()
    run_times.append((model_start_event.elapsed_time(model_end_event)) * 1e6)

    logits = result.logits  # result[0] is now deprecated, use result.logits instead
    probs = logits.softmax(dim=1)

    full_probs = probs.detach().cpu()
    list_of_ids = torch.argmax(full_probs, dim=1).tolist() 
    list_of_labels = [model.config.id2label[int(idx)] for idx in list_of_ids]
    print(f"the obtained labels are: {list_of_labels} ")
    return list_of_labels


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained('badmatr11x/distilroberta-base-offensive-hateful-speech-text-multiclassification')
    model = AutoModelForSequenceClassification.from_pretrained('badmatr11x/distilroberta-base-offensive-hateful-speech-text-multiclassification').to('cuda')


    file_path = '/mydata/msmarco/msmarco_3_clusters/doc_list.pkl'

    with open(file_path, 'rb') as file:
        data = pickle.load(file)

    iterator = iter(data)

    bsize = 2
    run_times = []

    for i in range(5):
        batch_premise = []
        for j in range(bsize):
            batch_premise.append(next(iterator))
        textcheck(batch_premise, run_times)
    throughput = (bsize * len(run_times)) / (sum(run_times) / 1e9)
    print(f"batch size {bsize}, throughput is {throughput}")
    avg_latency = int(sum(run_times) / len(run_times))
    print(f"avg latency is {avg_latency} ns")

    runtimes_file = 'roberta_tp' + str(throughput)+ '_text_check_batch' + str(bsize) + '_runtime.csv'

    with open(runtimes_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(run_times)
