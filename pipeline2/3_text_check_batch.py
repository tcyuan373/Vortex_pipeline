from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import csv
import pickle

def textcheck(batch_premise, run_times):
    hypothesis = 'harmful.'

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

    # Take only entailment and contradiction logits
    entail_contradiction_logits = logits[:, [0, 2]]
    probs = entail_contradiction_logits.softmax(dim=1)
    true_probs = probs[:, 1] * 100  # Probability of entailment

    # for i, prob in enumerate(true_probs.tolist()):
    #     print(f'Premise {i+1}: Probability that the label is true: {prob:.2f}%')

    return true_probs.tolist()


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained('badmatr11x/distilroberta-base-offensive-hateful-speech-text-multiclassification')
    model = AutoModelForSequenceClassification.from_pretrained('badmatr11x/distilroberta-base-offensive-hateful-speech-text-multiclassification').to('cuda')


    file_path = '/mydata/msmarco/msmarco_3_clusters/doc_list.pkl'

    with open(file_path, 'rb') as file:
        data = pickle.load(file)

    iterator = iter(data)

    bsize = 4
    run_times = []

    for i in range(1000):
        batch_premise = []
        for j in range(bsize):
            batch_premise.append(next(iterator))
        textcheck(batch_premise, run_times)

    runtimes_file = 'text_check_batch_runtime.csv'

    with open(runtimes_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(run_times)
    throughput = (bsize * len(run_times)) / (sum(run_times) / 1e9)
    print(f"throughput is {throughput}")
