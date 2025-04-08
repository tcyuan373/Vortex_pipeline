import argparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import csv
import pickle
import os

# === Global configuration ===
TOTAL_RUNS = 1000  # Number of batches to run

def textcheck(batch_premise, run_times, tokenizer, model, device):
    inputs = tokenizer(batch_premise,
                       return_tensors='pt', padding=True, truncation=True).to(device)

    model_start_event = torch.cuda.Event(enable_timing=True)
    model_end_event = torch.cuda.Event(enable_timing=True)

    model_start_event.record()
    with torch.no_grad():
        result = model(**inputs)
    model_end_event.record()

    torch.cuda.synchronize()
    run_times.append((model_start_event.elapsed_time(model_end_event)) * 1e6)  # microseconds to nanoseconds

    logits = result.logits
    probs = logits.softmax(dim=1)

    full_probs = probs.detach().cpu()
    list_of_ids = torch.argmax(full_probs, dim=1).tolist()
    list_of_labels = [model.config.id2label[int(idx)] for idx in list_of_ids]
    return list_of_labels

def main(output_dir, pid, bsize):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(
        'badmatr11x/distilroberta-base-offensive-hateful-speech-text-multiclassification')
    model = AutoModelForSequenceClassification.from_pretrained(
        'badmatr11x/distilroberta-base-offensive-hateful-speech-text-multiclassification').to(device)

    file_path = '/mydata/msmarco/msmarco_3_clusters/doc_list.pkl'
    with open(file_path, 'rb') as file:
        data = pickle.load(file)

    iterator = iter(data)
    run_times = []

    for _ in range(TOTAL_RUNS):
        batch_premise = [next(iterator) for _ in range(bsize)]
        textcheck(batch_premise, run_times, tokenizer, model, device)

    throughput = (bsize * len(run_times)) / (sum(run_times) / 1e9)
    avg_latency = int(sum(run_times) / len(run_times))

    print(f"Batch size {bsize}, throughput is {throughput} queries/sec")
    print(f"Average latency per batch: {avg_latency} ns")

    os.makedirs(output_dir, exist_ok=True)
    runtimes_file = os.path.join(
        output_dir,
        f'roberta_tp{throughput:.2f}_text_check_batch{bsize}_runtime{pid}.csv'
    )

    with open(runtimes_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(run_times)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark RoBERTa inference timing")
    parser.add_argument('-p', '--output_dir', type=str, required=True, help='Directory to store output CSV')
    parser.add_argument('-id', '--pid', type=str, required=True, help='String identifier for process/GPU setup (e.g., 000, MIG1, A100_FULL)')
    parser.add_argument('-b', '--bsize', type=int, required=True, help='Batch size for inference')
    args = parser.parse_args()

    main(args.output_dir, args.pid, args.bsize)
