import argparse
import os
import torch
import pickle
import csv
from transformers import BartTokenizer, BartForSequenceClassification

# === Global configuration ===
TOTAL_RUNS = 1000  # Number of batches to run
FILE_PATH = '/mydata/msmarco/msmarco_3_clusters/doc_list.pkl'
HYPOTHESIS = "harmful."


def textcheck(batch_premise, run_times, tokenizer, model, device):
    inputs = tokenizer(batch_premise,
                       [HYPOTHESIS] * len(batch_premise),
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
    entail_contradiction_logits = logits[:, [0, 2]]  # entailment = index 2
    probs = entail_contradiction_logits.softmax(dim=1)
    true_probs = probs[:, 1] * 100  # entailment probability
    return true_probs.tolist()


def main(output_dir, pid, bsize):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-mnli')
    model = BartForSequenceClassification.from_pretrained('facebook/bart-large-mnli').to(device)

    with open(FILE_PATH, 'rb') as file:
        data = pickle.load(file)

    iterator = iter(data)
    run_times = []

    for _ in range(TOTAL_RUNS):
        batch_premise = [next(iterator) for _ in range(bsize)]
        _ = textcheck(batch_premise, run_times, tokenizer, model, device)

    throughput = (bsize * len(run_times)) / (sum(run_times) / 1e9)
    avg_latency = int(sum(run_times) / len(run_times))

    print(f"Batch size {bsize}, throughput is {throughput:.2f} queries/sec")
    print(f"Average latency per batch: {avg_latency} ns")

    os.makedirs(output_dir, exist_ok=True)
    runtimes_file = os.path.join(
        output_dir,
        f'bartnli_batch{bsize}_runtime{pid}_tp{throughput:.2f}.csv'
    )

    with open(runtimes_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(run_times)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark BART NLI (entailment) timing")
    parser.add_argument('-p', '--output_dir', type=str, required=True, help='Directory to store output CSV')
    parser.add_argument('-id', '--pid', type=str, required=True, help='String identifier for process/GPU setup')
    parser.add_argument('-b', '--bsize', type=int, required=True, help='Batch size for inference')
    args = parser.parse_args()

    main(args.output_dir, args.pid, args.bsize)