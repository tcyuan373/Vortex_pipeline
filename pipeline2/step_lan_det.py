import argparse
import torch
import time
import csv
import pickle
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# === Global configuration ===
TOTAL_RUNS = 500  # Number of inference batches to run
FILE_PATH = '/mydata/msmarco/queries_audio5000.pkl'

def language_detection(batch_text, run_times, tokenizer, model, device):
    inputs = tokenizer(batch_text, padding=True, truncation=True, return_tensors="pt").to(device)

    model_start_event = torch.cuda.Event(enable_timing=True)
    model_end_event = torch.cuda.Event(enable_timing=True)

    model_start_event.record()
    with torch.no_grad():
        logits = model(**inputs).logits
    model_end_event.record()

    torch.cuda.synchronize()
    run_times.append((model_start_event.elapsed_time(model_end_event)) * 1e6)  # microseconds to nanoseconds

    probs = torch.softmax(logits, dim=-1)
    predictions = torch.argmax(probs, dim=1).tolist()
    labels = [model.config.id2label[idx] for idx in predictions]
    return labels

def main(output_dir, pid, bsize):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_ckpt = "papluca/xlm-roberta-base-language-detection"
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    model = AutoModelForSequenceClassification.from_pretrained(model_ckpt).to(device)

    with open(FILE_PATH, 'rb') as file:
        data = pickle.load(file)

    iterator = iter(data)
    run_times = []

    for _ in range(TOTAL_RUNS):
        batch_text = [next(iterator) for _ in range(bsize)]
        language_detection(batch_text, run_times, tokenizer, model, device)

    throughput = (bsize * len(run_times)) / (sum(run_times) / 1e9)
    avg_latency = int(sum(run_times) / len(run_times))

    print(f"Batch size {bsize}, throughput is {throughput:.2f} queries/sec")
    print(f"Average latency per batch: {avg_latency} ns")

    os.makedirs(output_dir, exist_ok=True)
    runtimes_file = os.path.join(
        output_dir,
        f'langdet_batch{bsize}_runtime{pid}_tp{throughput:.2f}.csv'
    )

    with open(runtimes_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(run_times)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark XLM-R Language Detection")
    parser.add_argument('-p', '--output_dir', type=str, required=True, help='Directory to store output CSV')
    parser.add_argument('-id', '--pid', type=str, required=True, help='String identifier for MIG (e.g., 0,1,2,3,000 if no MIG)')
    parser.add_argument('-b', '--bsize', type=int, required=True, help='Batch size for inference')
    args = parser.parse_args()

    main(args.output_dir, args.pid, args.bsize)