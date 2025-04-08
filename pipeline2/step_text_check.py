import argparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import csv
import pickle
import os
from pynvml import *

# === Global configuration ===
TOTAL_RUNS = 1000  # Number of batches to run

def init_nvml():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    return handle

def query_gpu_metrics(handle):
    mem_info = nvmlDeviceGetMemoryInfo(handle)
    util = nvmlDeviceGetUtilizationRates(handle)
    return mem_info.used // 1024**2, util.gpu  # Memory in MB, GPU utilization %

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
    run_times.append((model_start_event.elapsed_time(model_end_event)) * 1e6)  # µs → ns

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

    # === GPU metric tracking ===
    handle = init_nvml()
    memory_checkpoints = []
    utilization_checkpoints = []

    for _ in range(TOTAL_RUNS):
        batch_premise = [next(iterator) for _ in range(bsize)]

        # Log GPU usage before inference
        used_mem_mb, util_percent = query_gpu_metrics(handle)
        memory_checkpoints.append(used_mem_mb)
        utilization_checkpoints.append(util_percent)

        textcheck(batch_premise, run_times, tokenizer, model, device)

    throughput = (bsize * len(run_times)) / (sum(run_times) / 1e9)  # queries/sec
    avg_latency = int(sum(run_times) / len(run_times))

    print(f"Batch size {bsize}, throughput: {throughput:.2f} queries/sec")
    print(f"Average latency per batch: {avg_latency} ns")

    os.makedirs(output_dir, exist_ok=True)

    # === Save latency log ===
    runtimes_file = os.path.join(
        output_dir,
        f'roberta_tp{throughput:.2f}_text_check_batch{bsize}_runtime{pid}.csv'
    )
    with open(runtimes_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(run_times)

    # === Save GPU metrics log ===
    gpu_metrics_file = os.path.join(
        output_dir,
        f'roberta_gpu_tp{throughput:.2f}_batch{bsize}_runtime{pid}.csv'
    )
    with open(gpu_metrics_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(memory_checkpoints)
        writer.writerow(utilization_checkpoints)

    nvmlShutdown()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark RoBERTa inference timing with GPU metrics")
    parser.add_argument('-p', '--output_dir', type=str, required=True, help='Directory to store output CSV files')
    parser.add_argument('-id', '--pid', type=str, required=True, help='Process/GPU identifier (e.g., 000, MIG1, A100_FULL)')
    parser.add_argument('-b', '--bsize', type=int, required=True, help='Batch size for inference')
    args = parser.parse_args()

    main(args.output_dir, args.pid, args.bsize)