import os
import re
import csv
from collections import defaultdict
import argparse

def extract_batch_and_tp(filename):
    match = re.search(r'bsize(\d+)_runtime\d+_tp([\d.]+)\.csv', filename)
    if match:
        return int(match.group(1)), float(match.group(2))
    return None, None

def process_config_folder(config_name, config_path):
    batch_to_throughputs = defaultdict(list)

    for root, dirs, files in os.walk(config_path):
        for file in files:
            if file.startswith("stepE_bsize") and file.endswith(".csv"):
                batch_size, tp = extract_batch_and_tp(file)
                if batch_size is not None:
                    batch_to_throughputs[batch_size].append(tp)

    return batch_to_throughputs

def summarize_all_configs(parent_dir, configs, output_csv):
    all_data = []

    for config in configs:
        config_path = os.path.join(parent_dir, config)
        if not os.path.isdir(config_path):
            print(f"Skipping missing config: {config_path}")
            continue

        batch_tp_map = process_config_folder(config, config_path)

        for batch_size in sorted(batch_tp_map.keys()):
            throughputs = batch_tp_map[batch_size]
            total_tp = round(sum(throughputs), 2)
            for i, tp in enumerate(throughputs):
                all_data.append((config, batch_size, f"component_{i}", tp, total_tp))

    # Output to CSV
    all_data.sort(key=lambda x: (x[0], x[1], x[2]))  # config, batch_size, component
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["config", "batch_size", "component", "throughput", "total_throughput"])
        writer.writerows(all_data)

    print(f"Saved summarized throughput data to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize stepE throughput across deployment configurations.")
    parser.add_argument("parent_dir", type=str, help="Folder containing NOMIG, MIG2, MIG4, processes2, etc.")
    args = parser.parse_args()
    filename = "stepE_deployment_throughput_summary.csv"

    configs = ["NOMIG", "MIG2", "MIG4", "processes2", "processes4"]
    summarize_all_configs(args.parent_dir, configs, filename)