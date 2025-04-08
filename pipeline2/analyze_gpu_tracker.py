import argparse
import csv

def analyze_gpu_metrics(file_path):
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        rows = list(reader)

    if len(rows) != 2:
        raise ValueError("Expected 2 rows in the CSV: first for memory, second for utilization.")

    # Parse values as integers
    memory_values = list(map(int, rows[0]))
    utilization_values = list(map(int, rows[1]))

    # Compute stats
    max_memory = max(memory_values)
    avg_memory = sum(memory_values) / len(memory_values)

    max_util = max(utilization_values)
    avg_util = sum(utilization_values) / len(utilization_values)

    # Display results
    print(f"File: {file_path}")
    print(f"Memory Usage (MB): max = {max_memory}, avg = {avg_memory:.2f}")
    print(f"GPU Utilization (%): max = {max_util}, avg = {avg_util:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze GPU memory and utilization metrics from a CSV file.")
    parser.add_argument('file', type=str, help='Path to GPU metrics CSV file (2-row format)')
    args = parser.parse_args()

    analyze_gpu_metrics(args.file)
