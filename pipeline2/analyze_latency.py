import argparse
import csv

def analyze_latency(file_path):
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        rows = list(reader)

    if len(rows) == 0 or len(rows[0]) == 0:
        raise ValueError("Latency CSV is empty or improperly formatted.")

    runtimes_ns = [int(float(x)) for x in rows[0]][50:]
    avg_ns = sum(runtimes_ns) / len(runtimes_ns)
    max_ns = max(runtimes_ns)
    min_ns = min(runtimes_ns)

    print(f"File: {file_path}")
    print(f"--- Runtime (ns) ---")
    print(f"Average: {avg_ns:.2f} ns")
    print(f"Maximum: {max_ns} ns")
    print(f"Minimum: {min_ns} ns")

    print(f"--- Runtime (ms) ---")
    print(f"Average: {avg_ns / 1e6:.4f} ms")
    print(f"Maximum: {max_ns / 1e6:.4f} ms")
    print(f"Minimum: {min_ns / 1e6:.4f} ms")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze latency CSV file (in nanoseconds).")
    parser.add_argument('file', type=str, help='Path to latency CSV file')
    args = parser.parse_args()

    analyze_latency(args.file)
