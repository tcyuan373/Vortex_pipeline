import os
import re
import csv
import statistics
import argparse

def extract_info_from_filename(filename):
    match = re.match(r"step\w+_bsize(\d+)_runtime\d+_tp([\d.]+)\.csv", filename)
    if not match:
        return None, None
    batch_size = int(match.group(1))
    throughput = float(match.group(2))
    return batch_size, throughput

def analyze_latency(file_path):
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        rows = list(reader)

    if not rows or not rows[0]:
        return None

    runtimes_ns = [int(float(x)) for x in rows[0]][50:]  # Skip warmup
    if not runtimes_ns:
        return None

    avg_latency_ms = sum(runtimes_ns) / len(runtimes_ns) / 1e6
    med_latency_ms = statistics.median(runtimes_ns) / 1e6
    return avg_latency_ms, med_latency_ms

def summarize_all_steps(parent_folder, output_csv):
    step_folders = ["micro_stepA", "micro_stepB", "micro_stepCD", "micro_stepE"]
    output_rows = [("type", "batch_size", "throughput", "avg_latency_ms", "med_latency_ms")]

    for step_folder in step_folders:
        full_path = os.path.join(parent_folder, step_folder)
        if not os.path.isdir(full_path):
            print(f"Skipping missing folder: {full_path}")
            continue

        for filename in os.listdir(full_path):
            if not filename.startswith("step") or not filename.endswith(".csv"):
                continue

            batch_size, throughput = extract_info_from_filename(filename)
            if batch_size is None:
                continue

            file_path = os.path.join(full_path, filename)
            latency_stats = analyze_latency(file_path)
            if latency_stats is None:
                print(f"Skipping {filename} in {step_folder}: invalid content")
                continue

            avg_ms, med_ms = latency_stats
            step_type = step_folder.replace("micro_", "")
            output_rows.append((step_type, batch_size, throughput, round(avg_ms, 4), round(med_ms, 4)))

    # Sort by type, then batch size
    output_rows = [output_rows[0]] + sorted(output_rows[1:], key=lambda x: (x[0], x[1]))

    with open(output_csv, 'w', newline='') as out_f:
        writer = csv.writer(out_f)
        writer.writerows(output_rows)

    print(f"Saved combined summary to {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize all micro step latencies into one CSV.")
    parser.add_argument("folder", type=str, help="Parent folder containing micro_stepA, micro_stepB, etc.")
    parser.add_argument("-o", "--output", type=str, default="all_steps_latency_summary.csv")
    args = parser.parse_args()

    summarize_all_steps(args.folder, args.output)