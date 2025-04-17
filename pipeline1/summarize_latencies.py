import os
import re
import csv
import statistics
import argparse

def extract_info_from_filename(filename):
    match = re.match(r"step\w+_bsize(\d+)_runtime(\d+)_tp([\d.]+)\.csv", filename)
    if not match:
        return None, None, None
    batch_size = int(match.group(1))
    runtime_id = int(match.group(2))
    throughput = float(match.group(3))
    return batch_size, runtime_id, throughput

def analyze_latency(file_path):
    try:
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
    except Exception as e:
        print(f"Error parsing latency file {file_path}: {e}")
        return None

def analyze_gpu_metrics(folder, runtime_id, batch_size):
    runtime_str = f"pid{int(runtime_id):03d}"
    pattern = re.compile(rf"^gpu_util_{runtime_str}_bsize{batch_size}\.csv$")

    for fname in os.listdir(folder):
        if pattern.match(fname):
            gpu_file = os.path.join(folder, fname)
            print(f"Analyzing GPU file: {gpu_file}")
            try:
                with open(gpu_file, 'r') as file:
                    lines = file.readlines()

                if len(lines) != 2:
                    print(f"Skipping {fname}: expected 2 lines (memory, utilization).")
                    return None, None, None, None

                mem_values = list(map(int, lines[0].strip().split(',')))
                util_values = list(map(int, lines[1].strip().split(',')))

                if not mem_values or not util_values:
                    return None, None, None, None

                max_mem = max(mem_values)
                avg_mem = sum(mem_values) / len(mem_values)

                max_util = max(util_values)
                avg_util = sum(util_values) / len(util_values)

                return max_mem, round(avg_mem, 2), max_util, round(avg_util, 2)

            except Exception as e:
                print(f"Error parsing GPU file {fname}: {e}")
                return None, None, None, None

    print(f"No GPU file found for runtime={runtime_str}, batch_size={batch_size}")
    return None, None, None, None

def summarize_all_steps(parent_folder, output_csv):
    step_folders = ["micro_stepA", "micro_stepB", "micro_stepCD", "micro_stepE"]
    output_rows = [
        ("type", "batch_size", "throughput",
         "avg_latency_ms", "med_latency_ms",
         "gpu_max_mem_MB", "gpu_avg_mem_MB",
         "gpu_max_util_percent", "gpu_avg_util_percent")
    ]

    for step_folder in step_folders:
        full_path = os.path.join(parent_folder, step_folder)
        if not os.path.isdir(full_path):
            print(f"Skipping missing folder: {full_path}")
            continue

        for filename in os.listdir(full_path):
            if not filename.startswith("step") or not filename.endswith(".csv"):
                continue

            batch_size, runtime_id, throughput = extract_info_from_filename(filename)
            if batch_size is None:
                continue

            file_path = os.path.join(full_path, filename)
            latency_stats = analyze_latency(file_path)
            if latency_stats is None:
                print(f"Skipping {filename} in {step_folder}: invalid or missing latency data.")
                continue

            avg_ms, med_ms = latency_stats
            step_type = step_folder.replace("micro_", "")

            max_mem, avg_mem, max_util, avg_util = analyze_gpu_metrics(full_path, runtime_id, batch_size)

            output_rows.append((
                step_type, batch_size, throughput,
                round(avg_ms, 4), round(med_ms, 4),
                max_mem, avg_mem, max_util, avg_util
            ))

    output_rows = [output_rows[0]] + sorted(output_rows[1:], key=lambda x: (x[0], x[1]))

    with open(output_csv, 'w', newline='') as out_f:
        writer = csv.writer(out_f)
        writer.writerows(output_rows)

    print(f"\nSummary saved to: {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize latency and GPU metrics from step output folders.")
    parser.add_argument("folder", type=str, help="Path to parent folder (contains micro_stepA, micro_stepB, etc.)")
    parser.add_argument("-o", "--output", type=str, default="all_steps_latency_gpu_summary.csv", help="Output CSV file")
    args = parser.parse_args()

    summarize_all_steps(args.folder, args.output)