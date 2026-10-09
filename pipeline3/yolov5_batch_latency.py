import argparse
import csv
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from download_yolov5 import download_yolov5_weights, load_yolov5
from generate_sample_images import create_sample_images, repeat_image_paths

DEFAULT_BATCH_SIZES = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 28, 30, 32, 64]


def read_gpu_snapshot() -> dict:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        parts = [float(v.strip()) for v in result.stdout.strip().split(",") if v.strip()]
        if len(parts) >= 3:
            return {
                "gpu_util_pct": parts[0],
                "mem_used_mb": parts[1],
                "mem_total_mb": parts[2],
            }
    except Exception:
        pass

    return {
        "gpu_util_pct": None,
        "mem_used_mb": None,
        "mem_total_mb": None,
    }


def benchmark_batch_size(model, batch_image_paths: list[str], img_size: int = 640, warmup_steps: int = 2, repeat_steps: int = 5) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark. No GPU was detected.")

    device = torch.device("cuda:0")
    model.to(device)
    model.eval()

    for _ in range(warmup_steps):
        images = [Image.open(path).convert("RGB") for path in batch_image_paths]
        with torch.no_grad():
            _ = model(images, size=img_size, augment=False)

    torch.cuda.synchronize()
    latencies_ms: list[float] = []
    peak_mem_mb: list[float] = []
    gpu_util_pct: list[float] = []
    mem_used_mb: list[float] = []

    for _ in range(repeat_steps):
        images = [Image.open(path).convert("RGB") for path in batch_image_paths]
        before = read_gpu_snapshot()
        torch.cuda.reset_peak_memory_stats(device)

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        with torch.no_grad():
            _ = model(images, size=img_size, augment=False)
        end_event.record()
        torch.cuda.synchronize()

        after = read_gpu_snapshot()
        latency_ms = start_event.elapsed_time(end_event)
        peak_mem = torch.cuda.max_memory_allocated(device) / (1024**2)

        latencies_ms.append(float(latency_ms))
        peak_mem_mb.append(float(peak_mem))
        gpu_util_pct.append(float(after.get("gpu_util_pct") or before.get("gpu_util_pct") or 0.0))
        mem_used_mb.append(float(after.get("mem_used_mb") or before.get("mem_used_mb") or 0.0))

    avg_latency_ms = float(np.mean(latencies_ms))
    avg_peak_mem_mb = float(np.mean(peak_mem_mb))
    avg_gpu_util = float(np.mean(gpu_util_pct)) if gpu_util_pct else 0.0
    avg_mem_used_mb = float(np.mean(mem_used_mb)) if mem_used_mb else 0.0
    throughput = len(batch_image_paths) / (avg_latency_ms / 1000.0)

    return {
        "batch_size": len(batch_image_paths),
        "avg_latency_ms": avg_latency_ms,
        "latency_per_sample_ms": avg_latency_ms / len(batch_image_paths),
        "throughput_samples_per_sec": throughput,
        "avg_peak_gpu_mem_mb": avg_peak_mem_mb,
        "avg_gpu_util_pct": avg_gpu_util,
        "avg_mem_used_mb": avg_mem_used_mb,
    }


def benchmark_batch_sizes(model_name: str = "yolov5s", sample_image_paths: list[str] | None = None, batch_sizes: list[int] | None = None, img_size: int = 640, warmup_steps: int = 2, repeat_steps: int = 5) -> list[dict]:
    if sample_image_paths is None:
        sample_image_paths = create_sample_images(num_images=2)
    if batch_sizes is None:
        batch_sizes = DEFAULT_BATCH_SIZES

    weights_path = download_yolov5_weights(model_name=model_name, output_dir="weights")
    model = load_yolov5(weights_path=weights_path, model_name=model_name)

    results: list[dict] = []
    for batch_size in batch_sizes:
        batch_paths = repeat_image_paths(sample_image_paths, batch_size)
        try:
            result = benchmark_batch_size(model, batch_paths, img_size=img_size, warmup_steps=warmup_steps, repeat_steps=repeat_steps)
            results.append(result)
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                results.append({
                    "batch_size": batch_size,
                    "avg_latency_ms": None,
                    "latency_per_sample_ms": None,
                    "throughput_samples_per_sec": None,
                    "avg_peak_gpu_mem_mb": None,
                    "avg_gpu_util_pct": None,
                    "avg_mem_used_mb": None,
                    "status": "OOM",
                })
                continue
            raise

    return results


def save_results_csv(results: list[dict], output_path: str | Path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as csv_file:
        field_names = [
            "batch_size",
            "avg_latency_ms",
            "latency_per_sample_ms",
            "throughput_samples_per_sec",
            "avg_peak_gpu_mem_mb",
            "avg_gpu_util_pct",
            "avg_mem_used_mb",
            "status",
        ]
        writer = csv.DictWriter(csv_file, fieldnames=field_names)
        writer.writeheader()
        for row in results:
            writer.writerow({key: row.get(key) for key in field_names})


def main():
    parser = argparse.ArgumentParser(description="Benchmark YOLOv5 batch inference latency, throughput, GPU memory, and utilization.")
    parser.add_argument("--model-name", default="yolov5s", help="YOLOv5 model name, e.g. yolov5n, yolov5s, yolov5m")
    parser.add_argument("--sample-dir", default="sample_images", help="Directory containing demo images")
    parser.add_argument("--num-samples", type=int, default=2, help="Number of base sample images to generate")
    parser.add_argument("--batch-sizes", nargs="*", type=int, default=DEFAULT_BATCH_SIZES, help="Batch sizes to benchmark")
    parser.add_argument("--img-size", type=int, default=640, help="Input image size sent to YOLOv5")
    parser.add_argument("--warmup-steps", type=int, default=2, help="Warmup passes before timing")
    parser.add_argument("--repeat-steps", type=int, default=5, help="Timed inferences per batch size")
    parser.add_argument("--output-csv", default="results/yolov5_batch_benchmark.csv", help="CSV file to save benchmark results")
    args = parser.parse_args()

    sample_image_paths = create_sample_images(output_dir=args.sample_dir, num_images=args.num_samples)

    results = benchmark_batch_sizes(
        model_name=args.model_name,
        sample_image_paths=sample_image_paths,
        batch_sizes=args.batch_sizes,
        img_size=args.img_size,
        warmup_steps=args.warmup_steps,
        repeat_steps=args.repeat_steps,
    )

    for row in results:
        print(row)

    save_results_csv(results, args.output_csv)
    print(f"Saved results to {args.output_csv}")


if __name__ == "__main__":
    main()
