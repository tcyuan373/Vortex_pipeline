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
from generate_sample_images import create_sample_images, cycle_batch

DEFAULT_BATCH_SIZES = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 32, 48, 64]


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


def _load_images(paths: list[str]) -> list[Image.Image]:
    images = []
    for path in paths:
        with Image.open(path) as img:
            images.append(img.convert("RGB"))
    return images


def _summarize(batch_size: int, trials: list[dict]) -> dict:
    latencies = np.array([t["latency_ms"] for t in trials], dtype=float)
    avg_latency_ms = float(latencies.mean())
    # Sample standard deviation (n - 1 in the denominator) across trials.
    std_latency_ms = float(latencies.std(ddof=1)) if len(latencies) > 1 else 0.0
    return {
        "batch_size": batch_size,
        "num_trials": len(trials),
        "avg_latency_ms": avg_latency_ms,
        "std_latency_ms": std_latency_ms,
        "latency_per_sample_ms": avg_latency_ms / batch_size,
        "throughput_samples_per_sec": batch_size / (avg_latency_ms / 1000.0),
        "avg_peak_gpu_mem_mb": float(np.mean([t["peak_gpu_mem_mb"] for t in trials])),
        "avg_gpu_util_pct": float(np.mean([t["gpu_util_pct"] for t in trials])),
        "avg_mem_used_mb": float(np.mean([t["mem_used_mb"] for t in trials])),
        "status": "ok",
    }


def benchmark_batch_size(
    model,
    image_paths: list[str],
    batch_size: int,
    img_size: int = 640,
    warmup_steps: int = 2,
    num_trials: int = 16,
) -> tuple[dict, list[dict]]:
    """Time `num_trials` batches of `batch_size`, each on the next images in the cycle.

    Batch k (counting warmup batches first) uses images [k*batch_size, (k+1)*batch_size)
    modulo the number of images, so the trials walk through the whole image set. The
    cycle restarts for every batch size, so all batch sizes see the same image sequence.
    Images are decoded before the timer starts; YOLOv5's preprocessing, inference and
    NMS are timed.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark. No GPU was detected.")

    device = torch.device("cuda:0")
    model.to(device)
    model.eval()

    for k in range(warmup_steps):
        images = _load_images(cycle_batch(image_paths, batch_size, k)[0])
        with torch.no_grad():
            _ = model(images, size=img_size, augment=False)
    torch.cuda.synchronize()

    trials: list[dict] = []
    for t in range(num_trials):
        paths, first_index = cycle_batch(image_paths, batch_size, warmup_steps + t)
        images = _load_images(paths)
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
        latency_ms = float(start_event.elapsed_time(end_event))
        trials.append({
            "batch_size": batch_size,
            "trial": t + 1,
            "first_image_index": first_index,
            "latency_ms": latency_ms,
            "latency_per_sample_ms": latency_ms / batch_size,
            "peak_gpu_mem_mb": torch.cuda.max_memory_allocated(device) / (1024**2),
            "gpu_util_pct": float(after.get("gpu_util_pct") or before.get("gpu_util_pct") or 0.0),
            "mem_used_mb": float(after.get("mem_used_mb") or before.get("mem_used_mb") or 0.0),
        })

    return _summarize(batch_size, trials), trials


def benchmark_batch_sizes(
    model_name: str = "yolov5s",
    sample_image_paths: list[str] | None = None,
    batch_sizes: list[int] | None = None,
    img_size: int = 640,
    warmup_steps: int = 2,
    num_trials: int = 16,
) -> tuple[list[dict], list[dict]]:
    """Return (one summary row per batch size, one row per timed trial)."""
    if sample_image_paths is None:
        sample_image_paths = create_sample_images(num_images=200)
    if batch_sizes is None:
        batch_sizes = DEFAULT_BATCH_SIZES

    weights_path = download_yolov5_weights(model_name=model_name, output_dir="weights")
    model = load_yolov5(weights_path=weights_path, model_name=model_name)

    summaries: list[dict] = []
    all_trials: list[dict] = []
    for batch_size in batch_sizes:
        print(f"Batch size {batch_size}: {num_trials} trials...")
        try:
            summary, trials = benchmark_batch_size(
                model, sample_image_paths, batch_size,
                img_size=img_size, warmup_steps=warmup_steps, num_trials=num_trials,
            )
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            torch.cuda.empty_cache()
            print(f"  out of GPU memory; skipping batch size {batch_size}")
            summaries.append({"batch_size": batch_size, "num_trials": 0, "status": "OOM"})
            continue
        print(f"  {summary['avg_latency_ms']:.2f} ms +/- {summary['std_latency_ms']:.2f} ms")
        summaries.append(summary)
        all_trials.extend(trials)

    return summaries, all_trials


SUMMARY_FIELDS = [
    "batch_size",
    "num_trials",
    "avg_latency_ms",
    "std_latency_ms",
    "latency_per_sample_ms",
    "throughput_samples_per_sec",
    "avg_peak_gpu_mem_mb",
    "avg_gpu_util_pct",
    "avg_mem_used_mb",
    "status",
]
TRIAL_FIELDS = [
    "batch_size",
    "trial",
    "first_image_index",
    "latency_ms",
    "latency_per_sample_ms",
    "peak_gpu_mem_mb",
    "gpu_util_pct",
    "mem_used_mb",
]


def trials_csv_path(summary_path: str | Path) -> Path:
    """results/x.csv -> results/x_trials.csv"""
    summary_path = Path(summary_path)
    return summary_path.with_name(f"{summary_path.stem}_trials{summary_path.suffix}")


def _write_csv(rows: list[dict], fields: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})


def save_results_csv(summaries: list[dict], output_path: str | Path, trials: list[dict] | None = None) -> Path | None:
    """Write the per-batch-size summary to `output_path`, and the trials next to it."""
    _write_csv(summaries, SUMMARY_FIELDS, Path(output_path))
    if trials is None:
        return None
    trials_path = trials_csv_path(output_path)
    _write_csv(trials, TRIAL_FIELDS, trials_path)
    return trials_path


def main():
    parser = argparse.ArgumentParser(description="Benchmark YOLOv5 batch inference latency, throughput, GPU memory, and utilization.")
    parser.add_argument("--model-name", default="yolov5s", help="YOLOv5 model name, e.g. yolov5n, yolov5s, yolov5m")
    parser.add_argument("--sample-dir", default="sample_images", help="Directory containing demo images")
    parser.add_argument("--num-samples", type=int, default=200, help="Number of distinct images to download and cycle through")
    parser.add_argument("--batch-sizes", nargs="*", type=int, default=DEFAULT_BATCH_SIZES, help="Batch sizes to benchmark")
    parser.add_argument("--img-size", type=int, default=640, help="Input image size sent to YOLOv5")
    parser.add_argument("--warmup-steps", type=int, default=2, help="Warmup passes before timing")
    parser.add_argument("--trials", type=int, default=16, help="Timed trials per batch size")
    parser.add_argument("--output-csv", default="results/yolov5_batch_benchmark.csv", help="CSV file to save benchmark results")
    args = parser.parse_args()

    sample_image_paths = create_sample_images(output_dir=args.sample_dir, num_images=args.num_samples)

    results = benchmark_batch_sizes(
        model_name=args.model_name,
        sample_image_paths=sample_image_paths,
        batch_sizes=args.batch_sizes,
        img_size=args.img_size,
        warmup_steps=args.warmup_steps,
        num_trials=args.trials,
    )

    summaries, trials = results
    trials_path = save_results_csv(summaries, args.output_csv, trials)
    print(f"Saved summary to {args.output_csv} and per-trial times to {trials_path}")


if __name__ == "__main__":
    main()
