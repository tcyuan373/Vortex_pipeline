import csv
import os
import subprocess
import time
from pathlib import Path

import torch
from PIL import Image

from download_models import download_dino_model, download_fasterrcnn_model, download_ocr_reader
from generate_sample_images import create_sample_images, repeat_image_paths

DEFAULT_BATCH_SIZES = [1, 2, 4, 8, 10, 12, 14, 16, 20, 28, 30, 32, 64]


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
        values = [float(v.strip()) for v in result.stdout.strip().split(",") if v.strip()]
        if len(values) >= 3:
            return {
                "gpu_util_pct": values[0],
                "mem_used_mb": values[1],
                "mem_total_mb": values[2],
            }
    except Exception:
        pass

    return {
        "gpu_util_pct": None,
        "mem_used_mb": None,
        "mem_total_mb": None,
    }


def benchmark_batch(run_batch_fn, batch_paths: list[str], warmup_steps: int = 2, repeat_steps: int = 5) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark. No GPU was detected.")

    for _ in range(warmup_steps):
        run_batch_fn(batch_paths)

    torch.cuda.synchronize()
    latencies_ms: list[float] = []
    gpu_utils: list[float] = []
    mem_useds: list[float] = []

    for _ in range(repeat_steps):
        before = read_gpu_snapshot()
        start = time.perf_counter()
        run_batch_fn(batch_paths)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        torch.cuda.synchronize()
        after = read_gpu_snapshot()

        latencies_ms.append(elapsed_ms)
        gpu_utils.append(float(after.get("gpu_util_pct") or before.get("gpu_util_pct") or 0.0))
        mem_useds.append(float(after.get("mem_used_mb") or before.get("mem_used_mb") or 0.0))

    avg_latency_ms = float(sum(latencies_ms) / len(latencies_ms))
    avg_gpu_util = float(sum(gpu_utils) / len(gpu_utils)) if gpu_utils else 0.0
    avg_mem_used = float(sum(mem_useds) / len(mem_useds)) if mem_useds else 0.0
    throughput = len(batch_paths) / (avg_latency_ms / 1000.0)

    return {
        "batch_size": len(batch_paths),
        "avg_latency_ms": avg_latency_ms,
        "latency_per_sample_ms": avg_latency_ms / len(batch_paths),
        "throughput_samples_per_sec": throughput,
        "avg_gpu_util_pct": avg_gpu_util,
        "avg_mem_used_mb": avg_mem_used,
        "status": "ok",
    }


def benchmark_ocr(reader, sample_paths: list[str], batch_sizes: list[int], warmup_steps: int = 2, repeat_steps: int = 5) -> list[dict]:
    results: list[dict] = []
    for batch_size in batch_sizes:
        batch_paths = repeat_image_paths(sample_paths, batch_size)

        def run_batch(paths):
            for p in paths:
                reader.readtext(p, detail=0)

        result = benchmark_batch(run_batch, batch_paths, warmup_steps=warmup_steps, repeat_steps=repeat_steps)
        results.append({"model": "ocr", **result})

    return results


def benchmark_fasterrcnn(model, sample_paths: list[str], batch_sizes: list[int], warmup_steps: int = 2, repeat_steps: int = 5) -> list[dict]:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    weights = model.weights if hasattr(model, "weights") else None
    if weights is None:
        from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights
        weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT

    transform = weights.transforms()
    results: list[dict] = []

    for batch_size in batch_sizes:
        batch_paths = repeat_image_paths(sample_paths, batch_size)

        def run_batch(paths):
            images = [Image.open(p).convert("RGB") for p in paths]
            inputs = [transform(img) for img in images]
            with torch.no_grad():
                model(inputs)

        result = benchmark_batch(run_batch, batch_paths, warmup_steps=warmup_steps, repeat_steps=repeat_steps)
        results.append({"model": "fasterrcnn", **result})

    return results


def benchmark_dino(processor, model, sample_paths: list[str], batch_sizes: list[int], warmup_steps: int = 2, repeat_steps: int = 5) -> list[dict]:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    results: list[dict] = []
    for batch_size in batch_sizes:
        batch_paths = repeat_image_paths(sample_paths, batch_size)

        def run_batch(paths):
            images = [Image.open(p).convert("RGB") for p in paths]
            inputs = processor(images=images, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                model(**inputs)

        result = benchmark_batch(run_batch, batch_paths, warmup_steps=warmup_steps, repeat_steps=repeat_steps)
        results.append({"model": "dino", **result})

    return results


def save_results_csv(results: list[dict], output_path: str | Path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "batch_size",
        "avg_latency_ms",
        "latency_per_sample_ms",
        "throughput_samples_per_sec",
        "avg_gpu_util_pct",
        "avg_mem_used_mb",
        "status",
    ]

    with open(output_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({key: row.get(key) for key in fieldnames})


def benchmark_all(batch_sizes: list[int] | None = None, sample_dir: str = "sample_images", num_samples: int = 2) -> list[dict]:
    if batch_sizes is None:
        batch_sizes = DEFAULT_BATCH_SIZES

    sample_paths = create_sample_images(output_dir=sample_dir, num_images=num_samples)

    reader = download_ocr_reader()
    fasterrcnn_model, fasterrcnn_weights = download_fasterrcnn_model()
    dino_processor, dino_model = download_dino_model()

    results: list[dict] = []
    results.extend(benchmark_ocr(reader, sample_paths, batch_sizes=batch_sizes, warmup_steps=2, repeat_steps=5))
    results.extend(benchmark_fasterrcnn(fasterrcnn_model, sample_paths, batch_sizes=batch_sizes, warmup_steps=2, repeat_steps=5))
    results.extend(benchmark_dino(dino_processor, dino_model, sample_paths, batch_sizes=batch_sizes, warmup_steps=2, repeat_steps=5))

    return results


if __name__ == "__main__":
    batch_sizes = DEFAULT_BATCH_SIZES
    results = benchmark_all(batch_sizes=batch_sizes, sample_dir="sample_images", num_samples=2)
    for row in results:
        print(row)

    out_path = Path("results") / "model_latency_benchmark.csv"
    save_results_csv(results, out_path)
    print(f"Saved benchmark results to {out_path}")
