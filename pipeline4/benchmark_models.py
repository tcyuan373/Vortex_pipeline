import csv
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from download_models import download_dino_model, download_fasterrcnn_model, download_ocr_reader
from generate_sample_images import create_sample_images, cycle_batch

DEFAULT_BATCH_SIZES = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24]
DEFAULT_TRIALS = 16
DEFAULT_WARMUP_STEPS = 2


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


def _load_images(paths: list[str]) -> list[Image.Image]:
    images = []
    for path in paths:
        with Image.open(path) as img:
            images.append(img.convert("RGB"))
    return images


def _summarize(model_name: str, batch_size: int, trials: list[dict]) -> dict:
    latencies = np.array([t["latency_ms"] for t in trials], dtype=float)
    avg_latency_ms = float(latencies.mean())
    # Sample standard deviation (n - 1 in the denominator) across trials.
    std_latency_ms = float(latencies.std(ddof=1)) if len(latencies) > 1 else 0.0
    return {
        "model": model_name,
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


def benchmark_batch(
    model_name: str,
    run_batch_fn,
    image_paths: list[str],
    batch_size: int,
    warmup_steps: int = DEFAULT_WARMUP_STEPS,
    num_trials: int = DEFAULT_TRIALS,
) -> tuple[dict, list[dict]]:
    """Time `num_trials` batches of `batch_size`, each on the next images in the cycle.

    Batch k (counting warmup batches first) uses images [k*batch_size, (k+1)*batch_size)
    modulo the number of images, so the trials walk through the whole image set. The
    cycle restarts for every batch size and model, so all of them see the same image
    sequence. Images are decoded before the timer starts; `run_batch_fn` (the model's
    preprocessing and inference) is what gets timed.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark. No GPU was detected.")
    device = torch.device("cuda:0")

    for k in range(warmup_steps):
        run_batch_fn(_load_images(cycle_batch(image_paths, batch_size, k)[0]))
    torch.cuda.synchronize()

    trials: list[dict] = []
    for t in range(num_trials):
        paths, first_index = cycle_batch(image_paths, batch_size, warmup_steps + t)
        images = _load_images(paths)
        before = read_gpu_snapshot()
        torch.cuda.reset_peak_memory_stats(device)

        torch.cuda.synchronize()
        start = time.perf_counter()
        run_batch_fn(images)
        torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - start) * 1000.0

        after = read_gpu_snapshot()
        trials.append({
            "model": model_name,
            "batch_size": batch_size,
            "trial": t + 1,
            "first_image_index": first_index,
            "latency_ms": latency_ms,
            "latency_per_sample_ms": latency_ms / batch_size,
            "peak_gpu_mem_mb": torch.cuda.max_memory_allocated(device) / (1024**2),
            "gpu_util_pct": float(after.get("gpu_util_pct") or before.get("gpu_util_pct") or 0.0),
            "mem_used_mb": float(after.get("mem_used_mb") or before.get("mem_used_mb") or 0.0),
        })

    return _summarize(model_name, batch_size, trials), trials


def _benchmark_model(
    model_name: str,
    run_batch_fn,
    image_paths: list[str],
    batch_sizes: list[int],
    warmup_steps: int,
    num_trials: int,
) -> tuple[list[dict], list[dict]]:
    """Run every batch size for one model; record batch sizes that run out of memory."""
    summaries: list[dict] = []
    all_trials: list[dict] = []
    for batch_size in batch_sizes:
        print(f"[{model_name}] batch size {batch_size}: {num_trials} trials...")
        try:
            summary, trials = benchmark_batch(
                model_name, run_batch_fn, image_paths, batch_size,
                warmup_steps=warmup_steps, num_trials=num_trials,
            )
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            torch.cuda.empty_cache()
            print(f"  out of GPU memory; skipping batch size {batch_size}")
            summaries.append({"model": model_name, "batch_size": batch_size, "num_trials": 0, "status": "OOM"})
            continue
        print(f"  {summary['avg_latency_ms']:.2f} ms +/- {summary['std_latency_ms']:.2f} ms")
        summaries.append(summary)
        all_trials.extend(trials)
    return summaries, all_trials


def benchmark_ocr(reader, image_paths, batch_sizes, warmup_steps=DEFAULT_WARMUP_STEPS, num_trials=DEFAULT_TRIALS):
    # EasyOCR reads one image at a time, so a "batch" is that many readtext calls.
    def run_batch(images):
        for img in images:
            reader.readtext(np.asarray(img), detail=0)

    return _benchmark_model("ocr", run_batch, image_paths, batch_sizes, warmup_steps, num_trials)


def benchmark_fasterrcnn(model, image_paths, batch_sizes, warmup_steps=DEFAULT_WARMUP_STEPS, num_trials=DEFAULT_TRIALS):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    weights = model.weights if hasattr(model, "weights") else None
    if weights is None:
        from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights
        weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    transform = weights.transforms()

    def run_batch(images):
        inputs = [transform(img).to(device) for img in images]
        with torch.no_grad():
            model(inputs)

    return _benchmark_model("fasterrcnn", run_batch, image_paths, batch_sizes, warmup_steps, num_trials)


def benchmark_dino(processor, model, image_paths, batch_sizes, warmup_steps=DEFAULT_WARMUP_STEPS, num_trials=DEFAULT_TRIALS):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    def run_batch(images):
        inputs = processor(images=images, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            model(**inputs)

    return _benchmark_model("dino", run_batch, image_paths, batch_sizes, warmup_steps, num_trials)


SUMMARY_FIELDS = [
    "model",
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
    "model",
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


def benchmark_all(
    batch_sizes: list[int] | None = None,
    sample_dir: str = "sample_images",
    num_samples: int = 200,
    warmup_steps: int = DEFAULT_WARMUP_STEPS,
    num_trials: int = DEFAULT_TRIALS,
) -> tuple[list[dict], list[dict]]:
    """Return (one summary row per model and batch size, one row per timed trial)."""
    if batch_sizes is None:
        batch_sizes = DEFAULT_BATCH_SIZES

    image_paths = create_sample_images(output_dir=sample_dir, num_images=num_samples)

    reader = download_ocr_reader()
    fasterrcnn_model, _ = download_fasterrcnn_model()
    dino_processor, dino_model = download_dino_model()

    summaries: list[dict] = []
    trials: list[dict] = []
    for s, t in (
        benchmark_ocr(reader, image_paths, batch_sizes, warmup_steps, num_trials),
        benchmark_fasterrcnn(fasterrcnn_model, image_paths, batch_sizes, warmup_steps, num_trials),
        benchmark_dino(dino_processor, dino_model, image_paths, batch_sizes, warmup_steps, num_trials),
    ):
        summaries.extend(s)
        trials.extend(t)
    return summaries, trials


if __name__ == "__main__":
    summaries, trials = benchmark_all()
    out_path = Path("results") / "model_latency_benchmark.csv"
    trials_path = save_results_csv(summaries, out_path, trials)
    print(f"Saved summary to {out_path} and per-trial times to {trials_path}")
