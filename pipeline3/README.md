# YOLOv5 batch benchmark pipeline

This pipeline creates a minimal end-to-end benchmark flow for YOLOv5 on GPU:

1. Download 200 manufacturing defect-inspection images from the VisA dataset (CC BY 4.0).
2. Cycle through them to build each batch.
3. Download the YOLOv5 model weights.
4. Run 16 timed trials per batch size and record latency, throughput, GPU memory, and utilization.
5. Save per-trial times and per-batch-size averages and standard deviations to CSV.

## Files

- `generate_sample_images.py`: streams the [VisA](https://github.com/amazon-science/spot-diff) archive from S3, keeps 200 images of one category (default: `pcb1` circuit boards, all 100 defective ones plus 100 defect-free ones, alternating), caches them in `sample_images/` as `visa_*`, and writes `ATTRIBUTION_VisA.md`.
- `download_yolov5.py`: clones the YOLOv5 repo and downloads the official weights.
- `yolov5_batch_latency.py`: benchmarks latency, throughput, GPU memory, and utilization for selected batch sizes.
- `main.py`: runs the full workflow end to end.

## Default batch sizes tested

[1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 32, 48, 64]

## Run

```bash
cd /Users/alicia/Desktop/temp/Vortex_pipeline
python pipeline3/main.py --model-name yolov5s

# other VisA categories / labels
python pipeline3/main.py --visa-category pcb3 --visa-label normal
```

VisA categories: candle, capsules, cashew, chewinggum, fryum, macaroni1, macaroni2, pcb1, pcb2, pcb3, pcb4, pipe_fryum. Labels: `mixed` (default: all defective images, the rest defect-free, interleaved), `anomaly` (defective only; each category has just 100), `normal`, or `any`.

The first run streams the VisA tar until it has enough images, which can mean reading a large part of the archive. Later runs reuse the cached images. If you already have `VisA_20220922.tar` locally, point to it with `export VISA_TAR=/path/to/VisA_20220922.tar` or `--tar-path`.

Or run each stage separately:

```bash
python pipeline3/generate_sample_images.py
python pipeline3/download_yolov5.py
python pipeline3/yolov5_batch_latency.py --model-name yolov5s --batch-sizes 1 2 4 8 16 32 64 --trials 16
```

## How the benchmark runs

For every batch size, the benchmark runs 2 untimed warmup batches and then 16 timed trials. Batch *k* (warmups first) uses images *k*·b to (*k*+1)·b − 1 of the downloaded set, wrapping back to the start after the last image, so the trials walk through all 200 images instead of reusing the same few. The cycle restarts at image 0 for each batch size, so every configuration sees the same image sequence. Images are decoded before the timer starts; the model's preprocessing and inference are timed.

Two CSVs are written:

- `results/yolov5_batch_benchmark.csv`: one row per batch size, with `num_trials`, `avg_latency_ms`, `std_latency_ms` (sample standard deviation across the trials), per-image latency, throughput, peak GPU memory, and average GPU utilization.
- `results/yolov5_batch_benchmark_trials.csv`: one row per trial, with its `latency_ms`, the index of its first image, peak GPU memory, and GPU utilization.

Change the number of trials or warmups with `--trials N` and `--warmup-steps N`. Both CSVs are overwritten on every run; pass `--output-csv` to keep earlier results.

## Notes

- The benchmark automatically skips impossible batch sizes if the GPU runs out of memory.
- Delete `sample_images/visa_*` to pick a new set of images.
- The YOLOv5 weights are COCO-pretrained, so they measure inference cost on these images but do not detect defects without fine-tuning.
- The script uses `nvidia-smi` and the CUDA runtime to capture approximate GPU memory and utilization values.
