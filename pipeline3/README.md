# YOLOv5 batch benchmark pipeline

This pipeline creates a minimal end-to-end benchmark flow for YOLOv5 on GPU:

1. Download a small real image from a public URL.
2. Duplicate it as needed to simulate batching.
3. Download the YOLOv5 model weights.
4. Benchmark runtime latency, throughput, GPU memory, and utilization across batch sizes.
5. Save the results to CSV for comparison.

## Files

- `generate_sample_images.py`: downloads one or more simple real-world sample images from public URLs and reuses them for batching.
- `download_yolov5.py`: clones the YOLOv5 repo and downloads the official weights.
- `yolov5_batch_latency.py`: benchmarks latency, throughput, GPU memory, and utilization for selected batch sizes.
- `main.py`: runs the full workflow end to end.

## Default batch sizes tested

[1, 2, 4, 8, 10, 12, 14, 16, 20, 28, 30, 32, 64]

## Run

```bash
cd /Users/alicia/Desktop/temp/Vortex_pipeline
python pipeline3/main.py --model-name yolov5s
```

Or run each stage separately:

```bash
python pipeline3/generate_sample_images.py
python pipeline3/download_yolov5.py
python pipeline3/yolov5_batch_latency.py --model-name yolov5s --batch-sizes 1 2 4 8 10 12 14 16 20 28 30 32 64
```

## Notes

- The benchmark automatically skips impossible batch sizes if the GPU runs out of memory.
- The pipeline downloads a real image instead of generating a synthetic one, then reuses it to emulate batch inference.
- The script uses `nvidia-smi` and the CUDA runtime to capture approximate GPU memory and utilization values.

