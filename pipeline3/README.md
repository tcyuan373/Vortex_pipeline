# YOLOv5 batch benchmark pipeline

This pipeline creates a minimal end-to-end benchmark flow for YOLOv5 on GPU:

1. Download a few manufacturing defect-inspection images from the VisA dataset (CC BY 4.0).
2. Duplicate it as needed to simulate batching.
3. Download the YOLOv5 model weights.
4. Benchmark runtime latency, throughput, GPU memory, and utilization across batch sizes.
5. Save the results to CSV for comparison.

## Files

- `generate_sample_images.py`: streams the [VisA](https://github.com/amazon-science/spot-diff) archive from S3, keeps the first N images of one category and label (default: defective `pcb1` circuit boards), caches them in `sample_images/` as `visa_*`, and writes `ATTRIBUTION_VisA.md`.
- `download_yolov5.py`: clones the YOLOv5 repo and downloads the official weights.
- `yolov5_batch_latency.py`: benchmarks latency, throughput, GPU memory, and utilization for selected batch sizes.
- `main.py`: runs the full workflow end to end.

## Default batch sizes tested

[1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 28, 30, 32, 64]

## Run

```bash
cd /Users/alicia/Desktop/temp/Vortex_pipeline
python pipeline3/main.py --model-name yolov5s

# other VisA categories / labels
python pipeline3/main.py --visa-category pcb3 --visa-label normal
```

VisA categories: candle, capsules, cashew, chewinggum, fryum, macaroni1, macaroni2, pcb1, pcb2, pcb3, pcb4, pipe_fryum. Labels: `anomaly` (defective), `normal`, or `any`.

The first run streams the VisA tar until it has enough images, which can mean reading a large part of the archive. Later runs reuse the cached images. If you already have `VisA_20220922.tar` locally, point to it with `export VISA_TAR=/path/to/VisA_20220922.tar` or `--tar-path`.

Or run each stage separately:

```bash
python pipeline3/generate_sample_images.py
python pipeline3/download_yolov5.py
python pipeline3/yolov5_batch_latency.py --model-name yolov5s --batch-sizes 1 2 4 8 10 12 14 16 20 28 30 32 64
```

## Notes

- The benchmark automatically skips impossible batch sizes if the GPU runs out of memory.
- The pipeline reuses the cached VisA images to fill each batch. Delete `sample_images/visa_*` to pick a new set.
- The YOLOv5 weights are COCO-pretrained, so they measure inference cost on these images but do not detect defects without fine-tuning.
- The script uses `nvidia-smi` and the CUDA runtime to capture approximate GPU memory and utilization values.
