# Model benchmark pipeline

This pipeline benchmarks three model families on GPU using the same pattern as the earlier pipeline folders:

1. Download a few real sample images.
2. Download the model weights for OCR, Faster R-CNN, and DINO.
3. Repeat the same inputs across batch sizes from 1 to 64.
4. Measure latency, throughput, GPU memory usage, and utilization.
5. Save the results to CSV.

## Files

- `generate_sample_images.py`: downloads a few public sample images and reuses them for the batch benchmark.
- `download_models.py`: downloads the model weights for OCR, Faster R-CNN, and DINO.
- `benchmark_models.py`: runs the runtime benchmark across the target batch sizes.
- `main.py`: runs the whole workflow end to end.

## Default batch sizes

[1, 2, 4, 8, 10, 12, 14, 16, 20, 28, 30, 32, 64]

## Run

```bash
cd /Users/alicia/Desktop/temp/Vortex_pipeline
python3 pipeline4/main.py
```

## Notes

- OCR uses `easyocr`.
- Faster R-CNN uses the official pretrained torchvision implementation.
- DINO uses the Hugging Face `facebook/dinov2_vits14` model.
- The benchmark automatically skips impossible batch sizes if the GPU runs out of memory.
