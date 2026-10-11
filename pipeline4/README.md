# Model benchmark pipeline

This pipeline benchmarks three model families on GPU using the same pattern as the earlier pipeline folders:

1. Download dairy-cow udder photos from the Roboflow [cow_udder](https://universe.roboflow.com/butter-vfw4d/cow_udder) dataset (509 images, CC BY 4.0) and resize them to 1024x768.
2. Download the model weights for OCR, Faster R-CNN, and DINO.
3. Repeat the same inputs across batch sizes from 1 to 64.
4. Measure latency, throughput, GPU memory usage, and utilization.
5. Save the results to CSV.

## Files

- `generate_sample_images.py`: exports the cow_udder dataset from Roboflow, picks 64 distinct photos (skipping Roboflow's augmented copies), resizes them to `sample_images/cowudder_*.jpg`, keeps the originals in `sample_images/.cowudder_originals/`, and writes `ATTRIBUTION_cow_udder.md`.
- `download_models.py`: downloads the model weights for OCR, Faster R-CNN, and DINO.
- `benchmark_models.py`: runs the runtime benchmark across the target batch sizes.
- `main.py`: runs the whole workflow end to end.

## Default batch sizes

[1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24]

## Run

Roboflow downloads need a free account. Copy your API key from roboflow.com (Settings -> API Keys):

```bash
cd /Users/alicia/Desktop/temp/Vortex_pipeline
export ROBOFLOW_API_KEY=your_key_here
python3 pipeline4/main.py
```

The script tries the newest dataset versions and several export formats until one downloads. If none do, download the dataset in your browser (dataset page -> Download Dataset -> any format -> zip) and point to it; no key is needed:

```bash
ROBOFLOW_ZIP=/path/to/export.zip python3 pipeline4/main.py
```

The key or zip is only needed for the first run; later runs reuse the cached images. To fetch the images on their own, pin a version, or change the size:

```bash
python3 pipeline4/generate_sample_images.py --version 2 --size 800x600
```

## Notes

- Every image is resized to 1024x768 (scaled to fit, then padded with gray bars, so nothing is cropped or stretched). Uniform sizes stop Faster R-CNN from padding whole batches up to their largest image. `--size` re-resizes from the kept originals without downloading.
- Delete `sample_images/cowudder_*`, `sample_images/.cowudder_originals/` and `sample_images/cow_udder_sources.json` to pick a new set.
- OCR uses `easyocr`.
- Faster R-CNN uses the official pretrained torchvision implementation.
- DINO uses the Hugging Face `facebook/dinov2-small` model.
- The benchmark automatically skips impossible batch sizes if the GPU runs out of memory.
