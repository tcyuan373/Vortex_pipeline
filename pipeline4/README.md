# Model benchmark pipeline

This pipeline benchmarks three model families on GPU using the same pattern as the earlier pipeline folders:

1. Download 200 dairy-cow udder photos from the Roboflow [cow_udder](https://universe.roboflow.com/butter-vfw4d/cow_udder) dataset (509 images, CC BY 4.0) and resize them to 1024x768.
2. Download the model weights for OCR, Faster R-CNN, and DINO.
3. Cycle through the images to build each batch, for every batch size.
4. Run 16 timed trials per batch size and measure latency, throughput, GPU memory usage, and utilization.
5. Save per-trial times and per-batch-size averages and standard deviations to CSV.

## Files

- `generate_sample_images.py`: exports the cow_udder dataset from Roboflow, picks 200 distinct photos (skipping Roboflow's augmented copies), resizes them to `sample_images/cowudder_*.jpg`, keeps the originals in `sample_images/.cowudder_originals/`, and writes `ATTRIBUTION_cow_udder.md`.
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

## How the benchmark runs

For every batch size, the benchmark runs 2 untimed warmup batches and then 16 timed trials. Batch *k* (warmups first) uses images *k*·b to (*k*+1)·b − 1 of the downloaded set, wrapping back to the start after the last image, so the trials walk through all 200 images instead of reusing the same few. The cycle restarts at image 0 for each batch size and model, so every configuration sees the same image sequence. Images are decoded before the timer starts; the model's preprocessing and inference are timed.

Two CSVs are written:

- `results/model_latency_benchmark.csv`: one row per model and batch size, with `num_trials`, `avg_latency_ms`, `std_latency_ms` (sample standard deviation across the trials), per-image latency, throughput, peak GPU memory, and average GPU utilization.
- `results/model_latency_benchmark_trials.csv`: one row per trial, with its `latency_ms`, the index of its first image, peak GPU memory, and GPU utilization.

Change the number of trials or warmups with `--trials N` and `--warmup-steps N`. Both CSVs are overwritten on every run; pass `--output-csv` to keep earlier results.

EasyOCR reads one image at a time, so an OCR "batch" of size b is b consecutive `readtext` calls.

## Notes

- Every image is resized to 1024x768 (scaled to fit, then padded with gray bars, so nothing is cropped or stretched). Uniform sizes stop Faster R-CNN from padding whole batches up to their largest image. `--size` re-resizes from the kept originals without downloading.
- Delete `sample_images/cowudder_*`, `sample_images/.cowudder_originals/` and `sample_images/cow_udder_sources.json` to pick a new set.
- OCR uses `easyocr`.
- Faster R-CNN uses the official pretrained torchvision implementation.
- DINO uses the Hugging Face `facebook/dinov2-small` model.
- If a batch size runs out of GPU memory, it is recorded with status `OOM` and the benchmark moves on, so the CSVs are still written.
