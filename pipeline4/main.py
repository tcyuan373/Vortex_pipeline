import argparse

from benchmark_models import (
    DEFAULT_BATCH_SIZES,
    DEFAULT_TRIALS,
    DEFAULT_WARMUP_STEPS,
    benchmark_all,
    save_results_csv,
)
from download_models import download_dino_model, download_fasterrcnn_model, download_ocr_reader
from generate_sample_images import create_sample_images


def main():
    parser = argparse.ArgumentParser(description="Benchmark OCR, Faster R-CNN, and DINO across multiple batch sizes on GPU.")
    parser.add_argument("--sample-dir", default="sample_images")
    parser.add_argument("--num-samples", type=int, default=200, help="Distinct images to download and cycle through")
    parser.add_argument("--batch-sizes", nargs="*", type=int, default=DEFAULT_BATCH_SIZES)
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS, help="Timed trials per batch size")
    parser.add_argument("--warmup-steps", type=int, default=DEFAULT_WARMUP_STEPS, help="Untimed batches before the trials")
    parser.add_argument("--output-csv", default="results/model_latency_benchmark.csv",
                        help="Summary CSV; per-trial times go to the same name with _trials")
    args = parser.parse_args()

    sample_paths = create_sample_images(output_dir=args.sample_dir, num_images=args.num_samples)
    print(f"Using {len(sample_paths)} images from {args.sample_dir}")

    print("Downloading OCR model...")
    download_ocr_reader()

    print("Downloading Faster R-CNN model...")
    download_fasterrcnn_model()

    print("Downloading DINO model...")
    download_dino_model()

    summaries, trials = benchmark_all(
        batch_sizes=args.batch_sizes,
        sample_dir=args.sample_dir,
        num_samples=args.num_samples,
        warmup_steps=args.warmup_steps,
        num_trials=args.trials,
    )

    trials_path = save_results_csv(summaries, args.output_csv, trials)
    print(f"Saved summary to {args.output_csv} and per-trial times to {trials_path}")


if __name__ == "__main__":
    main()
