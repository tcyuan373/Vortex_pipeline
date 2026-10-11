import argparse
from pathlib import Path

from download_yolov5 import download_yolov5_weights, ensure_yolov5_repo
from generate_sample_images import DEFAULT_CATEGORY, DEFAULT_LABEL, VISA_CATEGORIES, VISA_LABELS, create_sample_images
from yolov5_batch_latency import DEFAULT_BATCH_SIZES, benchmark_batch_sizes, save_results_csv


def main():
    parser = argparse.ArgumentParser(description="Orchestrate YOLOv5 sample generation, model download, and latency benchmarking.")
    parser.add_argument("--model-name", default="yolov5s")
    parser.add_argument("--sample-dir", default="sample_images")
    parser.add_argument("--num-samples", type=int, default=200, help="Distinct images to download and cycle through")
    parser.add_argument("--visa-category", default=DEFAULT_CATEGORY, choices=VISA_CATEGORIES)
    parser.add_argument("--visa-label", default=DEFAULT_LABEL, choices=VISA_LABELS,
                        help="mixed (all defective + defect-free fill), anomaly, normal, or any")
    parser.add_argument("--batch-sizes", nargs="*", type=int, default=DEFAULT_BATCH_SIZES)
    parser.add_argument("--trials", type=int, default=16, help="Timed trials per batch size")
    parser.add_argument("--warmup-steps", type=int, default=2, help="Untimed batches before the trials")
    parser.add_argument("--output-csv", default="results/yolov5_batch_benchmark.csv",
                        help="Summary CSV; per-trial times go to the same name with _trials")
    args = parser.parse_args()

    ensure_yolov5_repo()
    sample_image_paths = create_sample_images(
        output_dir=args.sample_dir,
        num_images=args.num_samples,
        category=args.visa_category,
        label=args.visa_label,
    )
    weights_path = download_yolov5_weights(model_name=args.model_name, output_dir="weights")
    print(f"Using weights: {weights_path}")

    print(f"Using {len(sample_image_paths)} images from {args.sample_dir}")

    summaries, trials = benchmark_batch_sizes(
        model_name=args.model_name,
        sample_image_paths=sample_image_paths,
        batch_sizes=args.batch_sizes,
        img_size=640,
        warmup_steps=args.warmup_steps,
        num_trials=args.trials,
    )

    trials_path = save_results_csv(summaries, args.output_csv, trials)
    print(f"Summary saved to {args.output_csv}; per-trial times saved to {trials_path}")


if __name__ == "__main__":
    main()
