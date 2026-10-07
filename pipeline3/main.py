import argparse
from pathlib import Path

from download_yolov5 import download_yolov5_weights, ensure_yolov5_repo
from generate_sample_images import create_sample_images
from yolov5_batch_latency import DEFAULT_BATCH_SIZES, benchmark_batch_sizes, save_results_csv


def main():
    parser = argparse.ArgumentParser(description="Orchestrate YOLOv5 sample generation, model download, and latency benchmarking.")
    parser.add_argument("--model-name", default="yolov5s")
    parser.add_argument("--sample-dir", default="sample_images")
    parser.add_argument("--num-samples", type=int, default=2)
    parser.add_argument("--batch-sizes", nargs="*", type=int, default=DEFAULT_BATCH_SIZES)
    parser.add_argument("--output-csv", default="results/yolov5_batch_benchmark.csv")
    args = parser.parse_args()

    ensure_yolov5_repo()
    sample_image_paths = create_sample_images(output_dir=args.sample_dir, num_images=args.num_samples)
    weights_path = download_yolov5_weights(model_name=args.model_name, output_dir="weights")
    print(f"Using weights: {weights_path}")

    results = benchmark_batch_sizes(
        model_name=args.model_name,
        sample_image_paths=sample_image_paths,
        batch_sizes=args.batch_sizes,
        img_size=640,
        warmup_steps=2,
        repeat_steps=5,
    )

    save_results_csv(results, args.output_csv)
    print(f"Benchmark results saved to {args.output_csv}")


if __name__ == "__main__":
    main()
