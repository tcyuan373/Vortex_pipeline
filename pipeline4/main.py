import argparse
from pathlib import Path

from benchmark_models import DEFAULT_BATCH_SIZES, benchmark_all, save_results_csv
from download_models import download_dino_model, download_fasterrcnn_model, download_ocr_reader
from generate_sample_images import create_sample_images


def main():
    parser = argparse.ArgumentParser(description="Benchmark OCR, Faster R-CNN, and DINO across multiple batch sizes on GPU.")
    parser.add_argument("--sample-dir", default="sample_images")
    parser.add_argument("--num-samples", type=int, default=2)
    parser.add_argument("--batch-sizes", nargs="*", type=int, default=DEFAULT_BATCH_SIZES)
    parser.add_argument("--output-csv", default="results/model_latency_benchmark.csv")
    args = parser.parse_args()

    sample_paths = create_sample_images(output_dir=args.sample_dir, num_images=args.num_samples)
    print(f"Using sample images: {sample_paths}")

    print("Downloading OCR model...")
    download_ocr_reader()

    print("Downloading Faster R-CNN model...")
    download_fasterrcnn_model()

    print("Downloading DINO model...")
    download_dino_model()

    results = benchmark_all(batch_sizes=args.batch_sizes, sample_dir=args.sample_dir, num_samples=args.num_samples)
    for row in results:
        print(row)

    save_results_csv(results, args.output_csv)
    print(f"Saved benchmark outputs to {args.output_csv}")


if __name__ == "__main__":
    main()
