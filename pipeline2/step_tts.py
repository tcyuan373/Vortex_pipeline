import argparse
import os
import time
import torch
import csv
import pickle
from torch.nn.utils.rnn import pad_sequence
from nemo.collections.tts.models import FastPitchModel, HifiGanModel

# === Global configuration ===
TOTAL_RUNS = 1000
FILE_PATH = "/mydata/msmarco/msmarco_3_clusters/doc_list.pkl"

def run_tts(batch_texts, run_times, spec_generator, vocoder, device):
    with torch.no_grad():
        token_tensors = [spec_generator.parse(text).squeeze() for text in batch_texts]
        padded_tokens = pad_sequence(token_tensors, batch_first=True).to(device)

        model_start_event = torch.cuda.Event(enable_timing=True)
        model_end_event = torch.cuda.Event(enable_timing=True)

        model_start_event.record()
        spectrograms = spec_generator.generate_spectrogram(tokens=padded_tokens)
        audios = [vocoder.convert_spectrogram_to_audio(spec=spec.unsqueeze(0)).squeeze().detach().cpu().numpy()
                  for spec in spectrograms]
        model_end_event.record()

        torch.cuda.synchronize()
        run_times.append(model_start_event.elapsed_time(model_end_event) * 1e6)  # microseconds to nanoseconds

    return list(zip(batch_texts, audios))

def main(output_dir, pid, bsize):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading FastPitch...")
    spec_generator = FastPitchModel.from_pretrained("nvidia/tts_en_fastpitch").to(device).eval()
    print("Loading HiFi-GAN vocoder...")
    vocoder = HifiGanModel.from_pretrained("nvidia/tts_hifigan").to(device).eval()

    with open(FILE_PATH, "rb") as f:
        data = pickle.load(f)

    iterator = iter(data)
    run_times = []
    results = []

    for _ in range(TOTAL_RUNS):
        batch_texts = [next(iterator) for _ in range(bsize)]
        results.extend(run_tts(batch_texts, run_times, spec_generator, vocoder, device))

    throughput = (bsize * len(run_times)) / (sum(run_times) / 1e9)
    avg_latency = int(sum(run_times) / len(run_times))

    print(f"Batch size {bsize}, throughput: {throughput:.2f} queries/sec")
    print(f"Average latency per batch: {avg_latency} ns")

    # Save audio to pkl
    output_pkl = os.path.join(output_dir, f"tts_fastpitch_output_{pid}.pkl")
    with open(output_pkl, "wb") as f:
        pickle.dump(results, f)

    # Save runtime
    runtimes_file = os.path.join(
        output_dir, f"tts_fastpitch_bsize{bsize}_runtime{pid}_tp{throughput:.2f}.csv"
    )
    with open(runtimes_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(run_times)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark FastPitch + HiFi-GAN TTS")
    parser.add_argument('-p', '--output_dir', type=str, required=True, help="Directory to store output files")
    parser.add_argument('-id', '--pid', type=str, required=True, help="Identifier for process/GPU setup")
    parser.add_argument('-b', '--bsize', type=int, required=True, help="Batch size for synthesis")
    args = parser.parse_args()

    main(args.output_dir, args.pid, args.bsize)