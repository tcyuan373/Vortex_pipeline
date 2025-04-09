import logging
logging.disable(logging.CRITICAL)
import argparse
import os
import csv
import pickle
import torch
from nemo.collections.tts.models import FastPitchModel, HifiGanModel
from torch.nn.utils.rnn import pad_sequence

# === Global config ===
TOTAL_RUNS = 1000
TEXT_FILE_PATH = "/mydata/msmarco/msmarco_3_clusters/doc_list.pkl"  # List[str]
FASTPITCH_NAME = "nvidia/tts_en_fastpitch"
HIFIGAN_NAME = "nvidia/tts_hifigan"

def synthesize(batch_texts, run_times, fastpitch, hifigan, device):
    with torch.no_grad():
        token_list = [fastpitch.parse(text).squeeze(0) for text in batch_texts]
        tokens = pad_sequence(token_list, batch_first=True).to(device)

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        spectrograms = fastpitch.generate_spectrogram(tokens=tokens)
        audios = hifigan.convert_spectrogram_to_audio(spec=spectrograms)
        end_event.record()
        torch.cuda.synchronize()

        latency_ns = start_event.elapsed_time(end_event) * 1e6  # ms -> ns
        run_times.append(latency_ns)
    
    return audios


def main(output_dir, pid, bsize):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fastpitch = FastPitchModel.from_pretrained(FASTPITCH_NAME).to(device).eval()
    hifigan = HifiGanModel.from_pretrained(model_name=HIFIGAN_NAME).to(device).eval()

    with open(TEXT_FILE_PATH, "rb") as f:
        texts = pickle.load(f)  # should be a list of strings

    iterator = iter(texts)
    run_times = []

    for _ in range(TOTAL_RUNS):
        batch = [next(iterator) for _ in range(bsize)]
        _ = synthesize(batch, run_times, fastpitch, hifigan, device)

    throughput = (bsize * len(run_times)) / (sum(run_times) / 1e9)  # samples/sec
    avg_latency = int(sum(run_times) / len(run_times))  # ns

    print(f"Batch size {bsize}, throughput: {throughput:.2f} audios/sec")
    print(f"Average latency per batch: {avg_latency} ns")

    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(
        output_dir, f"tts_batch{bsize}_runtime{pid}_tp{throughput:.2f}.csv"
    )

    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(run_times)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark FastPitch + HiFi-GAN TTS")
    parser.add_argument("-p", "--output_dir", type=str, required=True, help="Directory to store runtime CSV")
    parser.add_argument("-id", "--pid", type=str, required=True, help="Identifier for process/GPU setup")
    parser.add_argument("-b", "--bsize", type=int, required=True, help="Batch size for TTS")
    args = parser.parse_args()

    main(args.output_dir, args.pid, args.bsize)
