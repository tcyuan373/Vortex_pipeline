#!/usr/bin/env python3
import argparse
import time
import torch
import csv
import pickle
import numpy as np
import os
import sys

sys.path.append("./SenseVoice")

from funasr.utils.postprocess_utils import rich_transcription_postprocess
from funasr.utils.load_utils import extract_fbank, load_audio_text_image_video
from model import SenseVoiceSmall

# === Global configuration ===
PKL_PATH = "/mydata/msmarco/queries_audio5000.pkl"
MODEL_DIR = "iic/SenseVoiceSmall"
TOTAL_RUNS = 1000


def run_benchmark(list_np_waveform, model, frontend, kwargs, bsize):
    list_of_runtimes = []
    batch_audio_sample = []
    j = 0

    for _ in range(bsize * TOTAL_RUNS):
        curr_array = list_np_waveform[j % len(list_np_waveform)][-1]
        j += 1

        if len(curr_array) <= 200000:
            batch_audio_sample.append(curr_array)
            if len(batch_audio_sample) == bsize:
                speech, speech_lengths = extract_fbank(
                    batch_audio_sample,
                    data_type=kwargs.get("data_type", "sound"),
                    frontend=frontend,
                )
                start_time = time.perf_counter_ns()
                res = model.inference(
                    data_in=speech,
                    data_lengths=speech_lengths,
                    language="en",
                    use_itn=False,
                    ban_emo_unk=True,
                    **kwargs,
                )
                torch.cuda.synchronize()  # Ensure GPU ops are completed
                end_time = time.perf_counter_ns()
                list_of_runtimes.append(end_time - start_time)
                batch_audio_sample = []

        else:
            continue
            # speech, speech_lengths = extract_fbank(
            #     torch.from_numpy(curr_array).unsqueeze(0),
            #     data_type=kwargs.get("data_type", "sound"),
            #     frontend=frontend,
            # )
            # _ = model.inference(
            #     data_in=speech,
            #     data_lengths=speech_lengths,
            #     language="en",
            #     use_itn=False,
            #     ban_emo_unk=True,
            #     **kwargs,
            # )
            # # Not timed — excluded from benchmark
    return list_of_runtimes


def main(output_dir, pid, bsize):
    with open(PKL_PATH, "rb") as f:
        list_np_waveform = pickle.load(f)

    print(f"Loaded {len(list_np_waveform)} queries and audio samples.")

    model, kwargs = SenseVoiceSmall.from_pretrained(model=MODEL_DIR, device="cuda:0")
    model.eval()

    kwargs["data_type"] = "fbank"
    kwargs["sound"] = "fbank"
    frontend = kwargs["frontend"]

    run_times = run_benchmark(list_np_waveform, model, frontend, kwargs, bsize)

    throughput = (bsize * len(run_times)) / (sum(run_times) / 1e9)
    avg_latency = int(sum(run_times) / len(run_times))

    print(f"Batch size {bsize}, throughput: {throughput:.2f} queries/sec")
    print(f"Average latency per batch: {avg_latency} ns")
    print(f"Recorded {len(run_times)} timed batches.")

    os.makedirs(output_dir, exist_ok=True)
    runtimes_file = os.path.join(
        output_dir,
        f"audio_bsize{bsize}_runtime{pid}_tp{throughput:.2f}.csv"
    )

    with open(runtimes_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(run_times)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark SenseVoiceSmall Audio Recognition")
    parser.add_argument('-p', '--output_dir', type=str, required=True, help='Directory to store runtime CSV')
    parser.add_argument('-id', '--pid', type=str, required=True, help='Identifier for process/GPU config')
    parser.add_argument('-b', '--bsize', type=int, required=True, help='Batch size for audio inference')
    args = parser.parse_args()

    main(args.output_dir, args.pid, args.bsize)