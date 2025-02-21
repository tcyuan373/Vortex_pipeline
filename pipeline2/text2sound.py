# Following pip packages need to be installed:
# !pip install git+https://github.com/huggingface/transformers sentencepiece datasets

## Using speech t5
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import csv
import torch
import soundfile as sf
import numpy as np

import textwrap


def split_text(text, chunk_size=600):
     return textwrap.wrap(text, width=chunk_size)  # Splits into smaller chunks


def text2soundfunc(text, run_times):
     device = "cuda" if torch.cuda.is_available() else "cpu"  # Automatically select GPU if available

     # Load models and move them to GPU if available
     processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
     model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").to(device)
     vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(device)

     # Load speaker embedding
     embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
     speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0).to(device)

     # Split text into smaller segments
     text_chunks = split_text(text)
     speech_outputs = []

     for idx, chunk in enumerate(text_chunks):
          inputs = processor(text=chunk, return_tensors="pt").to(device)  # Move inputs to GPU

          model_start_event = torch.cuda.Event(enable_timing=True)
          model_end_event = torch.cuda.Event(enable_timing=True)

          # time before running model
          model_start_event.record()
          speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
          # time after running model
          model_end_event.record()
          torch.cuda.synchronize()
          run_times.append((model_start_event.elapsed_time(model_end_event)) * 1e6)

          speech_outputs.append(speech.cpu().numpy())  # Move back to CPU before converting to NumPy
          print(f"Chunk {idx+1}/{len(text_chunks)} processed.")
     
     # Concatenate all speech chunks
     full_speech = np.concatenate(speech_outputs, axis=0)
     print(f"Size of speech_outputs: {full_speech.shape}")

     # Save as a WAV file
     sf.write("speech.wav", full_speech, samplerate=16000)

     return full_speech

response = " Based on the provided documents, here are some average salary ranges for bartenders:\
\
* Average annual salary: $19,050 or $9.16 per hour, including tips (according to the Bureau of Labor Statistics)\
* Average hourly wage: $10.36 (according to the Bureau of Labor Statistics)\
* Average yearly take-home: $21,550 (according to the Bureau of Labor Statistics)\
* Tips make up half or more of bartender's salaries, with average tips ranging from $12.00 to $18.00 per hour\
* In high-volume resorts or establishments, bartenders can earn $50,000 to $75,000 per year between hourly wages and tips\
* Median salary rates for bartenders in restaurants: $73,000 (according to Indeed.com 2010 results)\
* Average salaries for bartenders in different industries:\
        + Hotels and hotel restaurants: $26,180 per year\
        + Full-service restaurants: $22,130 per year\
        + Bars: $20,230 per year\
        + Civic and social organizations: $18,970 per year\
        + Median earnings: $9.06 per hour and $18,850 per year\
\
Keep in mind that these figures are averages and can vary widely"

run_times = []
# response = " Based on the provided documents, here are some average salary ranges for bartenders:"
for i in range(10):
     text2soundfunc(response, run_times)

runtimes_file = 'text2sound_runtime.csv'

with open(runtimes_file, mode='w', newline='') as file:
     writer = csv.writer(file)
     writer.writerow(run_times)
