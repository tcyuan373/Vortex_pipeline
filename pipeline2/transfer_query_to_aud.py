import os
import pickle
import numpy as np
from TTS.api import TTS
from contextlib import contextmanager
import sys
@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
# Step 1: Set custom model cache directory to avoid permission issues
os.environ["XDG_DATA_HOME"] = "/users/Alicia/workspace/tts_cache"  # Ensure this directory is writable
verbose = False
# Step 2: Load queries from the CSV file (one per line)
query_file = "query.csv"
with open(query_file, "r") as f:
    queries = [line.strip() for line in f if line.strip()]

print(f"Loaded {len(queries)} queries.")

# Step 3: Initialize the TTS model
print("Loading TTS model...")
tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=True)
tts.to("cuda")  # Use "cuda" if GPU is available, otherwise "cpu"
print("TTS model loaded.")

# Step 4: Convert queries to audio
results = []
count = 0
for i, query in enumerate(queries):
    if count >= 6000:
        break
    try:
        with suppress_stdout():
            audio = tts.tts(query)  # Returns NumPy float32 array at 22050 Hz
        results.append((query, audio))
    except Exception as e:
        print(f"[{i+1}/{len(queries)}] Failed: {query} — Error: {e}")
    count += 1
# Step 5: Save all results to a pickle file
output_pkl = "queries_audio.pkl"
with open(output_pkl, "wb") as f:
    pickle.dump(results, f)

print(f"Saved {len(results)} entries to {output_pkl}")
