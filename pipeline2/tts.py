import logging
logging.disable(logging.CRITICAL)

import torch
import soundfile as sf
from torch.nn.utils.rnn import pad_sequence
from nemo.collections.tts.models import FastPitchModel, HifiGanModel

# Load models
spec_generator = FastPitchModel.from_pretrained("nvidia/tts_en_fastpitch").cuda().eval()
vocoder = HifiGanModel.from_pretrained(model_name="nvidia/tts_hifigan").cuda().eval()

# Input text list
sentences = [
    "Hello, how are you?",
    "This is a test of the FastPitch model.",
    "Batched inference makes TTS faster!"
]

# Parse and pad
with torch.no_grad():
    token_list = [spec_generator.parse(s).squeeze(0) for s in sentences]  # Fix here
    tokens = pad_sequence(token_list, batch_first=True).to(spec_generator.device)  # [B, T]

    spectrograms = spec_generator.generate_spectrogram(tokens=tokens)
    audios = vocoder.convert_spectrogram_to_audio(spec=spectrograms)

# Save outputs
for i, audio in enumerate(audios):
    sf.write(f"output_{i}.wav", audio.cpu().numpy(), 22050)
