from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import torch, time
import soundfile as sf
from datasets import load_dataset



processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").cuda()
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").cuda()

inputs = processor(text="How could I know? It's an unanswerable question. Like asking an unborn child if they'll lead a good life. They haven't even been born.", return_tensors="pt")

# load xvector containing speaker's voice characteristics from a dataset
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0).cuda()
start = time.perf_counter()
for i in range(10):
    speech = model.generate_speech(inputs["input_ids"].cuda(), speaker_embeddings, vocoder=vocoder)
    print(torch.cuda.memory_allocated() / 1024**2)
print(f"Finished in {time.perf_counter() - start}")

# sf.write("speech.wav", speech.numpy(), samplerate=16000)
