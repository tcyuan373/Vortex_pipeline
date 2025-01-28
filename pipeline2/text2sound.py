# Following pip packages need to be installed:
# !pip install git+https://github.com/huggingface/transformers sentencepiece datasets

## Using speech t5
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import torch
import soundfile as sf
from datasets import load_dataset

processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

inputs = processor(text="Hello, my dog is cute", return_tensors="pt")

# load xvector containing speaker's voice characteristics from a dataset
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)

sf.write("speech.wav", speech.numpy(), samplerate=16000)


# ### now adapting espnet2
# from espnet2.bin.tts_inference import Text2Speech
# from espnet2.utils.types import str_or_none
# import time
# import torch
# import soundfile as sf


# lang = 'English'
# tag = 'kan-bayashi/ljspeech_vits' 
# vocoder_tag = "none" 
# new_div_idx = '0'
# new_device = 'cuda:' + new_div_idx


# text2speech = Text2Speech.from_pretrained(
#     model_tag=str_or_none(tag),
#     vocoder_tag=str_or_none(vocoder_tag),
#     device="cpu",
#     # Only for Tacotron 2 & Transformer
#     threshold=0.5,
#     # Only for Tacotron 2
#     minlenratio=0.0,
#     maxlenratio=10.0,
#     use_att_constraint=False,
#     backward_window=1,
#     forward_window=3,
#     # Only for FastSpeech & FastSpeech2 & VITS
#     speed_control_alpha=1.0,
#     # Only for VITS
#     noise_scale=0.333,
#     noise_scale_dur=0.333,
# )

# text2speech.model.to(new_device)
# text2speech.device = new_device
# # decide the input sentence by yourself
# # print(f"Input your favorite sentence in {lang}.")
# # x = 'May the force be with you!'
# x = 'To be, or not to be, that is the question:\
# Whether tis nobler in the mind to suffer\
# The slings and arrows of outrageous fortune,\
# Or to take arms against a sea of troubles'

# # synthesis
# with torch.no_grad():
#     start = time.time()
#     wav = text2speech(x)["wav"]
# rtf = (time.time() - start) / (len(wav) / text2speech.fs)
# print(f"RTF = {rtf:5f}")
# print(wav.device)
# # let us listen to generated samples
# # from IPython.display import display, Audio
# # display(Audio(wav.view(-1).cpu().numpy(), rate=text2speech.fs))

# torch.cuda.empty_cache()

# sf.write("out.wav", wav.cpu().numpy(), text2speech.fs, "PCM_16")