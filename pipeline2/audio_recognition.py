import sys
sys.path.append("./SenseVoice")
import time
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
from funasr.utils.load_utils import load_audio_text_image_video, extract_fbank
from utils.frontend import WavFrontend, WavFrontendOnline
import librosa
from model import SenseVoiceSmall
# model_dir = "FunAudioLLM/SenseVoiceSmall"
import pickle
import torch
import numpy as np

# audio_file_path = 
pkl_dir = "/mnt/nvme0/vortex_pipeline1/queries_audio1.pkl"

model_dir = "iic/SenseVoiceSmall"
m, kwargs = SenseVoiceSmall.from_pretrained(model=model_dir, device="cuda:0")
m.eval()

kwargs["data_type"] = "fbank"
kwargs["sound"] = "fbank"
frontend = kwargs["frontend"]

with open(pkl_dir, "rb") as f:
    list_np_waveform = pickle.load(f)
    
print(f"Got {len(list_np_waveform)} queries and audio samples")

audio_sample_list = load_audio_text_image_video(
    f"{kwargs['model_path']}/example/en.mp3",
    fs = frontend.fs,
    audio_fs=kwargs.get("fs", 16000),
    data_type=kwargs.get("data_type", "sound"),
    tokenizer=kwargs["tokenizer"],
)


BS = 8
total_runs = 1000
list_of_runtimes = []
batch_audio_sample = []



for i in range(total_runs):
    if i == 1:
        start = time.perf_counter()
        
    curr_array = list_np_waveform[i%(len(list_np_waveform))][-1]
    if len(curr_array) <= 200000:
        batch_audio_sample.append(curr_array)
    
        if (i+1) % BS == 0:
            # audio_sample_list = torch.from_numpy(np.array(batch_audio_sample))
        # raw_queries = list_np_waveform[i][0]
        # print(f"For item {i}, the raw text: \n {raw_queries}")
            # audio_sample_list = torch.from_numpy(list_np_waveform[i][-1])
            speech, speech_lengths = extract_fbank(
                batch_audio_sample, data_type=kwargs.get("data_type", "sound"), frontend=frontend
            )
            
            start_time = time.perf_counter()
            res = m.inference(
                data_in=speech,
                data_lengths=speech_lengths,
                language="en", # "zh", "en", "yue", "ja", "ko", "nospeech"
                use_itn=False,
                ban_emo_unk=True,
                **kwargs,
            )
            list_of_runtimes.append(time.perf_counter() - start_time)
            # import pdb; pdb.set_trace()
            text_list = []
            for idx in range(len(res[0])):
                text_list.append(rich_transcription_postprocess(res[0][idx]["text"]))
            # text = rich_transcription_postprocess(res[0][0]["text"])
            # print(f"For batch id {(i+1) / BS}, the list of audio recs are: \n {text_list}")
            batch_audio_sample = []
    else:
        speech, speech_lengths = extract_fbank(
            torch.from_numpy(curr_array).unsqueeze(0), data_type=kwargs.get("data_type", "sound"), frontend=frontend
        )
        res = m.inference(
                data_in=speech,
                data_lengths=speech_lengths,
                language="en", # "zh", "en", "yue", "ja", "ko", "nospeech"
                use_itn=False,
                ban_emo_unk=True,
                **kwargs,
            )
        text = rich_transcription_postprocess(res[0][0]["text"])
        # print(f"For single item {i}: \n {text}")
        
print(f"List of runtime: {list_of_runtimes}")
print(f"avg runetime: {np.mean(np.array(list_of_runtimes[1:]))}")