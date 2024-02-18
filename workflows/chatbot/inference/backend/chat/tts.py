# Copyright (c) 2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, set_seed
from datasets import load_dataset, Audio, Dataset, Features, ClassLabel
import os
import torch
from speechbrain.pretrained import EncoderClassifier
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from transformers import SpeechT5HifiGan
import soundfile as sf
from datetime import datetime
import intel_extension_for_pytorch as ipex
import time
import numpy as np
from torch.utils.data import DataLoader
import tempfile

class TextToSpeech:
    """Convert text to speech with a driven speaker embedding

    1) Default voice (Original model + Proved good default speaker embedding from trained dataset)
    2) Finetuned voice (Fine-tuned offline model of specific person's voice + corresponding embedding)
    3) Customized voice (Original model + User's customized input voice embedding)
    """
    def __init__(self):
        """Make sure your export LD_PRELOAD=<path to libiomp5.so and libtcmalloc> beforehand."""
        # default setting
        self.original_model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
        #self.original_model = ipex.optimize(self.original_model, torch.bfloat16)
        self.processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
        self.device = "cpu"
        self.spk_model_name = "speechbrain/spkrec-xvect-voxceleb"
        with tempfile.TemporaryFile(dir=os.path.join("/tmp", self.spk_model_name), mode="w+") as file:
            self.speaker_model = EncoderClassifier.from_hparams(
                source=self.spk_model_name,
                run_opts={"device": self.device},
                savedir=file.name
            )
        self.vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
        self.vocoder.eval()
        self.default_speaker_embedding = torch.load('speaker_embeddings/spk_embed_default.pt') # load the default speaker embedding

        # specific parameters for demo model
        # preload the model in case of time-consuming runtime loading
        self.demo_model = None
        if os.path.exists("finetuned_model_1000_125_few_shot.pt"):
            self.demo_model = torch.load("finetuned_model_1000_125_few_shot.pt", map_location=torch.device('cpu'))

        # self.demo_model = ipex.optimize(self.demo_model, torch.bfloat16)
        # self.speaker_embeddings = self.create_speaker_embedding(driven_audio_path)
        self.male_speaker_embeddings = None
        if os.path.exists('speaker_embeddings/spk_embed_male.pt'):
            self.male_speaker_embeddings = torch.load('speaker_embeddings/spk_embed_male.pt')

        # ipex IOMP hardware resources
        self.cpu_pool = ipex.cpu.runtime.CPUPool([i for i in range(24)])

    def create_speaker_embedding(self, driven_audio_path):
        """Create the speaker's embedding

        driven_audio_path: the driven audio of that speaker e.g. vgjwo-5bunm.mp3
        """
        audio_dataset = Dataset.from_dict({"audio": [driven_audio_path]}).cast_column("audio", Audio(sampling_rate=16000))
        waveform = audio_dataset[0]["audio"]['array']
        with torch.no_grad():
            speaker_embeddings = self.speaker_model.encode_batch(torch.tensor(waveform))
            speaker_embeddings = torch.nn.functional.normalize(speaker_embeddings, dim=2) # [1,1,512]
            # speaker_embeddings = speaker_embeddings.squeeze().cpu().numpy()
            speaker_embeddings = speaker_embeddings[0] # [1,512]
        return speaker_embeddings.cpu()

    def lookup_voice_embedding(self, voice):
        if os.path.exists(f"speaker_embeddings/spk_embed_{voice}.pt") == False:
            print("No customized speaker embedding is found! Use the default one")
            return "speaker_embeddings/spk_embed_default.pt"
        else:
            return f"speaker_embeddings/spk_embed_{voice}.pt"

    def text2speech(self, text, voice="default"):
        """Text to speech.

        text: the input text
        voice: default/male/female...
        """
        start = time.time()
        inputs = self.processor(text=text, return_tensors="pt")
        model = self.original_model
        speaker_embeddings = self.default_speaker_embedding

        if voice == "male":
            if self.demo_model == None:
                print("Finetuned model is not found! Use the default one")
            else:
                model = self.demo_model
            if self.male_speaker_embeddings == None:
                print("male speaker embedding is not found! Use the default one")
            else:
                speaker_embeddings = self.male_speaker_embeddings
        elif voice != "default":
            speaker_embeddings = torch.load(self.lookup_voice_embedding(voice))

        with torch.no_grad():
            with ipex.cpu.runtime.pin(self.cpu_pool):
                #with torch.cpu.amp.autocast(enabled=True, dtype=torch.bfloat16, cache_enabled=True):
                spectrogram = model.generate_speech(inputs["input_ids"], speaker_embeddings)
            speech = self.vocoder(spectrogram)
        now = datetime.now()
        time_stamp = now.strftime("%d_%m_%Y_%H_%M_%S")
        output_video_path = f"output_{time_stamp}.wav"
        print(f"text to speech in {time.time() - start} seconds, and dump the video at {output_video_path}")
        sf.write(output_video_path, speech.cpu().numpy(), samplerate=16000)
        return output_video_path
