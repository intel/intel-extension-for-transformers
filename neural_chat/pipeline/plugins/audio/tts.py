#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2023 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
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
from typing import Any, Dict, List, Union
from transformers import SpeechT5HifiGan
import soundfile as sf
import intel_extension_for_pytorch as ipex
import numpy as np
import contextlib

from .utils.english_normalizer import EnglishNormalizer


class TextToSpeech:
    """Convert text to speech with a driven speaker embedding

    1) Default voice (Original model + Proved good default speaker embedding from trained dataset)
    2) Finetuned voice (Fine-tuned offline model of specific person, such as Pat's voice + corresponding embedding)
    3) Customized voice (Original model + User's customized input voice embedding)
    """
    def __init__(self):
        """Make sure your export LD_PRELOAD=<path to libiomp5.so and libtcmalloc> beforehand."""
        # default setting
        self.original_model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
        self.processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
        self.device = "cpu"
        self.spk_model_name = "speechbrain/spkrec-xvect-voxceleb"
        self.speaker_model = EncoderClassifier.from_hparams(
            source=self.spk_model_name,
            run_opts={"device": self.device},
            savedir=os.path.join("/tmp", self.spk_model_name)
        )
        self.vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
        self.vocoder.eval()
        self.default_speaker_embedding = torch.load('speaker_embeddings/spk_embed_default.pt') # load the default speaker embedding

        # preload the demo model in case of time-consuming runtime loading
        self.pat_model = None
        if os.path.exists("pat.pt"):
            self.pat_model = torch.load("pat.pt", map_location=torch.device('cpu'))

        self.pat_speaker_embeddings = None
        if os.path.exists('speaker_embeddings/spk_embed_pat.pt'):
            self.pat_speaker_embeddings = torch.load('speaker_embeddings/spk_embed_pat.pt')

        # ipex IOMP hardware resources
        if 'LD_PRELOAD' in os.environ and 'libiomp' in os.environ['LD_PRELOAD']:
            self.cpu_pool = ipex.cpu.runtime.CPUPool([i for i in range(24)])
        else:
            print("Warning! You have not preloaded iomp beforehand and that may lead to performance issue")
            self.cpu_pool = None

        self.normalizer = EnglishNormalizer()

    def create_speaker_embedding(self, driven_audio_path):
        """Create the speaker's embedding.

        driven_audio_path: the driven audio of that speaker
        """
        audio_dataset = Dataset.from_dict({"audio": [driven_audio_path]}).cast_column("audio", Audio(sampling_rate=16000))
        waveform = audio_dataset[0]["audio"]['array']
        with torch.no_grad():
            speaker_embeddings = self.speaker_model.encode_batch(torch.tensor(waveform))
            speaker_embeddings = torch.nn.functional.normalize(speaker_embeddings, dim=2) # [1,1,512]
            speaker_embeddings = speaker_embeddings[0] # [1,512]
        return speaker_embeddings.cpu()

    def _lookup_voice_embedding(self, voice):
        if os.path.exists(f"speaker_embeddings/spk_embed_{voice}.pt") == False:
            print("No customized speaker embedding is found! Use the default one")
            return "speaker_embeddings/spk_embed_default.pt"
        else:
            return f"speaker_embeddings/spk_embed_{voice}.pt"

    def text2speech(self, text, output_audio_path, voice="default"):
        """Text to speech.

        text: the input text
        voice: default/pat/huma/tom/eric...
        """
        text = self.normalizer.correct_abbreviation(text)
        text = self.normalizer.correct_number(text)
        inputs = self.processor(text=text, return_tensors="pt")
        model = self.original_model
        speaker_embeddings = self.default_speaker_embedding

        if voice == "pat":
            if self.pat_model == None:
                print("Finetuned model is not found! Use the default one")
            else:
                model = self.pat_model
            if self.pat_speaker_embeddings == None:
                print("Pat's speaker embedding is not found! Use the default one")
            else:
                speaker_embeddings = self.pat_speaker_embeddings
        elif voice != "default":
            speaker_embeddings = torch.load(self._lookup_voice_embedding(voice))

        with torch.no_grad():
            with ipex.cpu.runtime.pin(self.cpu_pool) if self.cpu_pool else contextlib.nullcontext():
                spectrogram = model.generate_speech(inputs["input_ids"], speaker_embeddings)
            speech = self.vocoder(spectrogram)
        sf.write(output_audio_path, speech.cpu().numpy(), samplerate=16000)
        return output_audio_path

    def stream_text2speech(self, generator, answer_speech_path, voice="default"):
        """Stream the generation of audios with an LLM text generator."""
        for idx, response in enumerate(generator):
            yield self.text2speech(response, f"{answer_speech_path}_{idx}.wav", voice)
