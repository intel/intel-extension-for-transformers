# !/usr/bin/env python
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

from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech
from datasets import load_dataset, Audio, Dataset, Features, ClassLabel
import os
import torch
from speechbrain.pretrained import EncoderClassifier
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from transformers import SpeechT5HifiGan
import soundfile as sf
from datetime import datetime
from num2words import num2words
import logging
logging.basicConfig(
    format="%(asctime)s %(name)s:%(levelname)s:%(message)s",
    datefmt="%d-%M-%Y %H:%M:%S",
    level=logging.INFO
)

workdir = os.getcwd()

model = torch.load("finetuned_model.pt")
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")

spk_model_name = "speechbrain/spkrec-xvect-voxceleb"
device = "cuda" if torch.cuda.is_available() else "cpu"
speaker_model = EncoderClassifier.from_hparams(
    source=spk_model_name,
    run_opts={"device": device},
    savedir=os.path.join("/tmp", spk_model_name)
)
def create_speaker_embedding(waveform):
    with torch.no_grad():
        speaker_embeddings = speaker_model.encode_batch(torch.tensor(waveform))
        speaker_embeddings = torch.nn.functional.normalize(speaker_embeddings, dim=2)
        speaker_embeddings = speaker_embeddings.squeeze().cpu().numpy()
    return speaker_embeddings

audio_dataset = Dataset.from_dict({"audio": [os.path.join(workdir, "audios/samples_mp3_ted_speakers_FeiFeiLi_sample-0.mp3")]}).cast_column("audio", Audio(sampling_rate=16000))
sembeddings = create_speaker_embedding(audio_dataset[0]["audio"]['array'])
speaker_embeddings = torch.tensor(sembeddings).unsqueeze(0)
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(device)

def correct_abbreviation(text):
    correct_dict = {
        "A": "Eigh",
        "B": "bee",
        "C": "cee",
        "D": "dee",
        "E": "yee",
        "F": "ef",
        "G": "jee",
        "H": "aitch",
        "I": "I",
        "J": "jay",
        "K": "kay",
        "L": "el",
        "M": "em",
        "N": "en",
        "O": "o",
        "P": "pee",
        "Q": "cue",
        "R": "ar",
        "S": "ess",
        "T": "tee",
        "U": "u",
        "V": "vee",
        "W": "doubleliu",
        "X": "ex",
        "Y": "wy",
        "Z": "zed"
    }
    words = text.split()
    results = []
    for idx, word in enumerate(words):
        if word.isupper():
            for c in word:
                if c in correct_dict:
                    results.append(correct_dict[c])
                else:
                    results.append(c)
        else:
            results.append(word)
    return " ".join(results)

def correct_number(text):
    """Ignore the year or other exception right now"""
    words = text.split()
    results = []
    for idx, word in enumerate(words):
        if word.isdigit():
            try:
                word = num2words(word)
            except Exception as e:
                logging.info("num2words fail with word: %s and exception: %s", word, e)
        else:
            try:
                val = int(word)
                word = num2words(word)
            except ValueError:
                try:
                    val = float(word)
                    word = num2words(word)
                except ValueError:
                    pass
        results.append(word)
    return " ".join(results)


while True:
    try:
        text = input("Write a sentence to let the talker speak:\n").strip()
        text = correct_abbreviation(text)
        text = correct_number(text)
        inputs = processor(text=text, return_tensors="pt")
        spectrogram = model.generate_speech(inputs["input_ids"].to(device), speaker_embeddings.to(device))
        with torch.no_grad():
            speech = vocoder(spectrogram)
        now = datetime.now()
        time_stamp = now.strftime("%d_%m_%Y_%H_%M_%S")
        sf.write(f"output_{time_stamp}.wav", speech.cpu().numpy(), samplerate=16000)
    except Exception as e:
        logging.info("Catch exception: %s", e)
        logging.info("Restarting\n")
