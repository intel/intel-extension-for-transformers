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

workdir = os.getcwd()

model = torch.load("peppa_2000.pt")
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

audio_dataset = Dataset.from_dict({"audio": [os.path.join(workdir, "audios/4.mp3")]}).cast_column("audio", Audio(sampling_rate=16000))
sembeddings = create_speaker_embedding(audio_dataset[0]["audio"]['array'])
speaker_embeddings = torch.tensor(sembeddings).unsqueeze(0)
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").cuda()

def correct_abbreviation(text):
    # formula: if one word is all capital letters, then correct this whole word
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
        if word.isupper(): # W3C is also upper
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
        if word.isdigit(): # if word is positive integer, it must can be num2words
            try:
                word = num2words(word)
            except Exception as e:
                print(f"num2words fail with word: {word} and exception: {e}")
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
        text = input("Write a sentence to let peppa pig speak:\n")
        text = correct_abbreviation(text)
        text = correct_number(text)
        inputs = processor(text=text, return_tensors="pt")
        spectrogram = model.generate_speech(inputs["input_ids"].cuda(), speaker_embeddings.cuda())
        with torch.no_grad():
            speech = vocoder(spectrogram)
        now = datetime.now()
        time_stamp = now.strftime("%d_%m_%Y_%H_%M_%S")
        sf.write(f"output_{time_stamp}.wav", speech.cpu().numpy(), samplerate=16000)
    except Exception as e:
        print(f"Catch exception: {e}")
        print("Restarting\n")