# !/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2024 Intel Corporation
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

import torch

import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import tempfile
import os
from huggingsound import SpeechRecognitionModel
import commons

import soundfile as sf
import utils
from models import SynthesizerTrn
from text import text_to_sequence

"""Usage:
export no_proxy="localhost,127.0.0.1"
nohup python -u app.py &
"""


ASR_MODEL_PATH = "spycsh/shanghainese-wav2vec-3800"
TRANSLATE_MODEL_PATH = "spycsh/shanghainese-opus-sh-zh-3500"

REVERSE_MODEL_NAME = "spycsh/shanghainese-opus-zh-sh-4000"

device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 1

asr_model = SpeechRecognitionModel(ASR_MODEL_PATH, device=device)

translate_tokenizer = AutoTokenizer.from_pretrained(TRANSLATE_MODEL_PATH)
translate_model = AutoModelForSeq2SeqLM.from_pretrained(TRANSLATE_MODEL_PATH).to(device)


reverse_translate_tokenizer = AutoTokenizer.from_pretrained(REVERSE_MODEL_NAME)
reverse_translate_model = AutoModelForSeq2SeqLM.from_pretrained(REVERSE_MODEL_NAME).to(device)

hps = utils.get_hparams_from_file("model/config.json")
n_speakers = hps.data.n_speakers
net_g = SynthesizerTrn(
    len(hps.symbols),
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    n_speakers=n_speakers,  #####
    **hps.model)
_ = net_g.eval()
net_g = net_g.to(device)

_ = utils.load_checkpoint("model/model.pth", net_g)


demo = gr.Blocks()

def generate_translation(model, tokenizer, example):
    """print out the source, target and predicted raw text."""

    input_ids = example['input_ids']
    input_ids = torch.LongTensor(input_ids).view(1, -1).to(model.device)
    # print('input_ids: ', input_ids)
    generated_ids = model.generate(input_ids, max_new_tokens=64)
    # print('generated_ids: ', generated_ids)
    prediction = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    print('prediction: ', prediction)
    return prediction

def transcribe(inputs, translate=False):
    if inputs is None:
        raise gr.Error("No audio file submitted! Please upload or record an audio file before submitting your request.")
    print(inputs)
    # (44100, array([  0,   0,   0, ..., -60,  50, -37], dtype=int16))
    sr, waveform = inputs
    sf.write("test.wav", waveform, sr, format="wav")
    if not translate:
        return asr_model.transcribe(["test.wav"])[0]['transcription']
    else:
        txt = asr_model.transcribe(["test.wav"])[0]['transcription']
        with translate_tokenizer.as_target_tokenizer():
            model_inputs = translate_tokenizer(txt, max_length=64, truncation=True)
        example = {}
        example['sh'] = txt
        example['zh'] = txt    
        example['input_ids'] = model_inputs['input_ids']
        print(txt)
        print(example)
        return generate_translation(translate_model, translate_tokenizer, example)




translate=gr.Checkbox(label='Translate into Mandarin')

asr_tab = gr.Interface(
    fn=transcribe,
    inputs= [
        gr.Audio(sources=["microphone", "upload"],
                     waveform_options=gr.WaveformOptions(
                    waveform_color="#01C6FF",
                    waveform_progress_color="#0066B4",
                    skip_length=2,
                    show_controls=False,
                )
            ),
        translate
    ],

    outputs="text",

    title="Shanghainese ASR",
    description=(
        "Transcribe Mandarin long-form microphone or audio inputs to Shanghainese with the click of a button!"
    ),
    allow_flagging="never",
)

def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.symbols, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm

def t2s(inputs, reverse_translate=False):
    if inputs is None:
        raise gr.Error("No input text found! Please check the input text!")
    print(inputs) # inputs: text
    text = inputs
    if reverse_translate:
        model_inputs = reverse_translate_tokenizer(inputs,max_length=64, truncation=True)
        example = {}
        example['sh'] = text
        example['zh'] = text
        example['input_ids'] = model_inputs['input_ids']
        text = generate_translation(reverse_translate_model, reverse_translate_tokenizer, example)
        print(text)

    stn_tst = get_text(text, hps)
    with torch.no_grad():
        x_tst = stn_tst.unsqueeze(0).to(device)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).to(device)
        sid = torch.LongTensor([0]).to(device)
        print(x_tst, x_tst_lengths, sid)
        audio = net_g.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()
        print(audio)
    return (hps.data.sampling_rate, audio)

reverse_translate=gr.Checkbox(value=False, label='Mandarian as input text')


tts_tab = gr.Interface(
    fn=t2s,
    inputs=[
        gr.Textbox(label="input text", value="请侬让只位子，拨需要帮助个乘客，谢谢侬。"),
        reverse_translate
    ],
    outputs="audio",

    title="Shanghainese TTS",
    description=(
        "Shanghainese Text To Speech with one click!"
    ),
    allow_flagging="never",
)

with demo:
    gr.TabbedInterface([asr_tab, tts_tab], ["SH-ASR", "SH-TTS"])

    demo.launch(share=True)
