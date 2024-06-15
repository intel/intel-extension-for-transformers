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

import os
import json
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import time
import commons
import utils
# from data_utils import TextAudioLoader, TextAudioCollate, TextAudioSpeakerLoader, TextAudioSpeakerCollate
from models import SynthesizerTrn
# from text.symbols import symbols
from text import text_to_sequence

from scipy.io.wavfile import write

import soundfile as sf

device = "cuda" if torch.cuda.is_available() else "cpu"


def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.symbols, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm


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

stn_tst = get_text("请侬让只位子，拨需要帮助个乘客，谢谢侬。", hps)
# stn_tst = get_text("侬是公派出去还是因私出国", hps)
with torch.no_grad():
    x_tst = stn_tst.unsqueeze(0).to(device)
    x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).to(device)
    sid = torch.LongTensor([0]).to(device)
    print(x_tst, x_tst_lengths, sid)
    T = time.time()
    audio = net_g.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()
    print(f"time in seconds: {time.time()-T}")

out_path = "out.wav"
sf.write(out_path, audio, samplerate=hps.data.sampling_rate)
print(f"Result is dumped to: {out_path}")
