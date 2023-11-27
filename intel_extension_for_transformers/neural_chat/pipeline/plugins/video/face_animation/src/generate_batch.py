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

import os

from tqdm import tqdm
import torch
import numpy as np
import random
import scipy.io as scio
import intel_extension_for_transformers.neural_chat.pipeline.plugins.video.face_animation.src.utils.audio as audio


def crop_pad_audio(wav, audio_length):
    if len(wav) > audio_length:
        wav = wav[:audio_length]
    elif len(wav) < audio_length:
        wav = np.pad(wav, [0, audio_length - len(wav)], mode="constant", constant_values=0)
    return wav


def parse_audio_length(audio_length, sr, fps):
    bit_per_frames = sr / fps

    num_frames = int(audio_length / bit_per_frames)
    audio_length = int(num_frames * bit_per_frames)

    return audio_length, num_frames


def generate_blink_seq(num_frames):
    ratio = np.zeros((num_frames, 1))
    frame_id = 0
    while frame_id in range(num_frames):
        start = 80
        if frame_id + start + 9 <= num_frames - 1:
            ratio[frame_id + start : frame_id + start + 9, 0] = [0.5, 0.6, 0.7, 0.9, 1, 0.9, 0.7, 0.6, 0.5]
            frame_id = frame_id + start + 9
        else:
            break
    return ratio


def generate_blink_seq_randomly(num_frames):
    ratio = np.zeros((num_frames, 1))
    if num_frames <= 20:
        return ratio
    frame_id = 0
    while frame_id in range(num_frames):
        start = random.choice(range(min(10, num_frames), min(int(num_frames / 2), 70)))
        if frame_id + start + 5 <= num_frames - 1:
            ratio[frame_id + start : frame_id + start + 5, 0] = [0.5, 0.9, 1.0, 0.9, 0.5]
            frame_id = frame_id + start + 5
        else:
            break
    return ratio


def get_data(first_coeff_path, audio_path, device, still=False):
    syncnet_mel_step_size = 16
    fps = 25

    pic_name = os.path.splitext(os.path.split(first_coeff_path)[-1])[0]
    audio_name = os.path.splitext(os.path.split(audio_path)[-1])[0]

    wav = audio.load_wav(audio_path, 16000)
    wav_length, num_frames = parse_audio_length(len(wav), 16000, 25)
    wav = crop_pad_audio(wav, wav_length)
    orig_mel = audio.melspectrogram(wav).T
    spec = orig_mel.copy()  # nframes 80
    indiv_mels = []

    for i in tqdm(range(num_frames), "mel:"):
        start_frame_num = i - 2
        start_idx = int(80.0 * (start_frame_num / float(fps)))
        end_idx = start_idx + syncnet_mel_step_size
        seq = list(range(start_idx, end_idx))
        seq = [min(max(item, 0), orig_mel.shape[0] - 1) for item in seq]
        m = spec[seq, :]
        indiv_mels.append(m.T)
    indiv_mels = np.asarray(indiv_mels)  # T 80 16

    ratio = generate_blink_seq_randomly(num_frames)  # T
    source_semantics_path = first_coeff_path
    source_semantics_dict = scio.loadmat(source_semantics_path)
    ref_coeff = source_semantics_dict["coeff_3dmm"][:1, :70]  # 1 70
    ref_coeff = np.repeat(ref_coeff, num_frames, axis=0)

    indiv_mels = torch.FloatTensor(indiv_mels).unsqueeze(1).unsqueeze(0)  # bs T 1 80 16

    # if still:
    #     ratio = torch.FloatTensor(ratio).unsqueeze(0).fill_(0.0)  # bs T
    # else:
    # Enable eye blinking as default!
    ratio = torch.FloatTensor(ratio).unsqueeze(0)
    # bs T
    ref_coeff = torch.FloatTensor(ref_coeff).unsqueeze(0)  # bs 1 70

    indiv_mels = indiv_mels.to(device)
    ratio = ratio.to(device)
    ref_coeff = ref_coeff.to(device)

    return {
        "indiv_mels": indiv_mels,
        "ref": ref_coeff,
        "num_frames": num_frames,
        "ratio_gt": ratio,
        "audio_name": audio_name,
        "pic_name": pic_name,
    }
