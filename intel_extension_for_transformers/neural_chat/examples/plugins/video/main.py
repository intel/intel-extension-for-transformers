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

from intel_extension_for_transformers.neural_chat.pipeline.plugins.video.face_animation.sadtalker import SadTalker
import requests
import torch


sample_audio_url = "https://github.com/intel/intel-extension-for-transformers/raw/main/intel_extension_for_transformers/neural_chat/assets/audio/welcome.wav"
sample_img_url = "https://raw.githubusercontent.com/OpenTalker/SadTalker/main/examples/source_image/full_body_2.png"
img_data = requests.get(sample_img_url).content
source_image = "sample_img.jpg"
driven_audio = "sample_audio.wav"
with open(source_image, 'wb') as f:
    f.write(img_data)
audio_data = requests.get(sample_audio_url).content
with open(driven_audio, 'wb') as f:
    f.write(audio_data)
output_video_path = "response.mp4"
checkpoint_dir = "checkpoints"
device = "cuda" if torch.cuda.is_available() else "cpu"
sadtalker = SadTalker(device=device, checkpoint_dir=checkpoint_dir, bf16=True, p_num=4, enhancer=None, output_video_path=output_video_path)

sadtalker.convert(source_image=source_image, driven_audio=driven_audio)
