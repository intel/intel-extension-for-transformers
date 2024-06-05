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
from huggingsound import SpeechRecognitionModel
import os
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 1
MODEL_PATH = "spycsh/shanghainese-wav2vec-3800"
model = SpeechRecognitionModel(MODEL_PATH, device=device)
audio_paths = ["Shanghai_Dialect_Dict/Split_WAV1/1956.wav"]

transcriptions = model.transcribe(audio_paths, batch_size=batch_size)

print(transcriptions)
