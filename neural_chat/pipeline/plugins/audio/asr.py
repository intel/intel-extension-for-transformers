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

import torch
import intel_extension_for_pytorch as ipex
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from datasets import load_dataset, Audio, Dataset
import time
import contextlib

class AudioSpeechRecognition:
    """Convert audio to text."""
    def __init__(self, model_name_or_path="openai/whisper-small", bf16=False):
        self.device = "cpu"
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name_or_path).to(self.device)
        self.processor = WhisperProcessor.from_pretrained(model_name_or_path)
        self.model.eval()
        self.bf16 = bf16
        if self.bf16:
            self.model = ipex.optimize(self.model, dtype=torch.bfloat16)

    def audio2text(self, audio_path):
        """Convert audio to text

        audio_path: the path to the input audio, e.g. ~/xxx.mp3
        """
        start = time.time()
        audio_dataset = Dataset.from_dict({"audio": [audio_path]}).cast_column("audio", Audio(sampling_rate=16000))
        waveform = audio_dataset[0]["audio"]['array']
        inputs = self.processor.feature_extractor(waveform, return_tensors="pt", sampling_rate=16_000).input_features.to(self.device)
        with torch.cpu.amp.autocast() if self.bf16 else contextlib.nullcontext():
            predicted_ids = self.model.generate(inputs)
        result = self.processor.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True, normalize=True)[0]
        print(f"generated text in {time.time() - start} seconds, and the result is: {result}")
        return result