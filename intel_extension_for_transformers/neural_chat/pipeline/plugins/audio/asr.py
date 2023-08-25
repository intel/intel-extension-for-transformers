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
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from datasets import load_dataset, Audio, Dataset
import time
import contextlib
from pydub import AudioSegment

from ....plugins import register_plugin

@register_plugin('asr')
class AudioSpeechRecognition():
    """Convert audio to text."""
    def __init__(self, model_name_or_path="openai/whisper-small", bf16=False, device="cpu"):
        self.device = device
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name_or_path).to(self.device)
        self.processor = WhisperProcessor.from_pretrained(model_name_or_path)
        self.model.eval()
        self.bf16 = bf16
        if self.bf16:
            import intel_extension_for_pytorch as ipex
            self.model = ipex.optimize(self.model, dtype=torch.bfloat16)

    def _convert_audio_type(self, audio_path):
        print("[ASR WARNING] Recommend to use mp3 or wav input audio type!")
        audio_file_name = audio_path.split(".")[0]
        AudioSegment.from_file(audio_path).export(f"{audio_file_name}.mp3", format="mp3")
        return f"{audio_file_name}.mp3"

    def audio2text(self, audio_path):
        """Convert audio to text

        audio_path: the path to the input audio, e.g. ~/xxx.mp3
        """
        start = time.time()
        if audio_path.split(".")[-1] in ['flac', 'ogg', 'aac', 'm4a']:
            audio_path = self._convert_audio_type(audio_path)
        elif audio_path.split(".")[-1] not in ['mp3', 'wav']:
            raise Exception("[ASR ERROR] Audio format not supported!")
        audio_dataset = Dataset.from_dict({"audio": [audio_path]}).cast_column("audio", Audio(sampling_rate=16000))
        waveform = audio_dataset[0]["audio"]['array']
        inputs = self.processor.feature_extractor(waveform, return_tensors="pt", sampling_rate=16_000).input_features.to(self.device)
        with torch.cpu.amp.autocast() if self.bf16 else contextlib.nullcontext():
            predicted_ids = self.model.generate(inputs)
        result = self.processor.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True, normalize=True)[0]
        print(f"generated text in {time.time() - start} seconds, and the result is: {result}")
        return result


    def pre_llm_inference_actions(self, audio_path):
        return self.audio2text(audio_path)
