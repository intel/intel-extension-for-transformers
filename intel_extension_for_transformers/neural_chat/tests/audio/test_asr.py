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

from intel_extension_for_transformers.neural_chat.pipeline.plugins.audio.asr import AudioSpeechRecognition
import unittest
import shutil
import torch

class TestASR(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.asr = AudioSpeechRecognition("openai/whisper-small", device=device)
        if not torch.cuda.is_available():
            self.asr_bf16 = AudioSpeechRecognition("openai/whisper-small", bf16=True)

    def test_audio2text(self):
        audio_path = "../../assets/audio/welcome.wav"
        text = self.asr.audio2text(audio_path)
        self.assertEqual(text.lower(), "Welcome to Neural Chat".lower())

    def test_audio2text_bf16(self):
        if torch.cuda.is_available():
            return
        audio_path = "../../assets/audio/welcome.wav"
        text = self.asr_bf16.audio2text(audio_path)
        self.assertEqual(text.lower(), "Welcome to Neural Chat".lower())

if __name__ == "__main__":
    unittest.main()
