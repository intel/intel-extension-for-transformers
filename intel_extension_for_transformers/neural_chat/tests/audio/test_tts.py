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

from intel_extension_for_transformers.neural_chat.pipeline.plugins.audio.tts import TextToSpeech
from intel_extension_for_transformers.neural_chat.pipeline.plugins.audio.asr import AudioSpeechRecognition
import unittest
import shutil
import os
import time
import torch

class TestTTS(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        try:
            import habana_frameworks.torch.hpu as hthpu
            self.is_hpu_available = True
        except ImportError:
            self.is_hpu_available = False
        try:
            import intel_extension_for_pytorch as ipex
            self.is_ipex_available = True
        except ImportError:
            self.is_ipex_available = False
        if self.is_hpu_available:
            self.device = "hpu"
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tts = TextToSpeech(device=self.device)
        self.asr = AudioSpeechRecognition("openai/whisper-small", device=self.device)
        shutil.rmtree('./tmp_audio', ignore_errors=True)
        os.mkdir('./tmp_audio')

    @classmethod
    def tearDownClass(self):
        shutil.rmtree('./tmp_audio', ignore_errors=True)

    def test_tts(self):
        text = "Welcome to Neural Chat"
        output_audio_path = os.path.join(os.getcwd(), "tmp_audio/1.wav")
        output_audio_path = self.tts.text2speech(text, output_audio_path, voice="default")
        self.assertTrue(os.path.exists(output_audio_path))
        # verify accuracy
        result = self.asr.audio2text(output_audio_path)
        self.assertEqual(text.lower(), result.lower())

    def test_streaming_tts(self):
        def text_generate():
            for i in ["Ann", "Bob", "Tim"]:
                time.sleep(1)
                yield f"Welcome {i} to Neural Chat"
        gen = text_generate()
        output_audio_path = os.path.join(os.getcwd(), "tmp_audio/1.wav")
        for result_path in self.tts.stream_text2speech(gen, output_audio_path, voice="default"):
            self.assertTrue(os.path.exists(result_path))

    def test_create_speaker_embedding(self):
        driven_audio_path = \
           "/intel-extension-for-transformers/intel_extension_for_transformers/neural_chat/assets/audio/sample.wav"
        if os.path.exists(driven_audio_path):
            spk_embed = self.tts.create_speaker_embedding(driven_audio_path)
        else:
            spk_embed = self.tts.create_speaker_embedding("../../assets/audio/sample.wav")
        self.assertEqual(spk_embed.shape[0], 1)
        self.assertEqual(spk_embed.shape[1], 512)

if __name__ == "__main__":
    unittest.main()
