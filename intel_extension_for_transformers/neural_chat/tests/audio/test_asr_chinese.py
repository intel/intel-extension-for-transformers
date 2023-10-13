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

import os
import sys

from intel_extension_for_transformers.neural_chat.pipeline.plugins.audio.asr_chinese import ChineseAudioSpeechRecognition
import unittest
import shutil

class TestChineseASR(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.executor = ChineseAudioSpeechRecognition()

    def test_audio2text(self):
        audio_path = "/intel-extension-for-transformers/intel_extension_for_transformers/neural_chat/assets/audio/welcome.wav"
        if os.path.exists(audio_path):
            text = self.executor.pre_llm_inference_actions(audio_path)
        else:
            text = self.executor.pre_llm_inference_actions("../../assets/audio/welcome.wav")
        self.assertEqual(len(text), 5)

if __name__ == "__main__":
    unittest.main()