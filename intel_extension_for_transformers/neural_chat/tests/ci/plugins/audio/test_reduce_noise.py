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

from intel_extension_for_transformers.neural_chat.pipeline.plugins.audio.utils.reduce_noise import NoiseReducer
import unittest
import os
import librosa
from intel_extension_for_transformers.neural_chat.utils.common import get_device_type

class TestReduceNoise(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.audio_path = "/intel-extension-for-transformers/intel_extension_for_transformers/neural_chat/assets/audio/welcome.wav"
        if not os.path.exists(self.audio_path):
            self.audio_path = "../assets/audio/welcome.wav"
        self.y, sr = librosa.load(self.audio_path, sr=16000)
        self.reducer = NoiseReducer(sr=sr)
        self.reducer_nonstationary = NoiseReducer(sr=sr, nonstationary=True)


    @classmethod
    def tearDownClass(self):
        pass

    @unittest.skipIf(get_device_type() == 'xpu' or get_device_type() == 'hpu', "Skip this test on XPU and HPU devices")
    def test_reduce_noise_stationary(self):
        output_audio_path = self.reducer.reduce_audio_amplify(self.audio_path, self.y)
        self.assertTrue(os.path.exists(output_audio_path))

    @unittest.skipIf(get_device_type() == 'xpu' or get_device_type() == 'hpu', "Skip this test on XPU and HPU devices")
    def test_reduce_noise_nonstationary(self):
        output_audio_path = self.reducer_nonstationary.reduce_audio_amplify(self.audio_path, self.y)
        self.assertTrue(os.path.exists(output_audio_path))

if __name__ == "__main__":
    unittest.main()
