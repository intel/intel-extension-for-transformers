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
from transformers import set_seed

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
        self.tts_noise_reducer = TextToSpeech(device=self.device, reduce_noise=True)
        shutil.rmtree('./tmp_audio', ignore_errors=True)
        os.mkdir('./tmp_audio')

    @classmethod
    def tearDownClass(self):
        shutil.rmtree('./tmp_audio', ignore_errors=True)

    def test_tts(self):
        text = "Welcome to Neural Chat"
        output_audio_path = os.path.join(os.getcwd(), "tmp_audio/1.wav")
        set_seed(555)
        output_audio_path = self.tts.text2speech(text, output_audio_path, voice="default")
        self.assertTrue(os.path.exists(output_audio_path))
        # verify accuracy
        result = self.asr.audio2text(output_audio_path)
        self.assertEqual(text.lower(), result.lower())

    def test_tts_customized_voice(self):
        text = "Welcome to Neural Chat"
        output_audio_path = os.path.join(os.getcwd(), "tmp_audio/3.wav")
        set_seed(555)
        output_audio_path = self.tts.text2speech(text, output_audio_path, voice="male")
        self.assertTrue(os.path.exists(output_audio_path))
        # verify accuracy
        result = self.asr.audio2text(output_audio_path)
        self.assertEqual(text.lower(), result.lower())
        output_audio_path = os.path.join(os.getcwd(), "tmp_audio/4.wav")
        set_seed(555)
        output_audio_path = self.tts.text2speech(text, output_audio_path, voice="female")
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

    def test_tts_long_text(self):
        text = "Intel Extension for Transformers is an innovative toolkit to accelerate Transformer-based models on " + \
        "Intel platforms, in particular effective on 4th Intel Xeon Scalable processor Sapphire Rapids " + \
        "(codenamed Sapphire Rapids)"
        output_audio_path = os.path.join(os.getcwd(), "tmp_audio/2.wav")
        set_seed(555)
        output_audio_path = self.tts.text2speech(text, output_audio_path, voice="default", do_batch_tts=True, batch_length=120)
        result = self.asr.audio2text(output_audio_path)
        self.assertTrue(os.path.exists(output_audio_path))
        self.assertEqual("intel extension for transformers is an innovative toolkit to accelerate transformer based " + \
                         "models on intel platforms in particular effective on 4th intel xeon scalable processor " + \
                            "sapphire rapids codenamed sapphire rapids", result)

    def test_create_speaker_embedding(self):
        driven_audio_path = \
           "/intel-extension-for-transformers/intel_extension_for_transformers/neural_chat/assets/audio/sample.wav"
        if os.path.exists(driven_audio_path):
            spk_embed = self.tts.create_speaker_embedding(driven_audio_path)
        else:
            spk_embed = self.tts.create_speaker_embedding("../assets/audio/sample.wav")
        self.assertEqual(spk_embed.shape[0], 1)
        self.assertEqual(spk_embed.shape[1], 512)

    def test_tts_remove_noise(self):
        text = "hello there"
        output_audio_path = os.path.join(os.getcwd(), "tmp_audio/5.wav")
        set_seed(555)
        output_audio_path = self.tts_noise_reducer.text2speech(text, output_audio_path, voice="default")
        self.assertTrue(os.path.exists(output_audio_path))
        # verify accuracy
        result = self.asr.audio2text(output_audio_path)
        self.assertEqual(text.lower(), result.lower())

    def test_tts_messy_input(self):
        text = "Please refer to the following responses to this inquiry:\n" + 244 * "* " + "*"
        output_audio_path = os.path.join(os.getcwd(), "tmp_audio/6.wav")
        set_seed(555)
        output_audio_path = self.tts_noise_reducer.text2speech(text, output_audio_path, voice="default")
        self.assertTrue(os.path.exists(output_audio_path))
        # verify accuracy
        result = self.asr.audio2text(output_audio_path)
        self.assertEqual("please refer to the following responses to this inquiry", result.lower())

    def test_tts_speedup(self):
        text = "hello there."
        set_seed(555)
        output_audio_path1 = os.path.join(os.getcwd(), "tmp_audio/7.wav")
        output_audio_path1 = self.tts_noise_reducer.text2speech(text, output_audio_path1, voice="default", speedup=1.0,)
        set_seed(555)
        output_audio_path2 = os.path.join(os.getcwd(), "tmp_audio/8.wav")
        output_audio_path2 = self.tts_noise_reducer.text2speech(text, output_audio_path2, voice="default", speedup=2.0,)
        self.assertTrue(os.path.exists(output_audio_path2))
        from pydub import AudioSegment
        waveform1 = AudioSegment.from_file(output_audio_path1).set_frame_rate(16000)
        waveform2 = AudioSegment.from_file(output_audio_path2).set_frame_rate(16000)
        self.assertNotEqual(len(waveform1), len(waveform2))

if __name__ == "__main__":
    unittest.main()
