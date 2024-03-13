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

from intel_extension_for_transformers.neural_chat.pipeline.plugins.audio.tts_multilang import (
    MultilangTextToSpeech,
)
import unittest, os, shutil


class TestMultilangTextToSpeech(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        shutil.rmtree("./tmp_audio", ignore_errors=True)
        os.mkdir("./tmp_audio")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree('./tmp_audio', ignore_errors=True)

    def test_pre_llm_inference_actions_int8(self):
        text = "欢迎来到英特尔，welcome to Intel。こんにちは！"
        output_audio_path = MultilangTextToSpeech().post_llm_inference_actions(text=text)
        self.assertTrue(os.path.exists(output_audio_path))

    def test_pre_llm_inference_actions_bf16(self):
        text = "欢迎来到英特尔，welcome to Intel。こんにちは！"
        output_audio_path = MultilangTextToSpeech(
            device="cpu", precision="bf16"
        ).post_llm_inference_actions(text=text)
        self.assertTrue(os.path.exists(output_audio_path))

    def test_pre_llm_inference_actions_fp32(self):
        text = "欢迎来到英特尔，welcome to Intel。こんにちは！"
        output_audio_path = MultilangTextToSpeech(
            device="cpu", precision="fp32"
        ).post_llm_inference_actions(text=text)
        self.assertTrue(os.path.exists(output_audio_path))


if __name__ == "__main__":
    unittest.main()
