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

import unittest
import torch
import os
import shutil
import spacy
from unittest.mock import MagicMock, patch
from intel_extension_for_transformers.neural_chat import build_chatbot, PipelineConfig
from intel_extension_for_transformers.neural_chat.utils.common import get_device_type

class TestBuildChatbotNormalCases(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        pass

    @classmethod
    def tearDownClass(self) -> None:
        if os.path.exists("./app.log"):
            os.remove("./app.log")
        if os.path.exists("./output"):
            shutil.rmtree("./output")
        if os.path.exists("./gptcache_data"):
            shutil.rmtree("./gptcache_data")

    @unittest.skipIf(get_device_type() != 'cpu', "Only run this test on CPU")
    def test_valid_model_name(self):
        # Test with valid model name
        config = PipelineConfig(model_name_or_path="facebook/opt-125m")
        result = build_chatbot(config)
        self.assertIsNotNone(result)

    @unittest.skipIf(get_device_type() != 'cpu', "Only run this test on CPU")
    @patch('torch.cuda.is_available', MagicMock(return_value=True))
    def test_valid_cuda_device(self):
        # Test with valid CUDA configuration
        config = PipelineConfig(model_name_or_path="facebook/opt-125m")
        config.device = "cuda"
        result = build_chatbot(config)
        self.assertIsNone(result)

    @unittest.skipIf(get_device_type() != 'cpu', "Only run this test on CPU")
    @patch('torch.xpu.is_available', MagicMock(return_value=True))
    def test_valid_xpu_device(self):
        # Test with valid XPU configuration
        config = PipelineConfig(model_name_or_path="facebook/opt-125m")
        config.device = "xpu"
        result = build_chatbot(config)
        self.assertIsNone(result)

    @unittest.skipIf(get_device_type() != 'cpu', "Only run this test on CPU")
    def test_valid_cpu_device(self):
        # Test with valid CPU configuration
        config = PipelineConfig(model_name_or_path="facebook/opt-125m")
        config.device = "cpu"
        result = build_chatbot(config)
        self.assertIsNotNone(result)

    @unittest.skipIf(get_device_type() != 'cpu', "Only run this test on CPU")
    def test_enable_plugin_tts(self):
        # Test enabling Text-to-Speech plugin
        config = PipelineConfig(model_name_or_path="facebook/opt-125m")
        config.plugins = {"tts": {"enable": True, "args":
            {"device": "cpu", "voice": "default", "stream_mode": "true", "output_audio_path": "./output_audio"}}}
        result = build_chatbot(config)
        self.assertIsNotNone(result)

    @unittest.skipIf(get_device_type() != 'cpu', "Only run this test on CPU")
    def test_enable_plugin_tts_chinese(self):
        # Test enabling Chinese Text-to-Speech plugin
        config = PipelineConfig(model_name_or_path="facebook/opt-125m")
        config.plugins = {"tts_chinese": {"enable": True, "args": {}}}
        result = build_chatbot(config)
        self.assertIsNotNone(result)

    @unittest.skipIf(get_device_type() != 'cpu', "Only run this test on CPU")
    def test_enable_plugin_asr(self):
        # Test enabling Audio Speech Recognition plugin
        config = PipelineConfig(model_name_or_path="facebook/opt-125m")
        config.plugins = {"asr": {"enable": True, "args":
            {"device": "cpu", "model_name_or_path": "openai/whisper-small"}}}
        result = build_chatbot(config)
        self.assertIsNotNone(result)

    @unittest.skipIf(get_device_type() != 'cpu', "Only run this test on CPU")
    def test_enable_plugin_cache(self):
        # Test enabling Cache plugin
        config = PipelineConfig(model_name_or_path="facebook/opt-125m")
        config.plugins = {"cache": {"enable": True, "args": {}}}
        result = build_chatbot(config)
        self.assertIsNotNone(result)

    @unittest.skipIf(get_device_type() != 'cpu', "Only run this test on CPU")
    def test_enable_plugin_safety_checker(self):
        # Test enabling Safety Checker plugin
        config = PipelineConfig(model_name_or_path="facebook/opt-125m")
        config.plugins = {"safety_checker": {"enable": True, "args": {}}}
        result = build_chatbot(config)
        self.assertIsNotNone(result)

    @unittest.skipIf(get_device_type() != 'cpu', "Only run this test on CPU")
    def test_enable_plugin_ner(self):
        # Test enabling Named Entity Recognition plugin
        spacy.cli.download("en_core_web_lg")
        config = PipelineConfig(model_name_or_path="facebook/opt-125m")
        config.plugins = {"ner": {"enable": True, "args": {}}}
        result = build_chatbot(config)
        self.assertIsNotNone(result)

    @unittest.skipIf(get_device_type() != 'cpu', "Only run this test on CPU")
    def test_enable_plugin_face_animation(self):
        # Test enabling Face Animation plugin
        config = PipelineConfig(model_name_or_path="facebook/opt-125m")
        config.plugins = {"face_animation": {"enable": True, "args": {}}}
        result = build_chatbot(config)
        self.assertIsNotNone(result)


if __name__ == '__main__':
    unittest.main()
