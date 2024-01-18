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

gaudi2_content = """
Habana Gaudi2 and 4th Gen Intel Xeon Scalable processors deliver leading performance and optimal cost savings for AI training.
Today, MLCommons published results of its industry AI performance benchmark, MLPerf Training 3.0, in which both the Habana® Gaudi®2 deep learning accelerator and the 4th Gen Intel® Xeon® Scalable processor delivered impressive training results.
The latest MLPerf Training 3.0 results underscore the performance of Intel's products on an array of deep learning models. The maturity of Gaudi2-based software and systems for training was demonstrated at scale on the large language model, GPT-3. Gaudi2 is one of only two semiconductor solutions to submit performance results to the benchmark for LLM training of GPT-3. 
Gaudi2 also provides substantially competitive cost advantages to customers, both in server and system costs. The accelerator’s MLPerf-validated performance on GPT-3, computer vision and natural language models, plus upcoming software advances make Gaudi2 an extremely compelling price/performance alternative to Nvidia's H100.
On the CPU front, the deep learning training performance of 4th Gen Xeon processors with Intel AI engines demonstrated that customers can build with Xeon-based servers a single universal AI system for data pre-processing, model training and deployment to deliver the right combination of AI performance, efficiency, accuracy and scalability.
Gaudi2 delivered impressive time-to-train on GPT-31: 311 minutes on 384 accelerators.
Near-linear 95% scaling from 256 to 384 accelerators on GPT-3 model.
Excellent training results on computer vision — ResNet-50 8 accelerators and Unet3D 8 accelerators — and natural language processing models — BERT 8 and 64 accelerators.
Performance increases of 10% and 4%, respectively, for BERT and ResNet models as compared to the November submission, evidence of growing Gaudi2 software maturity.
Gaudi2 results were submitted “out of the box,” meaning customers can achieve comparable performance results when implementing Gaudi2 on premise or in the cloud.
"""

class TestBuildChatbotNormalCases(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        if not os.path.exists("./gaudi2.txt"):
            with open("./gaudi2.txt", "w") as file:
                file.write(gaudi2_content)

    @classmethod
    def tearDownClass(self) -> None:
        if os.path.exists("./gaudi2.txt"):
            os.remove("./gaudi2.txt")
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
    def test_enable_plugin_retrieval(self):
        # Test enabling Retrieval plugin
        config = PipelineConfig(model_name_or_path="facebook/opt-125m")
        config.plugins = {"retrieval": {"enable": True, "args": 
            {"input_path": "./gaudi2.txt", "persist_directory": "./output"}}}
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
