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

import unittest
import torch
from transformers import BitsAndBytesConfig
from transformers.utils.bitsandbytes import is_bitsandbytes_available
from intel_extension_for_transformers.neural_chat import build_chatbot
from intel_extension_for_transformers.neural_chat.config import PipelineConfig, WeightOnlyQuantizationConfig


class TestChatbotBuilder(unittest.TestCase):
    def setUp(self):
        return super().setUp()

    def tearDown(self) -> None:
        return super().tearDown()

    def test_build_chatbot_with_AMP(self):
        config = PipelineConfig(model_name_or_path="facebook/opt-125m")
        chatbot = build_chatbot(config)
        self.assertIsNotNone(chatbot)
        response = chatbot.predict(query="Tell me about Intel Xeon Scalable Processors.")
        print(response)
        print("Inference with streaming mode.")
        for new_text in chatbot.predict_stream(query="Tell me about Intel Xeon Scalable Processors."):
            print(new_text, end="", flush=True)
        print("\n")
        self.assertIsNotNone(response)

    def test_build_chatbot_with_weight_only_quant(self):
        config = PipelineConfig(model_name_or_path="facebook/opt-125m",
            optimization_config=WeightOnlyQuantizationConfig()
        )
        chatbot = build_chatbot(config)
        self.assertIsNotNone(chatbot)
        response = chatbot.predict(query="Tell me about Intel Xeon Scalable Processors.")
        print(response)
        self.assertIsNotNone(response)

    def test_build_chatbot_with_bitsandbytes_quant(self):
        if is_bitsandbytes_available() and torch.cuda.is_available():
            config = PipelineConfig(
                model_name_or_path="facebook/opt-125m",
                device='cuda',
                optimization_config=BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type='nf4',
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_compute_dtype="bfloat16"
                )
            )
            chatbot = build_chatbot(config)
            self.assertIsNotNone(chatbot)
            response = chatbot.predict(query="Tell me about Intel Xeon Scalable Processors.")
            print(response)
            self.assertIsNotNone(response)

if __name__ == '__main__':
    unittest.main()
