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
from transformers import BitsAndBytesConfig
from neural_chat.chatbot import build_chatbot
from neural_chat.config import PipelineConfig, OptimizationConfig, WeightOnlyQuantizationConfig

class TestChatbotBuilder(unittest.TestCase):
    def setUp(self):
        return super().setUp()

    def tearDown(self) -> None:
        return super().tearDown()

    def test_build_chatbot_with_weight_only_quant(self):
        config = PipelineConfig(
            optimization_config=OptimizationConfig(
                weight_only_quant_config=WeightOnlyQuantizationConfig()
            )
        )
        chatbot = build_chatbot(config)
        self.assertIsNotNone(chatbot)
        response = chatbot.predict(query="Tell me about Intel Xeon Scalable Processors.")
        print(response)
        self.assertIsNotNone(response)

    def test_build_chatbot_with_bitsandbytes_quant(self):
        config = PipelineConfig(
            device='cuda',
            optimization_config=OptimizationConfig(
                bitsandbytes_config=BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type='nf4',
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype="bfloat16"
                )
            )
        )
        chatbot = build_chatbot(config)
        self.assertIsNotNone(chatbot)
        response = chatbot.predict(query="Tell me about Intel Xeon Scalable Processors.")
        print(response)
        self.assertIsNotNone(response)

if __name__ == '__main__':
    unittest.main()
