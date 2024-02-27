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
import re, os
from transformers import BitsAndBytesConfig
from transformers.utils import is_bitsandbytes_available
from intel_extension_for_transformers.neural_chat import build_chatbot
from intel_extension_for_transformers.neural_chat.config import PipelineConfig
from intel_extension_for_transformers.neural_chat.config import LoadingModelConfig
from intel_extension_for_transformers.transformers import WeightOnlyQuantConfig, MixedPrecisionConfig
from intel_extension_for_transformers.neural_chat.utils.common import get_device_type

class TestChatbotBuilder(unittest.TestCase):
    def setUp(self):
        self.device = get_device_type()
        return super().setUp()

    def tearDown(self) -> None:
        for filename in os.getcwd():
            if re.match(r'ne_.*_fp32.bin', filename) or re.match(r'ne_.*_q.bin', filename):
                file_path = os.path.join(os.getcwd(), filename)
                try:
                    os.remove(file_path)
                    print(f"Deleted file: {filename}")
                except OSError as e:
                    print(f"Error deleting file {filename}: {str(e)}")
        return super().tearDown()

    def test_build_chatbot_with_AMP(self):
        config = PipelineConfig(model_name_or_path="facebook/opt-125m",
                                optimization_config = MixedPrecisionConfig(
                                  dtype="float16" if torch.cuda.is_available() else "bfloat16"))
        chatbot = build_chatbot(config)
        self.assertIsNotNone(chatbot)
        response = chatbot.predict(query="Tell me about Intel Xeon Scalable Processors.")
        print(response)
        print("Inference with streaming mode.")
        for new_text in chatbot.predict_stream(query="Tell me about Intel Xeon Scalable Processors.")[0]:
            print(new_text, end="", flush=True)
        print("\n")
        self.assertIsNotNone(response)

    def test_build_chatbot_with_bitsandbytes_quant(self):
        if torch.cuda.is_available():
            os.system("pip install bitsandbytes")
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

    def test_build_chatbot_with_weight_only_quant(self):
        if self.device == "cpu":
            loading_config = LoadingModelConfig(use_neural_speed=False)
            config = PipelineConfig(model_name_or_path="facebook/opt-125m",
                optimization_config=WeightOnlyQuantConfig(compute_dtype="fp32", weight_dtype="int4_fullrange"),
                loading_config=loading_config
            )
            chatbot = build_chatbot(config)
            self.assertIsNotNone(chatbot)
            response = chatbot.predict(query="Tell me about Intel Xeon Scalable Processors.")
            print(response)
            self.assertIsNotNone(response)

if __name__ == '__main__':
    unittest.main()
