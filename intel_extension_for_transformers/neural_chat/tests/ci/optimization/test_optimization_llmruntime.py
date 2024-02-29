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
import re, os
import torch
from intel_extension_for_transformers.neural_chat import build_chatbot
from intel_extension_for_transformers.neural_chat.config import PipelineConfig
from intel_extension_for_transformers.neural_chat.config import LoadingModelConfig
from intel_extension_for_transformers.transformers import WeightOnlyQuantConfig
from intel_extension_for_transformers.neural_chat.utils.common import get_device_type

class TestChatbotBuilder(unittest.TestCase):
    def setUp(self):
        self.device = get_device_type()
        if self.device != "cpu":
            self.skipTest("Skipping this test since LLM runtime optimization is for Intel CPU.")
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

    def test_build_chatbot_with_llm_runtime(self):
        loading_config = LoadingModelConfig(use_neural_speed=True)
        config = PipelineConfig(model_name_or_path="facebook/opt-125m",
            optimization_config=WeightOnlyQuantConfig(compute_dtype="int8", weight_dtype="int8"),
            loading_config=loading_config
        )
        chatbot = build_chatbot(config)
        self.assertIsNotNone(chatbot)
        response = chatbot.predict(query="Tell me about Intel Xeon Scalable Processors.")
        print(response)
        self.assertIsNotNone(response)


if __name__ == '__main__':
    unittest.main()
