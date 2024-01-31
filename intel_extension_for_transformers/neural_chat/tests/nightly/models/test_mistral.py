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

from intel_extension_for_transformers.neural_chat.models.mistral_model import MistralModel
from intel_extension_for_transformers.neural_chat import build_chatbot, PipelineConfig
from intel_extension_for_transformers.neural_chat.utils.common import get_device_type
import unittest

class TestMistralModel(unittest.TestCase):
    def setUp(self):
        return super().setUp()

    def tearDown(self) -> None:
        return super().tearDown()

    def test_match(self):
        result = MistralModel(model_name='/tf_dataset2/models/pytorch/Mistral-7B-v0.1').match()
        self.assertTrue(result)

    def test_get_default_conv_template(self):
        result = MistralModel(model_name='/tf_dataset2/models/pytorch/Mistral-7B-v0.1').get_default_conv_template()
        self.assertIn("[INST]{system_message}", str(result))
        config = PipelineConfig(
            model_name_or_path="/tf_dataset2/models/pytorch/Mistral-7B-v0.1")
        chatbot = build_chatbot(config=config)
        result = chatbot.predict("Tell me about Intel Xeon Scalable Processors.")
        print(result)
        self.assertIn('Intel Xeon Scalable processors', str(result))

if __name__ == "__main__":
    unittest.main()
