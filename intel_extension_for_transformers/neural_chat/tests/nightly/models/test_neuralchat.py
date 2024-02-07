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

from intel_extension_for_transformers.neural_chat.models.neuralchat_model import NeuralChatModel
from intel_extension_for_transformers.neural_chat import build_chatbot, PipelineConfig, GenerationConfig
from intel_extension_for_transformers.neural_chat.utils.common import get_device_type
import unittest

class TestNeuralChatModel(unittest.TestCase):
    def setUp(self):
        return super().setUp()

    def tearDown(self) -> None:
        return super().tearDown()

    def test_match(self):
        result = NeuralChatModel(model_name='/tf_dataset2/models/nlp_toolkit/neural-chat-7b-v1').match()
        self.assertTrue(result)

    def test_get_default_conv_template_v1(self):
        result = NeuralChatModel(
            model_name='/tf_dataset2/models/nlp_toolkit/neural-chat-7b-v1').get_default_conv_template()
        self.assertIn("<|im_start|>system", str(result))
        config = PipelineConfig(
            model_name_or_path="/tf_dataset2/models/nlp_toolkit/neural-chat-7b-v1")
        chatbot = build_chatbot(config=config)
        gen_config = GenerationConfig(top_k=1)
        result = chatbot.predict(query="Tell me about Intel Xeon Scalable Processors.", config=gen_config)
        print(result)
        self.assertIn('Intel® Xeon® Scalable processors', str(result))

    def test_get_default_conv_template_v2(self):
        result = NeuralChatModel(
            model_name='/tf_dataset2/models/nlp_toolkit/neural-chat-7b-v2').get_default_conv_template()
        self.assertIn("### System:", str(result))
        config = PipelineConfig(
            model_name_or_path="/tf_dataset2/models/nlp_toolkit/neural-chat-7b-v2")
        chatbot = build_chatbot(config=config)
        gen_config = GenerationConfig(top_k=1)
        result = chatbot.predict(query="Tell me about Intel Xeon Scalable Processors.", config=gen_config)
        self.assertIn('The Intel Xeon Scalable Processor', str(result))

    def test_get_default_conv_template_v3(self):
        result = NeuralChatModel(
            model_name='/tf_dataset2/models/nlp_toolkit/neural-chat-7b-v3').get_default_conv_template()
        self.assertIn("### System:", str(result))
        config = PipelineConfig(
            model_name_or_path="/tf_dataset2/models/nlp_toolkit/neural-chat-7b-v3")
        chatbot = build_chatbot(config=config)
        gen_config = GenerationConfig(top_k=1)
        result = chatbot.predict(query="Tell me about Intel Xeon Scalable Processors.", config=gen_config)
        self.assertIn('The Intel Xeon Scalable Processors', str(result))

    def test_get_default_conv_template_v3_1(self):
        result = NeuralChatModel(
            model_name='/tf_dataset2/models/nlp_toolkit/neural-chat-7b-v3-1').get_default_conv_template()
        self.assertIn("### System:", str(result))
        config = PipelineConfig(
            model_name_or_path="/tf_dataset2/models/nlp_toolkit/neural-chat-7b-v3-1")
        chatbot = build_chatbot(config=config)
        gen_config = GenerationConfig(top_k=1)
        result = chatbot.predict(query="Tell me about Intel Xeon Scalable Processors.", config=gen_config)
        self.assertIn('The Intel Xeon Scalable Processors', str(result))


if __name__ == "__main__":
    unittest.main()
