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

from intel_extension_for_transformers.neural_chat.models.chatglm_model import ChatGlmModel
from intel_extension_for_transformers.neural_chat.models.llama_model import LlamaModel
from intel_extension_for_transformers.neural_chat.models.mpt_model import MptModel
from intel_extension_for_transformers.neural_chat.models.neuralchat_model import NeuralChatModel
from intel_extension_for_transformers.neural_chat.models.mistral_model import MistralModel
from intel_extension_for_transformers.neural_chat import build_chatbot, PipelineConfig
from intel_extension_for_transformers.neural_chat.utils.common import get_device_type
import unittest

class TestChatGlmModel(unittest.TestCase):
    def setUp(self):
        self.device = get_device_type()
        return super().setUp()

    def tearDown(self) -> None:
        return super().tearDown()

    def test_match(self):
        result = ChatGlmModel().match(model_path='THUDM/chatglm2-6b')
        self.assertTrue(result)

    def test_get_default_conv_template(self):
        if self.device == "hpu":
            self.skipTest("ChatGLM is not supported on HPU.")
        result = ChatGlmModel().get_default_conv_template(model_path='THUDM/chatglm2-6b')
        self.assertIn('问', str(result))
        config = PipelineConfig(model_name_or_path="THUDM/chatglm2-6b")
        chatbot = build_chatbot(config=config)
        result = chatbot.predict("中国最大的城市是哪个？")
        print(result)
        self.assertIn('上海', str(result))

class TestLlamaModel(unittest.TestCase):
    def setUp(self):
        return super().setUp()

    def tearDown(self) -> None:
        return super().tearDown()

    def test_match(self):
        result = LlamaModel().match(model_path='meta-llama/Llama-2-7b-chat-hf')
        self.assertTrue(result)

    def test_get_default_conv_template(self):
        result = LlamaModel().get_default_conv_template(model_path='meta-llama/Llama-2-7b-chat-hf')
        self.assertIn("[INST] <<SYS>>", str(result))
        chatbot = build_chatbot()
        result = chatbot.predict("Tell me about Intel Xeon Scalable Processors.")
        print(result)
        self.assertIn('Intel Xeon Scalable Processors', str(result))

class TestMptModel(unittest.TestCase):
    def setUp(self):
        return super().setUp()

    def tearDown(self) -> None:
        return super().tearDown()

    def test_match(self):
        result = MptModel().match(model_path='mosaicml/mpt-7b-chat')
        self.assertTrue(result)

    def test_get_default_conv_template(self):
        result = MptModel().get_default_conv_template(model_path='mosaicml/mpt-7b-chat')
        self.assertIn("<|im_start|>system", str(result))
        config = PipelineConfig(model_name_or_path="mosaicml/mpt-7b-chat")
        chatbot = build_chatbot(config=config)
        result = chatbot.predict("Tell me about Intel Xeon Scalable Processors.")
        print(result)
        self.assertIn('Intel Xeon Scalable processors', str(result))

class TestNeuralChatModel(unittest.TestCase):
    def setUp(self):
        return super().setUp()

    def tearDown(self) -> None:
        return super().tearDown()

    def test_match(self):
        result = NeuralChatModel().match(model_path='Intel/neural-chat-7b-v1-1')
        self.assertTrue(result)

    def test_get_default_conv_template_v1(self):
        result = NeuralChatModel().get_default_conv_template(
            model_path='Intel/neural-chat-7b-v1-1')
        self.assertIn("<|im_start|>system", str(result))
        config = PipelineConfig(model_name_or_path="Intel/neural-chat-7b-v1-1")
        chatbot = build_chatbot(config=config)
        result = chatbot.predict("Tell me about Intel Xeon Scalable Processors.")
        print(result)
        self.assertIn('Intel® Xeon® Scalable processors', str(result))

    def test_get_default_conv_template_v2(self):
        result = NeuralChatModel().get_default_conv_template(model_path='Intel/neural-chat-7b-v2')
        self.assertIn("### System:", str(result))
        config = PipelineConfig(model_name_or_path="Intel/neural-chat-7b-v2")
        chatbot = build_chatbot(config=config)
        result = chatbot.predict("Tell me about Intel Xeon Scalable Processors.")
        self.assertIn('The Intel Xeon Scalable Processor', str(result))

    def test_get_default_conv_template_v3(self):
        result = NeuralChatModel().get_default_conv_template(model_path='Intel/neural-chat-7b-v3')
        self.assertIn("### System:", str(result))
        config = PipelineConfig(model_name_or_path="Intel/neural-chat-7b-v3")
        chatbot = build_chatbot(config=config)
        result = chatbot.predict("Tell me about Intel Xeon Scalable Processors.")
        self.assertIn('The Intel Xeon Scalable Processors', str(result))

    def test_get_default_conv_template_v3_1(self):
        result = NeuralChatModel().get_default_conv_template(model_path='Intel/neural-chat-7b-v3-1')
        self.assertIn("### System:", str(result))
        config = PipelineConfig(model_name_or_path="Intel/neural-chat-7b-v3-1")
        chatbot = build_chatbot(config=config)
        result = chatbot.predict("Tell me about Intel Xeon Scalable Processors.")
        self.assertIn('The Intel Xeon Scalable Processors', str(result))

    def test_get_default_conv_template_v3(self):
        result = NeuralChatModel().get_default_conv_template(model_path='Intel/neural-chat-7b-v3')
        self.assertIn("### System:", str(result))
        config = PipelineConfig(model_name_or_path="Intel/neural-chat-7b-v3")
        chatbot = build_chatbot(config=config)
        result = chatbot.predict("Tell me about Intel Xeon Scalable Processors.")
        print(result)
        self.assertIn('The Intel Xeon Scalable Processors', str(result))

    def test_get_default_conv_template_v3_1(self):
        result = NeuralChatModel().get_default_conv_template(model_path='Intel/neural-chat-7b-v3-1')
        self.assertIn("### System:", str(result))
        config = PipelineConfig(model_name_or_path="Intel/neural-chat-7b-v3-1")
        chatbot = build_chatbot(config=config)
        result = chatbot.predict("Tell me about Intel Xeon Scalable Processors.")
        print(result)
        self.assertIn('The Intel Xeon Scalable Processors', str(result))

class TestMistralModel(unittest.TestCase):
    def setUp(self):
        return super().setUp()

    def tearDown(self) -> None:
        return super().tearDown()

    def test_match(self):
        result = MistralModel().match(model_path='mistralai/Mistral-7B-v0.1')
        self.assertTrue(result)

    def test_get_default_conv_template(self):
        result = MistralModel().get_default_conv_template(model_path='mistralai/Mistral-7B-v0.1')
        self.assertIn("[INST]{system_message}", str(result))
        config = PipelineConfig(model_name_or_path="mistralai/Mistral-7B-v0.1")
        chatbot = build_chatbot(config=config)
        result = chatbot.predict("Tell me about Intel Xeon Scalable Processors.")
        print(result)
        self.assertIn('Intel Xeon Scalable processors', str(result))

class TestStarCoderModel(unittest.TestCase):
    def setUp(self):
        return super().setUp()

    def tearDown(self) -> None:
        return super().tearDown()

    def test_code_gen(self):
        config = PipelineConfig(model_name_or_path="bigcode/starcoder")
        chatbot = build_chatbot(config=config)
        result = chatbot.predict("def print_hello_world():")
        print(result)
        self.assertIn("Hello World", str(result))

class TestCodeLlamaModel(unittest.TestCase):
    def setUp(self):
        return super().setUp()

    def tearDown(self) -> None:
        return super().tearDown()

    def test_code_gen(self):
        config = PipelineConfig(model_name_or_path="codellama/CodeLlama-7b-hf")
        chatbot = build_chatbot(config=config)
        result = chatbot.predict("def print_hello_world():")
        print(result)
        self.assertIn("Hello World", str(result))

if __name__ == "__main__":
    unittest.main()