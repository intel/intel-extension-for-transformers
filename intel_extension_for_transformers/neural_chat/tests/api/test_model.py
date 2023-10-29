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
import unittest

class TestChatGlmModel(unittest.TestCase):
    def setUp(self):
        return super().setUp()

    def tearDown(self) -> None:
        return super().tearDown()

    def test_match(self):
        result = ChatGlmModel().match(model_path='/models/chatglm2-6b-int4')
        self.assertTrue(result)

    def test_get_default_conv_template_1(self):
        result = ChatGlmModel().get_default_conv_template(model_path='/models/chatglm-6b')
        self.assertIn('问', str(result))

    def test_get_default_conv_template_2(self):
        result = ChatGlmModel().get_default_conv_template(model_path='/models/chatglm2-6b-int4')
        self.assertIn('问', str(result))

class TestLlamaModel(unittest.TestCase):
    def setUp(self):
        return super().setUp()

    def tearDown(self) -> None:
        return super().tearDown()

    def test_match(self):
        result = LlamaModel().match(model_path='/models/Llama-2-7b-chat-hf')
        self.assertTrue(result)

    def test_get_default_conv_template(self):
        result = LlamaModel().get_default_conv_template(model_path='/models/Llama-2-7b-chat-hf')
        self.assertIn("[INST] <<SYS>>", str(result))

class TestMptModel(unittest.TestCase):
    def setUp(self):
        return super().setUp()

    def tearDown(self) -> None:
        return super().tearDown()

    def test_match(self):
        result = MptModel().match(model_path='/models/mpt-7b-chat')
        self.assertTrue(result)

    def test_get_default_conv_template(self):
        result = MptModel().get_default_conv_template(model_path='/models/mpt-7b-chat')
        self.assertIn("<|im_start|>system", str(result))

class TestNeuralChatModel(unittest.TestCase):
    def setUp(self):
        return super().setUp()

    def tearDown(self) -> None:
        return super().tearDown()

    def test_match(self):
        result = NeuralChatModel().match(model_path='/models/neural-chat-7b-v1-1')
        self.assertTrue(result)

    def test_get_default_conv_template_v1(self):
        result = NeuralChatModel().get_default_conv_template(model_path='/models/neural-chat-7b-v1-1')
        self.assertIn("<|im_start|>system", str(result))

    def test_get_default_conv_template_v2(self):
        result = NeuralChatModel().get_default_conv_template(model_path='/models/neural-chat-7b-v2')
        self.assertIn("### System:", str(result))

if __name__ == "__main__":
    unittest.main()