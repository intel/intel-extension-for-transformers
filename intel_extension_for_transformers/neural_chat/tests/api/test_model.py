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
        text = r"Conversation(name='chatglm', system_template='{system_message}', system_message='', " + \
        " roles=('问', '答'), messages=[], offset=0, sep_style=<SeparatorStyle.CHATGLM: 8>, sep='\n', " + \
        "sep2=None, stop_str=None, stop_token_ids=None)"
        result = ChatGlmModel().get_default_conv_template(model_path='/models/chatglm-6b')
        self.assertEqual.__self__.maxDiff =None
        self.assertEqual(text, str(result))

    def test_get_default_conv_template_2(self):
        text = r"Conversation(name='chatglm2', system_template='{system_message}', system_message='', " + \
        "roles=('问', '答'), messages=[], offset=0, sep_style=<SeparatorStyle.CHATGLM: 8>, sep='\n\n', " + \
        "sep2=None, stop_str=None, stop_token_ids=None)"
        result = ChatGlmModel().get_default_conv_template(model_path='/models/chatglm2-6b-int4')
        self.assertEqual.__self__.maxDiff =None
        self.assertEqual(text, str(result))

class TestLlamaModel(unittest.TestCase):
    def setUp(self):
        return super().setUp()

    def tearDown(self) -> None:
        return super().tearDown()

    def test_match(self):
        result = LlamaModel().match(model_path='/models/Llama-2-7b-chat-hf')
        self.assertTrue(result)

    def test_get_default_conv_template(self):
        text = r"Conversation(name='llama-2', system_template='[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n', " + \
        "system_message='', roles=('[INST]', '[/INST]'), messages=[], offset=0, " + \
        "sep_style=<SeparatorStyle.LLAMA2: 7>, sep=' ', sep2=' </s><s>', stop_str=None, stop_token_ids=None)"
        result = LlamaModel().get_default_conv_template(model_path='/models/Llama-2-7b-chat-hf')
        self.assertEqual.__self__.maxDiff =None
        self.assertEqual(text, str(result))

class TestMptModel(unittest.TestCase):
    def setUp(self):
        return super().setUp()

    def tearDown(self) -> None:
        return super().tearDown()

    def test_match(self):
        result = MptModel().match(model_path='/models/mpt-7b-chat')
        self.assertTrue(result)

    def test_get_default_conv_template(self):
        text = r"Conversation(name='mpt-7b-chat', system_template='<|im_start|>system\n{system_message}', " + \
        "system_message='- You are a helpful assistant chatbot trained by MosaicML." + \
        "\n- You answer questions.\n- You are excited to be able to help the user, but will refuse to do " + \
        "anything that could be considered harmful to the user.\n- You are more than just an information source, " + \
        "you are also able to write poetry, short stories, and make jokes.', " + \
        "roles=('<|im_start|>user', '<|im_start|>assistant'), messages=[], offset=0, " + \
        "sep_style=<SeparatorStyle.CHATML: 9>, sep='<|im_end|>', sep2=None, stop_str=None, stop_token_ids=[50278, 0])"
        result = MptModel().get_default_conv_template(model_path='/models/mpt-7b-chat')
        self.assertEqual.__self__.maxDiff =None
        self.assertEqual(text, str(result))

class TestNeuralChatModel(unittest.TestCase):
    def setUp(self):
        return super().setUp()

    def tearDown(self) -> None:
        return super().tearDown()

    def test_match(self):
        result = NeuralChatModel().match(model_path='/models/neural-chat-7b-v1-1')
        self.assertTrue(result)

    def test_get_default_conv_template_v1(self):
        text = r"Conversation(name='neural-chat-7b-v1.1', system_template='<|im_start|>system\n{system_message}', " + \
        "system_message='- You are a helpful assistant chatbot trained by Intel."  + \
        "\n- You answer questions.\n- You are excited to be able to help the user, " + \
        "but will refuse to do anything that could be considered harmful to the user.\n- You are more " + \
        "than just an information source, you are also able to write poetry, short stories, and make jokes.', " + \
        "roles=('<|im_start|>user', '<|im_start|>assistant'), messages=[], offset=0, " + \
        "sep_style=<SeparatorStyle.CHATML: 9>, sep='<|im_end|>', sep2=None, stop_str=None, stop_token_ids=[50278, 0])"
        result = NeuralChatModel().get_default_conv_template(model_path='/models/neural-chat-7b-v1-1')
        self.assertEqual.__self__.maxDiff =None
        self.assertEqual(text, str(result))

    def test_get_default_conv_template_v2(self):
        text = r"Conversation(name='neural-chat-7b-v2', system_template='{system_message}', " + \
        "system_message='### System:\n- You are a helpful assistant chatbot trained by Intel.\n- " + \
        "You answer questions.\n- You are excited to be able to help the user, but will refuse to do anything " + \
        "that could be considered harmful to the user.\n- You are more than just an information source, " + \
        "you are also able to write poetry, short stories, and make jokes.</s>\n', " + \
        "roles=('### User:', '### Assistant:'), messages=[], offset=0, " + \
        "sep_style=<SeparatorStyle.NO_COLON_TWO: 5>, sep='\n', sep2='</s>', stop_str=None, stop_token_ids=None)"
        result = NeuralChatModel().get_default_conv_template(model_path='/models/neural-chat-7b-v2')
        self.assertEqual.__self__.maxDiff =None
        self.assertEqual(text, str(result))

if __name__ == "__main__":
    unittest.main()