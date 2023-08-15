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
import os
from neural_chat.chatbot import build_chatbot
from neural_chat.config import PipelineConfig, GenerationConfig

class TestChatbotBuilder(unittest.TestCase):
    def setUp(self):
        return super().setUp()

    def tearDown(self) -> None:
        return super().tearDown()

    def test_build_chatbot_with_default_config(self):
        config = PipelineConfig()
        chatbot = build_chatbot(config)
        self.assertIsNotNone(chatbot)
        response = chatbot.predict(query="Tell me about Intel Xeon Scalable Processors.")
        print(response)
        self.assertIsNotNone(response)

    def test_build_chatbot_with_customized_pipelinecfg(self):
        config = PipelineConfig(model_name_or_path="mosaicml/mpt-7b-chat", use_cache=True)
        chatbot = build_chatbot(config)
        self.assertIsNotNone(chatbot)
        response = chatbot.predict(query="Tell me about Intel Xeon Scalable Processors.")
        print(response)
        self.assertIsNotNone(response)

    def test_build_chatbot_with_customized_generationcfg(self):
        config = PipelineConfig()
        chatbot = build_chatbot(config)
        self.assertIsNotNone(chatbot)
        config = GenerationConfig(max_new_tokens=512, temperature=0.1)
        response = chatbot.predict(query="Tell me about Intel Xeon Scalable Processors.", config=config)
        print(response)
        self.assertIsNotNone(response)

    def test_build_chatbot_with_audio_plugin(self):
        config = PipelineConfig(audio_input=True, audio_input_path="../../assets/audio/pat.wav",
                                audio_output=True, audio_output_path="./response.wav",
                                audio_lang="english")
        chatbot = build_chatbot(config)
        self.assertIsNotNone(chatbot)
        response = chatbot.predict()
        self.assertIsNotNone(response)
        print("output audio path: ", response)
        self.assertFalse(os.path.exists(config.audio_output_path))

if __name__ == '__main__':
    unittest.main()
