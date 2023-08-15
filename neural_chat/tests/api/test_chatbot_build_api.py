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
from neural_chat.chatbot import build_chatbot
from neural_chat.config import PipelineConfig, GenerationConfig

class TestChatbotBuilder(unittest.TestCase):
    def setUp(self):
        pass

    def test_build_chatbot_valid_config(self):
        config = PipelineConfig()
        chatbot = build_chatbot(config)
        self.assertIsNotNone(chatbot)
        response = chatbot.predict(query="Tell me about Intel Xeon Scalable Processors.")
        print(response)

    def test_build_chatbot_invalid_config(self):
        # Similar to the previous test, but with an invalid configuration
        pass

    def test_build_chatbot_retrieval(self):
        # Test the retrieval logic
        pass

    def test_build_chatbot_audio(self):
        # Test the audio logic
        pass

    # Add more tests for other components of the function

if __name__ == '__main__':
    unittest.main()
