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

from intel_extension_for_transformers.neural_chat.pipeline.plugins.caching.cache import ChatCache
from intel_extension_for_transformers.neural_chat import build_chatbot, PipelineConfig
import unittest

from huggingface_hub import login
login("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")

class TestChatCache(unittest.TestCase):
    def setUp(self):
        return super().setUp()

    def tearDown(self) -> None:
        return super().tearDown()
    
    def test_chat_cache(self):
        cache_plugin = ChatCache()
        cache_plugin.init_similar_cache_from_config()

        prompt = "Tell me about Intel Xeon Scable Processors."
        config = PipelineConfig()
        chatbot = build_chatbot(config)
        response = chatbot.predict(prompt)
        cache_plugin.put(prompt, response)

        answer = cache_plugin.get(prompt)
        self.assertIn('Tell me about Intel Xeon Scable Processors.', str(answer))

        
if __name__ == "__main__":
    unittest.main()
