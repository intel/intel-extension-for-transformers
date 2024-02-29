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
        result = ChatGlmModel(model_name='/tf_dataset2/models/pytorch/chatglm2-6b').match()
        self.assertTrue(result)

    def test_get_default_conv_template(self):
        if self.device == "hpu":
            self.skipTest("ChatGLM is not supported on HPU.")
        result = ChatGlmModel(model_name='/tf_dataset2/models/pytorch/chatglm2-6b').get_default_conv_template()
        self.assertIn('问', str(result))
        config = PipelineConfig(model_name_or_path="/tf_dataset2/models/pytorch/chatglm2-6b")
        chatbot = build_chatbot(config=config)
        result = chatbot.predict("中国最大的城市是哪个？")
        print(result)
        self.assertIn('上海', str(result))

if __name__ == "__main__":
    unittest.main()
