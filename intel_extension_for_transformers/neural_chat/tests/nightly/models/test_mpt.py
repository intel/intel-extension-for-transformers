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

from intel_extension_for_transformers.neural_chat.models.mpt_model import MptModel
from intel_extension_for_transformers.neural_chat import build_chatbot, PipelineConfig
from intel_extension_for_transformers.neural_chat.utils.common import get_device_type
import unittest

class TestMptModel(unittest.TestCase):
    def setUp(self):
        return super().setUp()

    def tearDown(self) -> None:
        return super().tearDown()

    def test_match(self):
        result = MptModel().match(model_path='/tf_dataset2/models/nlp_toolkit/mpt-7b-chat')
        self.assertTrue(result)

    def test_get_default_conv_template(self):
        result = MptModel().get_default_conv_template(model_path='/tf_dataset2/models/nlp_toolkit/mpt-7b-chat')
        self.assertIn("<|im_start|>system", str(result))
        config = PipelineConfig(model_name_or_path="/tf_dataset2/models/nlp_toolkit/mpt-7b-chat")
        chatbot = build_chatbot(config=config)
        result = chatbot.predict("Tell me about Intel Xeon Scalable Processors.")
        print(result)
        self.assertIn('Intel Xeon Scalable processors', str(result))

if __name__ == "__main__":
    unittest.main()