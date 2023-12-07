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
from intel_extension_for_transformers.neural_chat import build_chatbot
from intel_extension_for_transformers.neural_chat import PipelineConfig
from intel_extension_for_transformers.neural_chat import plugins

# All UT cases use 'facebook/opt-125m' to reduce test time.
class TestChatbotBuilder(unittest.TestCase):
    def setUp(self):
        return super().setUp()

    def tearDown(self) -> None:
        return super().tearDown()
    
    def test_retrieval_accuracy(self):
        plugins.retrieval.enable = True
        plugins.retrieval.args["input_path"] = "../assets/docs/sample.txt"
        plugins.retrieval.args["persist_dir"] = "./test_for_accuracy"
        config = PipelineConfig(model_name_or_path="facebook/opt-125m",
                                plugins=plugins)
        chatbot = build_chatbot(config)
        response = chatbot.predict("How many cores does the Intel Xeon Platinum 8480+ Processor have in total?")
        print(response)
        plugins.retrieval.args["persist_dir"] = "./output"
        self.assertIsNotNone(response)
        plugins.retrieval.enable = False


if __name__ == '__main__':
    unittest.main()
