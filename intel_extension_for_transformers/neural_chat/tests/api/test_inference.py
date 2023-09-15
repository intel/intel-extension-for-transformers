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

import os
import unittest
from intel_extension_for_transformers.neural_chat.chatbot import build_chatbot, optimize_model
from intel_extension_for_transformers.neural_chat.config import (
    PipelineConfig, GenerationConfig, AMPConfig,
)
from intel_extension_for_transformers.neural_chat import plugins

class UnitTest(unittest.TestCase):
    def setUp(self):
        return super().setUp()

    def tearDown(self) -> None:
        return super().tearDown()

    def test_text_chat(self):
        config = PipelineConfig(model_name_or_path="facebook/opt-125m")
        chatbot = build_chatbot(config)
        response = chatbot.predict("Tell me about Intel Xeon Scalable Processors.")
        print(response)
        self.assertIsNotNone(response)

    def test_retrieval(self):
        plugins.retrieval.enable = True
        plugins.retrieval.args["input_path"] = "../../assets/docs/"
        config = PipelineConfig(model_name_or_path="facebook/opt-125m",
                                plugins=plugins)
        chatbot = build_chatbot(config)
        response = chatbot.predict("Tell me about Intel Xeon Scalable Processors.")
        print(response)
        self.assertIsNotNone(response)
    
    def test_retrieval_accuracy(self):
        plugins.retrieval.enable = True
        plugins.retrieval.args["input_path"] = "../../examples/retrieval/Xeon8480Processor.doxc"
        plugins.retrieval.args["persist_dir"] = "./test_for_correct/"
        config = PipelineConfig(model_name_or_path="facebook/opt-125m",
                                plugins=plugins)
        chatbot = build_chatbot(config)
        response = chatbot.predict("How many cores does the Intel® Xeon® Platinum 8480+ Processor have in total?")
        self.assertTrue("56" is in response)

    def test_voice_chat(self):
        plugins.tts.enable = True
        plugins.tts.args["output_audio_path"] = "./response.wav"
        pipeline_config = PipelineConfig(model_name_or_path="facebook/opt-125m",
                                         plugins=plugins)
        chatbot = build_chatbot(config=pipeline_config)
        gen_config = GenerationConfig(max_new_tokens=64)
        response = chatbot.predict(query="Nice to meet you!", config=gen_config)
        print(response)
        self.assertIsNotNone(response)
        print("output audio path: ", response)
        self.assertTrue(os.path.exists("./response.wav"))

    def test_quantization(self):
        config = AMPConfig()
        optimize_model(model="facebook/opt-125m", config=config)

if __name__ == '__main__':
    unittest.main()
