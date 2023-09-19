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
        # plugins.retrieval.args["input_path"] = "../../assets/docs/"
        plugins.retrieval.args["input_path"] = 'sample.txt'
        config = PipelineConfig(model_name_or_path="facebook/opt-125m",
                                plugins=plugins)
        chatbot = build_chatbot(config)
        response = chatbot.predict("Tell me about Intel Xeon Scalable Processors.")
        print(response)
        self.assertIsNotNone(response)
        plugins.retrieval.enable = False
        
    def test_retrieval_override(self):
        plugins.retrieval.enable = True
        # plugins.retrieval.args["input_path"] = "../../assets/docs/"
        plugins.retrieval.args["input_path"] = 'sample.txt'
        config = PipelineConfig(model_name_or_path="facebook/opt-125m",
                                plugins=plugins)
        chatbot = build_chatbot(config)
        response1 = chatbot.predict("Tell me about Intel Xeon Scalable Processors.")
        print(response1)
        self.assertIsNotNone(response1)
        plugins.retrieval.override = False
        config = PipelineConfig(model_name_or_path="facebook/opt-125m",
                                plugins=plugins)
        response2 = chatbot.predict("Tell me about Intel Xeon Scalable Processors.")
        print(response2)
        self.assertIsNotNone(response2)
        plugins.retrieval.override = True
        plugins.retrieval.enable = False

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

    def test_text_chat_stream(self):
        config = PipelineConfig(model_name_or_path="facebook/opt-125m")
        chatbot = build_chatbot(config)
        stream_text = ""
        for text in chatbot.predict_stream("Tell me about Intel Xeon Scalable Processors."):
            stream_text += text
            print(text)
        self.assertIsNotNone(stream_text)

if __name__ == '__main__':
    unittest.main()
    
    suite = unittest.TestSuite()
    suite.addTest(TestLogin('test_text_chat'))
    suite.addTest(TestLogin('test_retrieval'))
    suite.addTest(TestLogin('test_retrieval_override'))
    suite.addTest(TestLogin('test_voice_chat'))
    suite.addTest(TestLogin('test_quantization'))
    suite.addTest(TestLogin('test_text_chat_stream'))
    
    runner = unittest.TextTestRunner()
    runner.run(suite)
    
    
