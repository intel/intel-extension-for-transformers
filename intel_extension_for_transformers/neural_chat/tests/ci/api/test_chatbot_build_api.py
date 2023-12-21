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
import shutil
from intel_extension_for_transformers.neural_chat import build_chatbot
from intel_extension_for_transformers.neural_chat import PipelineConfig, GenerationConfig
from intel_extension_for_transformers.neural_chat import plugins
import torch
from intel_extension_for_transformers.neural_chat.utils.common import get_device_type

# All UT cases use 'facebook/opt-125m' to reduce test time.
class TestChatbotBuilder(unittest.TestCase):
    def setUp(self):
        self.device = get_device_type()
        return super().setUp()

    @classmethod
    def tearDownClass(cls) -> None:
        if os.path.exists("./output"):
            shutil.rmtree("./output")
        for filename in os.listdir("./"):
            if filename.endswith(".wav"):
                os.remove(filename)

    def test_build_chatbot_with_default_config(self):
        config = PipelineConfig(model_name_or_path="facebook/opt-125m")
        chatbot = build_chatbot(config)
        self.assertIsNotNone(chatbot)
        response = chatbot.predict(query="Tell me about Intel Xeon Scalable Processors.")
        print(response)
        self.assertIsNotNone(response)

    def test_build_chatbot_with_customized_pipelinecfg(self):
        config = PipelineConfig(model_name_or_path="facebook/opt-125m",
                                tokenizer_name_or_path="EleutherAI/gpt-neox-20b")
        chatbot = build_chatbot(config)
        self.assertIsNotNone(chatbot)
        response = chatbot.predict(query="Tell me about Intel Xeon Scalable Processors.")
        print(response)
        self.assertIsNotNone(response)

    def test_build_chatbot_with_customized_generationcfg(self):
        for task in ['completion', 'chat', 'summarization']:
            config = PipelineConfig(model_name_or_path="facebook/opt-125m")
            chatbot = build_chatbot(config)
            self.assertIsNotNone(chatbot)
            config = GenerationConfig(max_new_tokens=512, temperature=0.1, task=task)
            response = chatbot.predict(query="Tell me about Intel Xeon Scalable Processors.", config=config)
            print(response)
            self.assertIsNotNone(response)

    @unittest.skipIf(get_device_type() != 'cpu', "Only run this test on CPU")
    def test_build_chatbot_with_audio_plugin(self):
        plugins.tts.enable = True
        plugins.tts.args["device"] = "cuda" if torch.cuda.is_available() else "cpu"
        plugins.tts.args["output_audio_path"] = "./output_audio.wav"
        pipeline_config = PipelineConfig(model_name_or_path="facebook/opt-125m",
                                         plugins=plugins)
        chatbot = build_chatbot(pipeline_config)
        self.assertIsNotNone(chatbot)
        gen_config = GenerationConfig(max_new_tokens=64)
        audio_path = \
            "/intel-extension-for-transformers/intel_extension_for_transformers/neural_chat/assets/audio/sample.wav"
        if os.path.exists(audio_path):
            response = chatbot.predict(query=audio_path, config=gen_config)
        else:
            response = chatbot.predict(query="../assets/audio/sample.wav", config=gen_config)

        self.assertIsNotNone(response)
        print("output audio path: ", response)
        self.assertTrue(os.path.exists("./output_audio.wav"))

    def test_build_chatbot_with_safety_checker_plugin(self):
        plugins.safety_checker.enable = True
        pipeline_config = PipelineConfig(model_name_or_path="facebook/opt-125m",
                                         plugins=plugins)
        chatbot = build_chatbot(pipeline_config)
        self.assertIsNotNone(chatbot)
        response = chatbot.predict(query="蔡英文是谁？")
        print("response: ", response)
        self.assertTrue(response, "Your query contains sensitive words, please try another query.")

    def test_build_chatbot_with_retrieval_plugin(self):
        plugins.retrieval.enable = True
        plugins.retrieval.args["input_path"] = "../../../README.md"
        pipeline_config = PipelineConfig(model_name_or_path="facebook/opt-125m",
                                         plugins=plugins)
        chatbot = build_chatbot(pipeline_config)
        self.assertIsNotNone(chatbot)
        response = chatbot.predict(query="What is Intel extension for transformers?")
        self.assertIsNotNone(response)

        # test intel_extension_for_transformers.langchain.embeddings.HuggingFaceEmbeddings
        plugins.retrieval.enable = True
        plugins.retrieval.args["input_path"] = "../../../README.md"
        plugins.retrieval.args["embedding_model"] = "thenlper/gte-base"
        pipeline_config = PipelineConfig(model_name_or_path="facebook/opt-125m",
                                         plugins=plugins)
        chatbot = build_chatbot(pipeline_config)
        self.assertIsNotNone(chatbot)
        response = chatbot.predict(query="What is Intel extension for transformers?")
        self.assertIsNotNone(response)

        # test intel_extension_for_transformers.langchain.embeddings.HuggingFaceInstructEmbeddings
        plugins.retrieval.enable = True
        plugins.retrieval.args["input_path"] = "../../../README.md"
        plugins.retrieval.args["embedding_model"] = "hkunlp/instructor-large"
        pipeline_config = PipelineConfig(model_name_or_path="facebook/opt-125m",
                                         plugins=plugins)
        chatbot = build_chatbot(pipeline_config)
        self.assertIsNotNone(chatbot)
        response = chatbot.predict(query="What is Intel extension for transformers?")
        self.assertIsNotNone(response)
        plugins.retrieval.enable = False

    def test_build_chatbot_with_retrieval_plugin_bge_int8(self):
        if self.device != "cpu":
            self.skipTest("Only support Intel/bge-base-en-v1.5-sts-int8-static run on Intel CPU")
        plugins.retrieval.enable = True
        plugins.retrieval.args["input_path"] = "../../../README.md"
        # Intel/bge-base-en-v1.5-sts-int8-static is private now, so we need to load it from local.
        plugins.retrieval.args["embedding_model"] = \
            "/tf_dataset2/inc-ut/bge-base-en-v1.5-sts-int8-static"
        pipeline_config = PipelineConfig(model_name_or_path="facebook/opt-125m",
                                         plugins=plugins)
        chatbot = build_chatbot(pipeline_config)
        self.assertIsNotNone(chatbot)
        response = chatbot.predict(query="What is Intel extension for transformers?")
        self.assertIsNotNone(response)
        plugins.retrieval.enable = False
    
    def test_build_chatbot_with_retrieval_plugin_using_local_file(self):

        def _run_retrieval(local_dir):
            plugins.retrieval.enable = True
            plugins.retrieval.args["input_path"] = "../../../README.md"
            plugins.retrieval.args["embedding_model"] = local_dir
            pipeline_config = PipelineConfig(model_name_or_path="facebook/opt-125m",
                                             plugins=plugins)
            chatbot = build_chatbot(pipeline_config)
            self.assertIsNotNone(chatbot)
            response = chatbot.predict(query="What is Intel extension for transformers?")
            self.assertIsNotNone(response)

        # test local file
        _run_retrieval(local_dir="/tf_dataset2/inc-ut/gte-base")
        _run_retrieval(local_dir="/tf_dataset2/inc-ut/instructor-large")
        _run_retrieval(local_dir="/tf_dataset2/inc-ut/bge-base-en-v1.5")

        

if __name__ == '__main__':
    unittest.main()
