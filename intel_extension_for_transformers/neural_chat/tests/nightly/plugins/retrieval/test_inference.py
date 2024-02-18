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
import torch
import unittest
import shutil
from intel_extension_for_transformers.neural_chat.chatbot import build_chatbot, optimize_model
from intel_extension_for_transformers.neural_chat.config import (
    PipelineConfig, GenerationConfig,
)
from intel_extension_for_transformers.neural_chat import plugins
from intel_extension_for_transformers.transformers import MixedPrecisionConfig
from transformers import AutoModelForCausalLM
from intel_extension_for_transformers.neural_chat.utils.common import get_device_type

class TestInference(unittest.TestCase):
    @classmethod
    def tearDownClass(self) -> None:
        if os.path.exists("output"):
            shutil.rmtree("output")
        if os.path.exists("check_append"):
            shutil.rmtree("check_append")

    def test_retrieval(self):
        plugins.retrieval.enable = True
        plugins.retrieval.args["input_path"] = "../assets/docs/"
        config = PipelineConfig(model_name_or_path="facebook/opt-125m",
                                plugins=plugins)
        chatbot = build_chatbot(config)
        response = chatbot.predict("Tell me about Intel Xeon Scalable Processors.")
        print(response)
        self.assertIsNotNone(response)
        plugins.retrieval.enable = False

    def test_retrieval_with_qdrant(self):
        plugins.retrieval.enable = True
        plugins.retrieval.args["input_path"] = "../assets/docs/"
        plugins.retrieval.args["vector_database"] = "Qdrant"
        config = PipelineConfig(model_name_or_path="facebook/opt-125m",
                                plugins=plugins)
        chatbot = build_chatbot(config)
        response = chatbot.predict("Tell me about Intel Xeon Scalable Processors.")
        print(response)
        self.assertIsNotNone(response)
        plugins.retrieval.enable = False

    def test_retrieval_append(self):
        plugins.retrieval.enable = True
        plugins.retrieval.args["append"] = True
        plugins.retrieval.args["input_path"] = "../assets/docs/"
        plugins.retrieval.args["persist_directory"] = "./check_append"
        config = PipelineConfig(model_name_or_path="facebook/opt-125m",
                                plugins=plugins)
        chatbot = build_chatbot(config)
        response = chatbot.predict("Tell me about Intel Xeon Scalable Processors.")
        print(response)
        self.assertIsNotNone(response)

        plugins.retrieval.args["append"] = False
        config = PipelineConfig(model_name_or_path="facebook/opt-125m",
                                plugins=plugins)
        chatbot = build_chatbot(config)
        response = chatbot.predict("Tell me about Intel Xeon Scalable Processors.")
        print(response)
        self.assertIsNotNone(response)
        plugins.retrieval.args["append"] = True
        plugins.retrieval.args["persist_directory"] = "./output"
        plugins.retrieval.enable = False

    def test_retrieval_append_with_qdrant(self):
        plugins.retrieval.enable = True
        plugins.retrieval.args["append"] = True
        plugins.retrieval.args["input_path"] = "../assets/docs/"
        plugins.retrieval.args["persist_directory"] = "./check_append"
        plugins.retrieval.args["vector_database"] = "Qdrant"
        config = PipelineConfig(model_name_or_path="facebook/opt-125m",
                                plugins=plugins)
        chatbot = build_chatbot(config)
        response = chatbot.predict("Tell me about Intel Xeon Scalable Processors.")
        print(response)
        self.assertIsNotNone(response)

        plugins.retrieval.args["append"] = False
        config = PipelineConfig(model_name_or_path="facebook/opt-125m",
                                plugins=plugins)
        chatbot = build_chatbot(config)
        response = chatbot.predict("Tell me about Intel Xeon Scalable Processors.")
        print(response)
        self.assertIsNotNone(response)
        plugins.retrieval.args["append"] = True
        plugins.retrieval.args["persist_directory"] = "./output"
        plugins.retrieval.enable = False

if __name__ == '__main__':
    unittest.main()
