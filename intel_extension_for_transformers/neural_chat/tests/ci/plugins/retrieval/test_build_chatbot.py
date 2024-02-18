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
from intel_extension_for_transformers.neural_chat import PipelineConfig
from intel_extension_for_transformers.neural_chat import plugins
from intel_extension_for_transformers.neural_chat.utils.common import get_device_type
from intel_extension_for_transformers.neural_chat.pipeline.plugins.retrieval.parser.parser import DocumentParser

gaudi2_content = """
Habana Gaudi2 and 4th Gen Intel Xeon Scalable processors deliver leading performance and optimal cost savings for AI training.
Today, MLCommons published results of its industry AI performance benchmark, MLPerf Training 3.0, in which both the Habana® Gaudi®2 deep learning accelerator and the 4th Gen Intel® Xeon® Scalable processor delivered impressive training results.
The latest MLPerf Training 3.0 results underscore the performance of Intel's products on an array of deep learning models. The maturity of Gaudi2-based software and systems for training was demonstrated at scale on the large language model, GPT-3. Gaudi2 is one of only two semiconductor solutions to submit performance results to the benchmark for LLM training of GPT-3.
Gaudi2 also provides substantially competitive cost advantages to customers, both in server and system costs. The accelerator’s MLPerf-validated performance on GPT-3, computer vision and natural language models, plus upcoming software advances make Gaudi2 an extremely compelling price/performance alternative to Nvidia's H100.
On the CPU front, the deep learning training performance of 4th Gen Xeon processors with Intel AI engines demonstrated that customers can build with Xeon-based servers a single universal AI system for data pre-processing, model training and deployment to deliver the right combination of AI performance, efficiency, accuracy and scalability.
Gaudi2 delivered impressive time-to-train on GPT-31: 311 minutes on 384 accelerators.
Near-linear 95% scaling from 256 to 384 accelerators on GPT-3 model.
Excellent training results on computer vision — ResNet-50 8 accelerators and Unet3D 8 accelerators — and natural language processing models — BERT 8 and 64 accelerators.
Performance increases of 10% and 4%, respectively, for BERT and ResNet models as compared to the November submission, evidence of growing Gaudi2 software maturity.
Gaudi2 results were submitted “out of the box,” meaning customers can achieve comparable performance results when implementing Gaudi2 on premise or in the cloud.
"""

# All UT cases use 'facebook/opt-125m' to reduce test time.
class TestBuildChatbot(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.device = get_device_type()
        if not os.path.exists("./gaudi2.txt"):
            with open("./gaudi2.txt", "w") as file:
                file.write(gaudi2_content)

    @classmethod
    def tearDownClass(self) -> None:
        if os.path.exists("./gaudi2.txt"):
            os.remove("./gaudi2.txt")
        if os.path.exists("./output"):
            shutil.rmtree("./output")

    @unittest.skipIf(get_device_type() != 'cpu', "Only run this test on CPU")
    def test_enable_plugin_retrieval(self):
        # Test enabling Retrieval plugin
        config = PipelineConfig(model_name_or_path="facebook/opt-125m")
        config.plugins = {"retrieval": {"enable": True, "args":
            {"input_path": "./gaudi2.txt", "persist_dir": "./output"}}}
        result = build_chatbot(config)
        self.assertIsNotNone(result)

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
            plugins.tts.enable = False
            plugins.retrieval.enable = True
            plugins.retrieval.args["input_path"] = "../../../README.md"
            plugins.retrieval.args["embedding_model"] = local_dir
            pipeline_config = PipelineConfig(model_name_or_path="facebook/opt-125m",
                                             plugins=plugins)
            chatbot = build_chatbot(pipeline_config)
            self.assertIsNotNone(chatbot)
            response = chatbot.predict(query="What is Intel extension for transformers?")
            self.assertIsNotNone(response)
            plugins.retrieval.enable = False

        # test local file
        _run_retrieval(local_dir="/tf_dataset2/inc-ut/gte-base")
        _run_retrieval(local_dir="/tf_dataset2/inc-ut/instructor-large")
        _run_retrieval(local_dir="/tf_dataset2/inc-ut/bge-base-en-v1.5")

if __name__ == "__main__":
    unittest.main()
