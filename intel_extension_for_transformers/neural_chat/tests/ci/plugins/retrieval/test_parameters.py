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

class TestResponseTemplate(unittest.TestCase):
    def setUp(self):
        if os.path.exists("./response_template"):
            shutil.rmtree("./response_template", ignore_errors=True)
        return super().setUp()

    def tearDown(self) -> None:
        if os.path.exists("./response_template"):
            shutil.rmtree("./response_template", ignore_errors=True)
        return super().tearDown()

    def test_response_template(self):
        plugins.retrieval.args = {}
        plugins.retrieval.enable = True
        plugins.retrieval.args["input_path"] = "../assets/docs/sample.txt"
        plugins.retrieval.args["persist_directory"] = "./response_template"
        plugins.retrieval.args["retrieval_type"] = 'default'
        response_template = "We cannot find suitable content to answer your query"
        plugins.retrieval.args["response_template"] = response_template
        plugins.retrieval.args["search_type"] = "similarity_score_threshold"
        plugins.retrieval.args["search_kwargs"] = {"k": 1, "score_threshold": 0.999}
        config = PipelineConfig(model_name_or_path="facebook/opt-125m",
                                plugins=plugins)
        chatbot = build_chatbot(config)
        response = chatbot.predict("How many QAs does the QA session have in total?")
        print(response)
        self.assertIsNotNone(response)
        plugins.retrieval.args = {}
        plugins.retrieval.enable = False

class TestAccuracyMode(unittest.TestCase):
    def setUp(self):
        if os.path.exists("./accuracy_mode"):
            shutil.rmtree("./accuracy_mode", ignore_errors=True)
        return super().setUp()

    def tearDown(self) -> None:
        if os.path.exists("./accuracy_mode"):
            shutil.rmtree("./accuracy_mode", ignore_errors=True)
        return super().tearDown()

    def test_accuracy_mode(self):
        plugins.retrieval.args = {}
        plugins.retrieval.enable = True
        plugins.retrieval.args["input_path"] = "../assets/docs/sample.txt"
        plugins.retrieval.args["persist_directory"] = "./accuracy_mode"
        plugins.retrieval.args["retrieval_type"] = 'default'
        plugins.retrieval.args["mode"] = 'accuracy'
        config = PipelineConfig(model_name_or_path="facebook/opt-125m",
                        plugins=plugins)
        chatbot = build_chatbot(config)
        response = chatbot.predict("How many cores does the Intel Xeon Platinum 8480+ Processor have in total?")
        print(response)
        self.assertIsNotNone(response)
        plugins.retrieval.args = {}
        plugins.retrieval.enable = False

class TestBM25Retriever(unittest.TestCase):
    def setUp(self):
        if os.path.exists("./bm25"):
            shutil.rmtree("./bm25", ignore_errors=True)
        return super().setUp()

    def tearDown(self) -> None:
        if os.path.exists("./bm25"):
            shutil.rmtree("./bm25", ignore_errors=True)
        return super().tearDown()

    def test_accuracy_mode(self):
        plugins.retrieval.args = {}
        plugins.retrieval.enable = True
        plugins.retrieval.args["input_path"] = "../assets/docs/sample.txt"
        plugins.retrieval.args["persist_directory"] = "./bm25"
        plugins.retrieval.args["retrieval_type"] = 'bm25'
        plugins.retrieval.args["mode"] = 'accuracy'
        config = PipelineConfig(model_name_or_path="facebook/opt-125m",
                        plugins=plugins)
        chatbot = build_chatbot(config)
        response = chatbot.predict("How many cores does the Intel Xeon Platinum 8480+ Processor have in total?")
        print(response)
        self.assertIsNotNone(response)
        plugins.retrieval.args = {}
        plugins.retrieval.enable = False

class TestRerank(unittest.TestCase):
    def setUp(self):
        if os.path.exists("./rerank"):
            shutil.rmtree("./rerank", ignore_errors=True)
        return super().setUp()

    def tearDown(self) -> None:
        if os.path.exists("./rerank"):
            shutil.rmtree("./rerank", ignore_errors=True)
        return super().tearDown()

    def test_general_mode(self):
        plugins.retrieval.args = {}
        plugins.retrieval.enable = True
        plugins.retrieval.args["input_path"] = "../assets/docs/sample.txt"
        plugins.retrieval.args["persist_directory"] = "./rerank"
        plugins.retrieval.args["retrieval_type"] = 'default'
        plugins.retrieval.args['enable_rerank'] = True
        plugins.retrieval.args['reranker_model'] = 'BAAI/bge-reranker-base'
        config = PipelineConfig(model_name_or_path="facebook/opt-125m",
                        plugins=plugins)
        chatbot = build_chatbot(config)
        response = chatbot.predict("How many cores does the Intel Xeon Platinum 8480+ Processor have in total?")
        print(response)
        self.assertIsNotNone(response)
        plugins.retrieval.args = {}
        plugins.retrieval.enable = False

class TestGeneralMode(unittest.TestCase):
    def setUp(self):
        if os.path.exists("./general_mode"):
            shutil.rmtree("./general_mode", ignore_errors=True)
        return super().setUp()

    def tearDown(self) -> None:
        if os.path.exists("./general_mode"):
            shutil.rmtree("./general_mode", ignore_errors=True)
        return super().tearDown()

    def test_general_mode(self):
        plugins.retrieval.args = {}
        plugins.retrieval.enable = True
        plugins.retrieval.args["input_path"] = "../assets/docs/sample.txt"
        plugins.retrieval.args["persist_directory"] = "./general_mode"
        plugins.retrieval.args["retrieval_type"] = 'default'
        plugins.retrieval.args["mode"] = 'general'
        config = PipelineConfig(model_name_or_path="facebook/opt-125m",
                        plugins=plugins)
        chatbot = build_chatbot(config)
        response = chatbot.predict("How many cores does the Intel Xeon Platinum 8480+ Processor have in total?")
        print(response)
        self.assertIsNotNone(response)
        plugins.retrieval.args = {}
        plugins.retrieval.enable = False

class TestSmallMaxChuckSize(unittest.TestCase):
    def setUp(self):
        if os.path.exists("./small_max_chuck_size"):
            shutil.rmtree("./small_max_chuck_size", ignore_errors=True)
        return super().setUp()

    def tearDown(self) -> None:
        if os.path.exists("./small_max_chuck_size"):
            shutil.rmtree("./small_max_chuck_size", ignore_errors=True)
        return super().tearDown()

    def test_small_max_chuck_size(self):
        plugins.retrieval.args = {}
        plugins.retrieval.enable = True
        plugins.retrieval.args["input_path"] = "../assets/docs/sample_1.txt"
        plugins.retrieval.args["persist_directory"] = "./small_max_chuck_size"
        plugins.retrieval.args["retrieval_type"] = 'default'
        plugins.retrieval.args["min_chuck_size"] = 5
        plugins.retrieval.args["max_chuck_size"] = 10
        config = PipelineConfig(model_name_or_path="facebook/opt-125m",
                        plugins=plugins)
        chatbot = build_chatbot(config)
        response = chatbot.predict("How many cores does the Intel Xeon Platinum 8480+ Processor have in total?")
        print(response)
        self.assertIsNotNone(response)
        plugins.retrieval.args = {}
        plugins.retrieval.enable = False

class TestLargeMaxChuckSize(unittest.TestCase):
    def setUp(self):
        if os.path.exists("./large_max_chuck_size"):
            shutil.rmtree("./large_max_chuck_size", ignore_errors=True)
        return super().setUp()

    def tearDown(self) -> None:
        if os.path.exists("./large_max_chuck_size"):
            shutil.rmtree("./large_max_chuck_size", ignore_errors=True)
        return super().tearDown()

    def test_large_max_chuck_size(self):
        plugins.retrieval.args = {}
        plugins.retrieval.enable = True
        plugins.retrieval.args["input_path"] = "../assets/docs/sample_1.txt"
        plugins.retrieval.args["persist_directory"] = "./large_max_chuck_size"
        plugins.retrieval.args["retrieval_type"] = 'default'
        plugins.retrieval.args["min_chuck_size"] = 5
        plugins.retrieval.args["max_chuck_size"] = 500
        config = PipelineConfig(model_name_or_path="facebook/opt-125m",
                        plugins=plugins)
        chatbot = build_chatbot(config)
        response = chatbot.predict("How many cores does the Intel Xeon Platinum 8480+ Processor have in total?")
        print(response)
        self.assertIsNotNone(response)
        plugins.retrieval.args = {}
        plugins.retrieval.enable = False

class TestSmallMinChuckSize(unittest.TestCase):
    def setUp(self):
        if os.path.exists("./small_min_chuck_size"):
            shutil.rmtree("./small_min_chuck_size", ignore_errors=True)
        return super().setUp()

    def tearDown(self) -> None:
        if os.path.exists("./small_min_chuck_size"):
            shutil.rmtree("./small_min_chuck_size", ignore_errors=True)
        return super().tearDown()

    def test_small_min_chuck_size(self):
        plugins.retrieval.args = {}
        plugins.retrieval.enable = True
        plugins.retrieval.args["input_path"] = "../assets/docs/sample_1.txt"
        plugins.retrieval.args["persist_directory"] = "./small_min_chuck_size"
        plugins.retrieval.args["retrieval_type"] = 'default'
        plugins.retrieval.args["min_chuck_size"] = 5
        plugins.retrieval.args["max_chuck_size"] = 150
        config = PipelineConfig(model_name_or_path="facebook/opt-125m",
                        plugins=plugins)
        chatbot = build_chatbot(config)
        response = chatbot.predict("How many cores does the Intel Xeon Platinum 8480+ Processor have in total?")
        print(response)
        self.assertIsNotNone(response)
        plugins.retrieval.args = {}
        plugins.retrieval.enable = False

class TestLargeMinChuckSize(unittest.TestCase):
    def setUp(self):
        if os.path.exists("./large_min_chuck_size"):
            shutil.rmtree("./large_min_chuck_size", ignore_errors=True)
        return super().setUp()

    def tearDown(self) -> None:
        if os.path.exists("./large_min_chuck_size"):
            shutil.rmtree("./large_min_chuck_size", ignore_errors=True)
        return super().tearDown()

    def test_large_min_chuck_size(self):
        plugins.retrieval.args = {}
        plugins.retrieval.enable = True
        plugins.retrieval.args["input_path"] = "../assets/docs/sample_1.txt"
        plugins.retrieval.args["persist_directory"] = "./large_min_chuck_size"
        plugins.retrieval.args["retrieval_type"] = 'default'
        plugins.retrieval.args["min_chuck_size"] = 100
        plugins.retrieval.args["max_chuck_size"] = 150
        plugins.retrieval.args["process"] = True
        plugins.retrieval.args["mode"] = 'general'
        config = PipelineConfig(model_name_or_path="facebook/opt-125m",
                        plugins=plugins)
        chatbot = build_chatbot(config)
        response = chatbot.predict("How many cores does the Intel Xeon Platinum 8480+ Processor have in total?")
        print(response)
        self.assertIsNotNone(response)
        plugins.retrieval.args = {}
        plugins.retrieval.enable = False

class TestTrueProcess(unittest.TestCase):
    def setUp(self):
        if os.path.exists("./true_process"):
            shutil.rmtree("./true_process", ignore_errors=True)
        return super().setUp()

    def tearDown(self) -> None:
        if os.path.exists("./true_process"):
            shutil.rmtree("./true_process", ignore_errors=True)
        return super().tearDown()

    def test_true_process(self):
        plugins.retrieval.args = {}
        plugins.retrieval.enable = True
        plugins.retrieval.args["input_path"] = "../assets/docs/sample_1.txt"
        plugins.retrieval.args["persist_directory"] = "./true_process"
        plugins.retrieval.args["retrieval_type"] = 'default'
        plugins.retrieval.args["min_chuck_size"] = 100
        plugins.retrieval.args["max_chuck_size"] = 150
        plugins.retrieval.args["process"] = True
        config = PipelineConfig(model_name_or_path="facebook/opt-125m",
                        plugins=plugins)
        chatbot = build_chatbot(config)
        response = chatbot.predict("How many cores does the Intel Xeon Platinum 8480+ Processor have in total?")
        print(response)
        self.assertIsNotNone(response)
        plugins.retrieval.args = {}
        plugins.retrieval.enable = False

class TestFalseProcess(unittest.TestCase):
    def setUp(self):
        if os.path.exists("./false_process"):
            shutil.rmtree("./false_process", ignore_errors=True)
        return super().setUp()

    def tearDown(self) -> None:
        if os.path.exists("./false_process"):
            shutil.rmtree("./false_process", ignore_errors=True)
        return super().tearDown()

    def test_false_process(self):
        plugins.retrieval.args = {}
        plugins.retrieval.enable = True
        plugins.retrieval.args["input_path"] = "../assets/docs/sample_1.txt"
        plugins.retrieval.args["persist_directory"] = "./false_process"
        plugins.retrieval.args["retrieval_type"] = 'default'
        plugins.retrieval.args["min_chuck_size"] = 10
        plugins.retrieval.args["max_chuck_size"] = 150
        plugins.retrieval.args["process"] = False
        config = PipelineConfig(model_name_or_path="facebook/opt-125m",
                        plugins=plugins)
        chatbot = build_chatbot(config)
        response = chatbot.predict("How many cores does the Intel Xeon Platinum 8480+ Processor have in total?")
        print(response)
        self.assertIsNotNone(response)
        plugins.retrieval.args = {}
        plugins.retrieval.enable = False

class TestMMRSearchTypeK1(unittest.TestCase):
    def setUp(self):
        if os.path.exists("./mmr_search_type_k_1"):
            shutil.rmtree("./mmr_search_type_k_1", ignore_errors=True)
        return super().setUp()

    def tearDown(self) -> None:
        if os.path.exists("./mmr_search_type_k_1"):
            shutil.rmtree("./mmr_search_type_k_1", ignore_errors=True)
        return super().tearDown()

    def test_mmr_search_type_k_1(self):
        plugins.retrieval.args = {}
        plugins.retrieval.enable = True
        plugins.retrieval.args["input_path"] = "../assets/docs/retrieve_multi_doc"
        plugins.retrieval.args["persist_directory"] = "./mmr_search_type_k_1"
        plugins.retrieval.args["retrieval_type"] = 'default'
        plugins.retrieval.args["search_type"] = "mmr"
        plugins.retrieval.args["search_kwargs"] = {"k": 1, "fetch_k": 1, "lambda_multi": 1}
        config = PipelineConfig(model_name_or_path="facebook/opt-125m",
                                plugins=plugins)
        chatbot = build_chatbot(config)
        response = chatbot.predict("Tell me about Intel Xeon Platinum 8480+ Processor.")
        print(response)
        self.assertIsNotNone(response)
        plugins.retrieval.args = {}
        plugins.retrieval.enable = False

class TestMMRSearchTypeK2(unittest.TestCase):
    def setUp(self):
        if os.path.exists("./mmr_search_type_k_2"):
            shutil.rmtree("./mmr_search_type_k_2", ignore_errors=True)
        return super().setUp()

    def tearDown(self) -> None:
        if os.path.exists("./mmr_search_type_k_2"):
            shutil.rmtree("./mmr_search_type_k_2", ignore_errors=True)
        return super().tearDown()

    def test_mmr_search_type_k_2(self):
        plugins.retrieval.args = {}
        plugins.retrieval.enable = True
        plugins.retrieval.args["input_path"] = "../assets/docs/retrieve_multi_doc"
        plugins.retrieval.args["persist_directory"] = "./mmr_search_type_k_2"
        plugins.retrieval.args["retrieval_type"] = 'default'
        plugins.retrieval.args["search_type"] = "mmr"
        plugins.retrieval.args["search_kwargs"] = {"k": 2, "fetch_k": 2, "lambda_multi": 1}
        config = PipelineConfig(model_name_or_path="facebook/opt-125m",
                                plugins=plugins)
        chatbot = build_chatbot(config)
        response = chatbot.predict("Tell me about Intel Xeon Platinum 8480+ Processor.")
        print(response)
        self.assertIsNotNone(response)
        plugins.retrieval.args = {}
        plugins.retrieval.enable = False

class TestMMRSearchTypeFetchK1(unittest.TestCase):
    def setUp(self):
        if os.path.exists("./mmr_search_type_fetch_k_1"):
            shutil.rmtree("./mmr_search_type_fetch_k_1", ignore_errors=True)
        return super().setUp()

    def tearDown(self) -> None:
        if os.path.exists("./mmr_search_type_fetch_k_1"):
            shutil.rmtree("./mmr_search_type_fetch_k_1", ignore_errors=True)
        return super().tearDown()

    def test_mmr_search_type_fetch_k_1(self):
        plugins.retrieval.args = {}
        plugins.retrieval.enable = True
        plugins.retrieval.args["input_path"] = "../assets/docs/retrieve_multi_doc"
        plugins.retrieval.args["persist_directory"] = "./mmr_search_type_fetch_k_1"
        plugins.retrieval.args["retrieval_type"] = 'default'
        plugins.retrieval.args["search_type"] = "mmr"
        plugins.retrieval.args["search_kwargs"] = {"k": 2, "fetch_k": 1, "lambda_multi": 1}
        config = PipelineConfig(model_name_or_path="facebook/opt-125m",
                                plugins=plugins)
        chatbot = build_chatbot(config)
        response = chatbot.predict("Tell me about Intel Xeon Platinum 8480+ Processor.")
        print(response)
        self.assertIsNotNone(response)
        plugins.retrieval.args = {}
        plugins.retrieval.enable = False

class TestMMRSearchTypeFetchK2(unittest.TestCase):
    def setUp(self):
        if os.path.exists("./mmr_search_type_fetch_k_2"):
            shutil.rmtree("./mmr_search_type_fetch_k_2", ignore_errors=True)
        return super().setUp()

    def tearDown(self) -> None:
        if os.path.exists("./mmr_search_type_fetch_k_2"):
            shutil.rmtree("./mmr_search_type_fetch_k_2", ignore_errors=True)
        return super().tearDown()

    def test_mmr_search_type_fetch_k_2(self):
        plugins.retrieval.args = {}
        plugins.retrieval.enable = True
        plugins.retrieval.args["input_path"] = "../assets/docs/retrieve_multi_doc"
        plugins.retrieval.args["persist_directory"] = "./mmr_search_type_fetch_k_2"
        plugins.retrieval.args["retrieval_type"] = 'default'
        plugins.retrieval.args["search_type"] = "mmr"
        plugins.retrieval.args["search_kwargs"] = {"k": 2, "fetch_k": 2, "lambda_multi": 1}
        config = PipelineConfig(model_name_or_path="facebook/opt-125m",
                                plugins=plugins)
        chatbot = build_chatbot(config)
        response = chatbot.predict("Tell me about Intel Xeon Platinum 8480+ Processor.")
        print(response)
        self.assertIsNotNone(response)
        plugins.retrieval.args = {}
        plugins.retrieval.enable = False

class TestMMRSearchTypeLowLambdaMulti(unittest.TestCase):
    def setUp(self):
        if os.path.exists("./mmr_search_type_low_lambda_multi"):
            shutil.rmtree("./mmr_search_type_low_lambda_multi", ignore_errors=True)
        return super().setUp()

    def tearDown(self) -> None:
        if os.path.exists("./mmr_search_type_low_lambda_multi"):
            shutil.rmtree("./mmr_search_type_low_lambda_multi", ignore_errors=True)
        return super().tearDown()

    def test_mmr_search_type_low_lambda_multi(self):
        plugins.retrieval.args = {}
        plugins.retrieval.enable = True
        plugins.retrieval.args["input_path"] = "../assets/docs/retrieve_multi_doc"
        plugins.retrieval.args["persist_directory"] = "./mmr_search_type_low_lambda_multi"
        plugins.retrieval.args["retrieval_type"] = 'default'
        plugins.retrieval.args["search_type"] = "mmr"
        plugins.retrieval.args["search_kwargs"] = {"k": 2, "fetch_k": 2, "lambda_multi": 0}
        config = PipelineConfig(model_name_or_path="facebook/opt-125m",
                                plugins=plugins)
        chatbot = build_chatbot(config)
        response = chatbot.predict("Tell me about Intel Xeon Platinum 8480+ Processor.")
        print(response)
        self.assertIsNotNone(response)
        plugins.retrieval.args = {}
        plugins.retrieval.enable = False

class TestMMRSearchTypeHighLambdaMulti(unittest.TestCase):
    def setUp(self):
        if os.path.exists("./mmr_search_type_high_lambda_multi"):
            shutil.rmtree("./mmr_search_type_high_lambda_multi", ignore_errors=True)
        return super().setUp()

    def tearDown(self) -> None:
        if os.path.exists("./mmr_search_type_high_lambda_multi"):
            shutil.rmtree("./mmr_search_type_high_lambda_multi", ignore_errors=True)
        return super().tearDown()

    def test_mmr_search_type_low_lambda_multi(self):
        plugins.retrieval.args = {}
        plugins.retrieval.enable = True
        plugins.retrieval.args["input_path"] = "../assets/docs/retrieve_multi_doc"
        plugins.retrieval.args["persist_directory"] = "./mmr_search_type_high_lambda_multi"
        plugins.retrieval.args["retrieval_type"] = 'default'
        plugins.retrieval.args["search_type"] = "mmr"
        plugins.retrieval.args["search_kwargs"] = {"k": 2, "fetch_k": 2, "lambda_multi": 1}
        config = PipelineConfig(model_name_or_path="facebook/opt-125m",
                                plugins=plugins)
        chatbot = build_chatbot(config)
        response = chatbot.predict("Tell me about Intel Xeon Platinum 8480+ Processor.")
        print(response)
        self.assertIsNotNone(response)
        plugins.retrieval.args = {}
        plugins.retrieval.enable = False

class TestSimilarityScoreThresholdSearchTypeLowThreshold(unittest.TestCase):
    def setUp(self):
        if os.path.exists("./similarity_score_threshold_search_type_low_threshold"):
            shutil.rmtree("./similarity_score_threshold_search_type_low_threshold", ignore_errors=True)
        return super().setUp()

    def tearDown(self) -> None:
        if os.path.exists("./similarity_score_threshold_search_type_low_threshold"):
            shutil.rmtree("./similarity_score_threshold_search_type_low_threshold", ignore_errors=True)
        return super().tearDown()

    def test_similarity_score_threshold_search_type_low_threshold(self):
        plugins.retrieval.args = {}
        plugins.retrieval.enable = True
        plugins.retrieval.args["input_path"] = "../assets/docs/sample.txt"
        plugins.retrieval.args["persist_directory"] = "./similarity_score_threshold_search_type_low_threshold"
        plugins.retrieval.args["retrieval_type"] = 'default'
        plugins.retrieval.args["search_type"] = "similarity_score_threshold"
        plugins.retrieval.args["search_kwargs"] = {"k": 1, "score_threshold": 0.001}
        config = PipelineConfig(model_name_or_path="facebook/opt-125m",
                                plugins=plugins)
        chatbot = build_chatbot(config)
        response = chatbot.predict("How many cores does the Intel Xeon Platinum 8480+ Processor have in total?")
        print(response)
        self.assertIsNotNone(response)
        plugins.retrieval.args = {}
        plugins.retrieval.enable = False

class TestSimilarityScoreThresholdSearchTypeHighThreshold(unittest.TestCase):
    def setUp(self):
        if os.path.exists("./similarity_score_threshold_search_type_high_threshold"):
            shutil.rmtree("./similarity_score_threshold_search_type_high_threshold", ignore_errors=True)
        return super().setUp()

    def tearDown(self) -> None:
        if os.path.exists("./similarity_score_threshold_search_type_high_threshold"):
            shutil.rmtree("./similarity_score_threshold_search_type_high_threshold", ignore_errors=True)
        return super().tearDown()

    def test_similarity_score_threshold_search_type_high_threshold(self):
        plugins.retrieval.args = {}
        plugins.retrieval.enable = True
        plugins.retrieval.args["input_path"] = "../assets/docs/sample.txt"
        plugins.retrieval.args["persist_directory"] = "./similarity_score_threshold_search_type_high_threshold"
        plugins.retrieval.args["retrieval_type"] = 'default'
        plugins.retrieval.args["search_type"] = "similarity_score_threshold"
        plugins.retrieval.args["search_kwargs"] = {"k": 1, "score_threshold": 0.999}
        config = PipelineConfig(model_name_or_path="facebook/opt-125m",
                                plugins=plugins)
        chatbot = build_chatbot(config)
        response = chatbot.predict("How many cores does the Intel Xeon Platinum 8480+ Processor have in total?")
        print(response)
        self.assertIsNotNone(response)
        plugins.retrieval.args = {}
        plugins.retrieval.enable = False

class TestSimilarityScoreThresholdSearchTypeK1(unittest.TestCase):
    def setUp(self):
        if os.path.exists("./similarity_score_threshold_search_type_k_1"):
            shutil.rmtree("./similarity_score_threshold_search_type_k_1", ignore_errors=True)
        return super().setUp()

    def tearDown(self) -> None:
        if os.path.exists("./similarity_score_threshold_search_type_k_1"):
            shutil.rmtree("./similarity_score_threshold_search_type_k_1", ignore_errors=True)
        return super().tearDown()

    def test_similarity_score_threshold_search_type_k_1(self):
        plugins.retrieval.args = {}
        plugins.retrieval.enable = True
        plugins.retrieval.args["input_path"] = "../assets/docs/retrieve_multi_doc"
        plugins.retrieval.args["persist_directory"] = "./similarity_score_threshold_search_type_k_1"
        plugins.retrieval.args["retrieval_type"] = 'default'
        plugins.retrieval.args["search_type"] = "similarity_score_threshold"
        plugins.retrieval.args["search_kwargs"] = {"k": 1, "score_threshold": 0.001}
        config = PipelineConfig(model_name_or_path="facebook/opt-125m",
                                plugins=plugins)
        chatbot = build_chatbot(config)
        response = chatbot.predict("Tell me about Intel Xeon Platinum 8480+ Processor.")
        print(response)
        self.assertIsNotNone(response)
        plugins.retrieval.args = {}
        plugins.retrieval.enable = False

class TestSimilarityScoreThresholdSearchTypeK2(unittest.TestCase):
    def setUp(self):
        if os.path.exists("./similarity_score_threshold_search_type_k_2"):
            shutil.rmtree("./similarity_score_threshold_search_type_k_2", ignore_errors=True)
        return super().setUp()

    def tearDown(self) -> None:
        if os.path.exists("./similarity_score_threshold_search_type_k_2"):
            shutil.rmtree("./similarity_score_threshold_search_type_k_2", ignore_errors=True)
        return super().tearDown()

    def test_similarity_score_threshold_search_type_k_2(self):
        plugins.retrieval.args = {}
        plugins.retrieval.enable = True
        plugins.retrieval.args["input_path"] = "../assets/docs/retrieve_multi_doc"
        plugins.retrieval.args["persist_directory"] = "./similarity_score_threshold_search_type_k_2"
        plugins.retrieval.args["retrieval_type"] = 'default'
        plugins.retrieval.args["search_type"] = "similarity_score_threshold"
        plugins.retrieval.args["search_kwargs"] = {"k": 2, "score_threshold": 0.001}
        config = PipelineConfig(model_name_or_path="facebook/opt-125m",
                                plugins=plugins)
        chatbot = build_chatbot(config)
        response = chatbot.predict("Tell me about Intel Xeon Platinum 8480+ Processor.")
        print(response)
        self.assertIsNotNone(response)
        plugins.retrieval.args = {}
        plugins.retrieval.enable = False

class TestSimilaritySearchTypeK1(unittest.TestCase):
    def setUp(self):
        if os.path.exists("./similarity_search_type_k_1"):
            shutil.rmtree("./similarity_search_type_k_1", ignore_errors=True)
        return super().setUp()

    def tearDown(self) -> None:
        if os.path.exists("./similarity_search_type_k_1"):
            shutil.rmtree("./similarity_search_type_k_1", ignore_errors=True)
        return super().tearDown()

    def test_similarity_search_type_k_1(self):
        plugins.retrieval.args = {}
        plugins.retrieval.enable = True
        plugins.retrieval.args["input_path"] = "../assets/docs/retrieve_multi_doc"
        plugins.retrieval.args["persist_directory"] = "./similarity_search_type_k_1"
        plugins.retrieval.args["retrieval_type"] = 'default'
        plugins.retrieval.args["search_type"] = "similarity"
        plugins.retrieval.args["search_kwargs"] = {"k": 1}
        config = PipelineConfig(model_name_or_path="facebook/opt-125m",
                                plugins=plugins)
        chatbot = build_chatbot(config)
        response = chatbot.predict("Tell me about Intel Xeon Platinum 8480+ Processor.")
        print(response)
        self.assertIsNotNone(response)
        plugins.retrieval.args = {}
        plugins.retrieval.enable = False

class TestSimilaritySearchTypeK2(unittest.TestCase):
    def setUp(self):
        if os.path.exists("./similarity_search_type_k_2"):
            shutil.rmtree("./similarity_search_type_k_2", ignore_errors=True)
        return super().setUp()

    def tearDown(self) -> None:
        if os.path.exists("./similarity_search_type_k_2"):
            shutil.rmtree("./similarity_search_type_k_2", ignore_errors=True)
        return super().tearDown()

    def test_similarity_search_type_k_2(self):
        plugins.retrieval.args = {}
        plugins.retrieval.enable = True
        plugins.retrieval.args["input_path"] = "../assets/docs/retrieve_multi_doc"
        plugins.retrieval.args["persist_directory"] = "./similarity_search_type_k_2"
        plugins.retrieval.args["retrieval_type"] = 'default'
        plugins.retrieval.args["search_type"] = "similarity"
        plugins.retrieval.args["search_kwargs"] = {"k": 2}
        config = PipelineConfig(model_name_or_path="facebook/opt-125m",
                                plugins=plugins)
        chatbot = build_chatbot(config)
        response = chatbot.predict("Tell me about Intel Xeon Platinum 8480+ Processor.")
        print(response)
        self.assertIsNotNone(response)
        plugins.retrieval.args = {}
        plugins.retrieval.enable = False

class TestEmbeddingPrecision(unittest.TestCase):
    def setUp(self):
        if os.path.exists("./embedding_precision_bf16"):
            shutil.rmtree("./embedding_precision_bf16", ignore_errors=True)
        if os.path.exists("./embedding_precision_fp32"):
            shutil.rmtree("./embedding_precision_fp32", ignore_errors=True)
        return super().setUp()

    def tearDown(self) -> None:
        if os.path.exists("./embedding_precision_bf16"):
            shutil.rmtree("./embedding_precision_bf16", ignore_errors=True)
        if os.path.exists("./embedding_precision_fp32"):
            shutil.rmtree("./embedding_precision_fp32", ignore_errors=True)
        return super().tearDown()

    def test_embedding_precision_bf16(self):
        plugins.retrieval.args = {}
        plugins.retrieval.enable = True
        plugins.retrieval.args["input_path"] = "../assets/docs/retrieve_multi_doc"
        plugins.retrieval.args["persist_directory"] = "./embedding_precision_bf16"
        plugins.retrieval.args["precision"] = 'bf16'
        config = PipelineConfig(model_name_or_path="facebook/opt-125m",
                                plugins=plugins)
        chatbot = build_chatbot(config)
        response = chatbot.predict("Tell me about Intel Xeon Platinum 8480+ Processor.")
        print(response)
        self.assertIsNotNone(response)
        plugins.retrieval.args = {}
        plugins.retrieval.enable = False

    def test_embedding_precision_fp32(self):
        plugins.retrieval.args = {}
        plugins.retrieval.enable = True
        plugins.retrieval.args["input_path"] = "../assets/docs/retrieve_multi_doc"
        plugins.retrieval.args["persist_directory"] = "./embedding_precision_fp32"
        plugins.retrieval.args["precision"] = 'fp32'
        config = PipelineConfig(model_name_or_path="facebook/opt-125m",
                                plugins=plugins)
        chatbot = build_chatbot(config)
        response = chatbot.predict("Tell me about Intel Xeon Platinum 8480+ Processor.")
        print(response)
        self.assertIsNotNone(response)
        plugins.retrieval.args = {}
        plugins.retrieval.enable = False

if __name__ == '__main__':
    unittest.main()
