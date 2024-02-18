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

# All UT cases use 'facebook/opt-125m' to reduce test time.
class TestRAG(unittest.TestCase):
    def setUp(self):
        if os.path.exists("test_for_accuracy"):
            shutil.rmtree("test_for_accuracy", ignore_errors=True)
        return super().setUp()

    def tearDown(self) -> None:
        if os.path.exists("test_for_accuracy"):
            shutil.rmtree("test_for_accuracy", ignore_errors=True)
        return super().tearDown()

    def test_retrieval_accuracy(self):
        plugins.retrieval.enable = True
        plugins.retrieval.args["input_path"] = "../assets/docs/"
        plugins.retrieval.args["persist_directory"] = "./test_for_accuracy"
        plugins.retrieval.args["retrieval_type"] = 'default'
        config = PipelineConfig(model_name_or_path="facebook/opt-125m",
                                plugins=plugins)
        chatbot = build_chatbot(config)
        response = chatbot.predict("How many cores does the Intel Xeon Platinum 8480+ Processor have in total?")
        print(response)
        plugins.retrieval.args["persist_directory"] = "./output"
        self.assertIsNotNone(response)
        plugins.retrieval.enable = False

class TestChatbotBuilder_txt(unittest.TestCase):
    def setUp(self):
        if os.path.exists("test_txt"):
            shutil.rmtree("test_txt", ignore_errors=True)
        return super().setUp()

    def tearDown(self) -> None:
        if os.path.exists("test_txt"):
            shutil.rmtree("test_txt", ignore_errors=True)
        return super().tearDown()

    def test_retrieval_txt(self):
        plugins.retrieval.enable = True
        plugins.retrieval.args["input_path"] = "../assets/docs/sample.txt"
        plugins.retrieval.args["persist_directory"] = "./test_txt"
        plugins.retrieval.args["retrieval_type"] = 'default'
        config = PipelineConfig(model_name_or_path="facebook/opt-125m",
                                plugins=plugins)
        chatbot = build_chatbot(config)
        response = chatbot.predict("How many cores does the Intel Xeon Platinum 8480+ Processor have in total?")
        print(response)
        plugins.retrieval.args["persist_directory"] = "./output"
        self.assertIsNotNone(response)
        plugins.retrieval.enable = False

class TestChatbotBuilder_docx(unittest.TestCase):
    def setUp(self):
        if os.path.exists("test_docx"):
            shutil.rmtree("test_docx", ignore_errors=True)
        return super().setUp()

    def tearDown(self) -> None:
        if os.path.exists("test_docx"):
            shutil.rmtree("test_docx", ignore_errors=True)
        return super().tearDown()

    def test_retrieval_docx(self):
        plugins.retrieval.enable = True
        plugins.retrieval.args["input_path"] = "../assets/docs/sample.docx"
        plugins.retrieval.args["persist_directory"] = "./test_docx"
        plugins.retrieval.args["retrieval_type"] = 'default'
        config = PipelineConfig(model_name_or_path="facebook/opt-125m",
                                plugins=plugins)
        chatbot = build_chatbot(config)
        response = chatbot.predict("How many cores does the Intel Xeon Platinum 8480+ Processor have in total?")
        print(response)
        plugins.retrieval.args["persist_directory"] = "./output"
        self.assertIsNotNone(response)
        plugins.retrieval.enable = False

class TestChatbotBuilder_xlsx(unittest.TestCase):
    def setUp(self):
        if os.path.exists("test_xlsx"):
            shutil.rmtree("test_xlsx", ignore_errors=True)
        return super().setUp()

    def tearDown(self) -> None:
        if os.path.exists("test_xlsx"):
            shutil.rmtree("test_xlsx", ignore_errors=True)
        return super().tearDown()

    def test_retrieval_xlsx(self):
        plugins.retrieval.enable = True
        plugins.retrieval.args["input_path"] = "../assets/docs/sample.xlsx"
        plugins.retrieval.args["persist_directory"] = "./test_xlsx"
        plugins.retrieval.args["retrieval_type"] = 'default'
        config = PipelineConfig(model_name_or_path="facebook/opt-125m",
                                plugins=plugins)
        chatbot = build_chatbot(config)
        response = chatbot.predict("Who is the CEO of Intel?")
        print(response)
        plugins.retrieval.args["persist_directory"] = "./output"
        self.assertIsNotNone(response)
        plugins.retrieval.enable = False

class TestChatbotBuilder_xlsx_1(unittest.TestCase):
    def setUp(self):
        if os.path.exists("test_xlsx_1"):
            shutil.rmtree("test_xlsx_1", ignore_errors=True)
        return super().setUp()

    def tearDown(self) -> None:
        if os.path.exists("test_xlsx_1"):
            shutil.rmtree("test_xlsx_1", ignore_errors=True)
        return super().tearDown()

    def test_retrieval_xlsx_1(self):
        plugins.retrieval.enable = True
        plugins.retrieval.args["input_path"] = "../assets/docs/sample_1.xlsx"
        plugins.retrieval.args["persist_directory"] = "./test_xlsx_1"
        plugins.retrieval.args["retrieval_type"] = 'default'
        config = PipelineConfig(model_name_or_path="facebook/opt-125m",
                                plugins=plugins)
        chatbot = build_chatbot(config)
        response = chatbot.predict("Who is the CEO of Intel?")
        print(response)
        plugins.retrieval.args["persist_directory"] = "./output"
        self.assertIsNotNone(response)
        plugins.retrieval.enable = False

class TestChatbotBuilder_xlsx_2(unittest.TestCase):
    def setUp(self):
        if os.path.exists("test_xlsx_2"):
            shutil.rmtree("test_xlsx_2", ignore_errors=True)
        return super().setUp()

    def tearDown(self) -> None:
        if os.path.exists("test_xlsx_2"):
            shutil.rmtree("test_xlsx_2", ignore_errors=True)
        return super().tearDown()

    def test_retrieval_xlsx_2(self):
        plugins.retrieval.enable = True
        plugins.retrieval.args["input_path"] = "../assets/docs/sample_2.xlsx"
        plugins.retrieval.args["persist_directory"] = "./test_xlsx_2"
        plugins.retrieval.args["retrieval_type"] = 'default'
        config = PipelineConfig(model_name_or_path="facebook/opt-125m",
                                plugins=plugins)
        chatbot = build_chatbot(config)
        response = chatbot.predict("Who is the CEO of Intel?")
        print(response)
        plugins.retrieval.args["persist_directory"] = "./output"
        self.assertIsNotNone(response)
        plugins.retrieval.enable = False

class TestChatbotBuilder_jsonl(unittest.TestCase):
    def setUp(self):
        if os.path.exists("test_jsonl"):
            shutil.rmtree("test_jsonl", ignore_errors=True)
        return super().setUp()

    def tearDown(self) -> None:
        if os.path.exists("test_jsonl"):
            shutil.rmtree("test_jsonl", ignore_errors=True)
        return super().tearDown()

    def test_retrieval_jsonl(self):
        plugins.retrieval.enable = True
        plugins.retrieval.args["input_path"] = "../assets/docs/sample.jsonl"
        plugins.retrieval.args["persist_directory"] = "./test_jsonl"
        plugins.retrieval.args["retrieval_type"] = 'default'
        config = PipelineConfig(model_name_or_path="facebook/opt-125m",
                                plugins=plugins)
        chatbot = build_chatbot(config)
        response = chatbot.predict("What does this blog talk about?")
        print(response)
        plugins.retrieval.args["persist_directory"] = "./output"
        self.assertIsNotNone(response)
        plugins.retrieval.enable = False

class TestChatbotBuilder_csv(unittest.TestCase):
    def setUp(self):
        if os.path.exists("test_csv"):
            shutil.rmtree("test_csv", ignore_errors=True)
        return super().setUp()

    def tearDown(self) -> None:
        if os.path.exists("test_csv"):
            shutil.rmtree("test_csv", ignore_errors=True)
        return super().tearDown()

    def test_retrieval_csv(self):
        plugins.retrieval.enable = True
        plugins.retrieval.args["input_path"] = "../assets/docs/sample.csv"
        plugins.retrieval.args["persist_directory"] = "./test_csv"
        plugins.retrieval.args["retrieval_type"] = 'default'
        config = PipelineConfig(model_name_or_path="facebook/opt-125m",
                                plugins=plugins)
        chatbot = build_chatbot(config)
        response = chatbot.predict("Who is the CEO of Intel?")
        print(response)
        plugins.retrieval.args["persist_directory"] = "./output"
        self.assertIsNotNone(response)
        plugins.retrieval.enable = False

class TestChatbotBuilder_markdown(unittest.TestCase):
    def setUp(self):
        if os.path.exists("test_markdown"):
            shutil.rmtree("test_markdown", ignore_errors=True)
        return super().setUp()

    def tearDown(self) -> None:
        if os.path.exists("test_markdown"):
            shutil.rmtree("test_markdown", ignore_errors=True)
        return super().tearDown()

    def test_retrieval_markdown(self):
        plugins.retrieval.enable = True
        plugins.retrieval.args["input_path"] = "../assets/docs/sample.md"
        plugins.retrieval.args["persist_directory"] = "./test_markdown"
        plugins.retrieval.args["retrieval_type"] = 'default'
        config = PipelineConfig(model_name_or_path="facebook/opt-125m",
                                plugins=plugins)
        chatbot = build_chatbot(config)
        response = chatbot.predict("How many cores does the Intel Xeon Platinum 8480+ Processor have in total?")
        print(response)
        plugins.retrieval.args["persist_directory"] = "./output"
        self.assertIsNotNone(response)
        plugins.retrieval.enable = False

class TestChatbotBuilder_html(unittest.TestCase):
    def setUp(self):
        if os.path.exists("test_html"):
            shutil.rmtree("test_html", ignore_errors=True)
        return super().setUp()

    def tearDown(self) -> None:
        if os.path.exists("test_html"):
            shutil.rmtree("test_html", ignore_errors=True)
        return super().tearDown()

    def test_retrieval_html(self):
        plugins.retrieval.enable = True
        plugins.retrieval.args["input_path"] = "../assets/docs/sample.html"
        plugins.retrieval.args["persist_directory"] = "./test_html"
        plugins.retrieval.args["retrieval_type"] = 'default'
        config = PipelineConfig(model_name_or_path="facebook/opt-125m",
                                plugins=plugins)
        chatbot = build_chatbot(config)
        response = chatbot.predict("How many cores does the Intel Xeon Platinum 8480+ Processor have in total?")
        print(response)
        plugins.retrieval.args["persist_directory"] = "./output"
        self.assertIsNotNone(response)
        plugins.retrieval.enable = False

class TestChatbotBuilder_pdf(unittest.TestCase):
    def setUp(self):
        if os.path.exists("test_pdf"):
            shutil.rmtree("test_pdf", ignore_errors=True)
        return super().setUp()

    def tearDown(self) -> None:
        if os.path.exists("test_pdf"):
            shutil.rmtree("test_pdf", ignore_errors=True)
        return super().tearDown()

    def test_retrieval_pdf(self):
        plugins.retrieval.enable = True
        plugins.retrieval.args["input_path"] = "../assets/docs/sample.pdf"
        plugins.retrieval.args["persist_directory"] = "./test_pdf"
        plugins.retrieval.args["retrieval_type"] = 'default'
        config = PipelineConfig(model_name_or_path="facebook/opt-125m",
                                plugins=plugins)
        chatbot = build_chatbot(config)
        response = chatbot.predict("How many cores does the Intel Xeon Platinum 8480+ Processor have in total?")
        print(response)
        plugins.retrieval.args["persist_directory"] = "./output"
        self.assertIsNotNone(response)
        plugins.retrieval.enable = False

class TestChatbotBuilder_child_parent(unittest.TestCase):
    def setUp(self):
        if os.path.exists("test_rag"):
            shutil.rmtree("test_rag", ignore_errors=True)
        if os.path.exists("test_rag_child"):
            shutil.rmtree("test_rag_child", ignore_errors=True)
        return super().setUp()

    def tearDown(self) -> None:
        if os.path.exists("test_rag"):
            shutil.rmtree("test_rag", ignore_errors=True)
        if os.path.exists("test_rag_child"):
            shutil.rmtree("test_rag_child", ignore_errors=True)
        return super().tearDown()

    def test_retrieval_child_parent(self):
        plugins.retrieval.enable = True
        plugins.retrieval.args["input_path"] = "../assets/docs/sample.txt"
        plugins.retrieval.args["persist_directory"] = "./test_rag"
        plugins.retrieval.args["retrieval_type"] = "child_parent"
        config = PipelineConfig(model_name_or_path="facebook/opt-125m",
                                plugins=plugins)
        chatbot = build_chatbot(config)
        response = chatbot.predict("How many cores does the Intel Xeon Platinum 8480+ Processor have in total?")
        print(response)
        plugins.retrieval.args["persist_directory"] = "./output"
        plugins.retrieval.args["retrieval_type"] = 'default'
        self.assertIsNotNone(response)
        plugins.retrieval.enable = False

if __name__ == '__main__':
    unittest.main()
