# !/usr/bin/env python
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
from unittest.mock import patch
from fastapi import FastAPI
from fastapi.testclient import TestClient
from intel_extension_for_transformers.neural_chat.server.restful.retrieval_api import router
from intel_extension_for_transformers.neural_chat import build_chatbot, plugins
from intel_extension_for_transformers.neural_chat import PipelineConfig
from intel_extension_for_transformers.neural_chat.pipeline.plugins.retrieval.retrieval_agent import Agent_QA

app = FastAPI()
app.include_router(router)
client = TestClient(app)

oneapi_content = """
This guide provides information about the Intel® oneAPI DPC++/C++ Compiler and runtime environment. This document is valid for version 2024.0 of the compilers.

The Intel® oneAPI DPC++/C++ Compiler is available as part of the Intel® oneAPI Base Toolkit, Intel® oneAPI HPC Toolkit, Intel® oneAPI IoT Toolkit, or as a standalone compiler.

Refer to the Intel® oneAPI DPC++/C++ Compiler product page and the Release Notes for more information about features, specifications, and downloads.

The compiler supports these key features:
Intel® oneAPI Level Zero: The Intel® oneAPI Level Zero (Level Zero) Application Programming Interface (API) provides direct-to-metal interfaces to offload accelerator devices.
OpenMP* Support: Compiler support for OpenMP 5.0 Version TR4 features and some OpenMP Version 5.1 features.
Pragmas: Information about directives to provide the compiler with instructions for specific tasks, including splitting large loops into smaller ones, enabling or disabling optimization for code, or offloading computation to the target.
Offload Support: Information about SYCL*, OpenMP, and parallel processing options you can use to affect optimization, code generation, and more.
Latest Standards: Use the latest standards including C++ 20, SYCL, and OpenMP 5.0 and 5.1 for GPU offload.
"""

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


class UnitTest(unittest.TestCase):
    def setUp(self) -> None:
        self.oneapi_doc = "oneapi.txt"
        self.gaudi2_doc = "gaudi2.txt"
        if not os.path.exists("./oneapi.txt"):
            with open(self.oneapi_doc, "w") as file:
                file.write(oneapi_content)
            print(f"File created at {self.oneapi_doc}")
        if not os.path.exists("./gaudi2.txt"):
            with open(self.gaudi2_doc, "w") as file:
                file.write(gaudi2_content)
            print(f"File created at {self.gaudi2_doc}")
        config = PipelineConfig(model_name_or_path="facebook/opt-125m")
        plugins.retrieval.enable = True
        plugins.retrieval.args["input_path"]="./oneapi.txt"
        chatbot = build_chatbot(config)
        router.set_chatbot(chatbot)

    @classmethod
    def tearDownClass(cls) -> None:
        # delete created resources
        import shutil
        if os.path.exists("./out_persist"):
            shutil.rmtree("./out_persist")
        if os.path.exists("./photoai_retrieval_docs"):
            shutil.rmtree("./photoai_retrieval_docs")
        if os.path.exists("./output"):
            shutil.rmtree("./output")
        if os.path.exists("./oneapi.txt"):
            os.remove("./oneapi.txt")
        if os.path.exists("./gaudi2.txt"):
            os.remove("./gaudi2.txt")

    async def test_create_new_kb_with_links(self):
        # Replace this with a sample link list you want to test with
        sample_link_list = {"link_list": ["https://www.ces.tech/"]}
        response = await client.post(
            "/v1/askdoc/upload_link",
            json=sample_link_list,
        )
        assert response.status_code == 200
        assert "knowledge_base_id" in response.json()

    async def test_append_existing_kb_with_links(self):
        # create gaudi2 knowledge base
        with open(self.gaudi2_doc, "rb") as file:
            response = await client.post(
                "/v1/askdoc/create",
                files={"file": ("./gaudi2.txt", file, "multipart/form-data")},
            )
        assert response.status_code == 200
        assert "knowledge_base_id" in response.json()
        gaudi2_kb_id = response.json()["knowledge_base_id"]
        sample_link_list = {"link_list": ["https://www.ces.tech/"]}
        response = client.post(
            "/v1/askdoc/upload_link",
            json={**sample_link_list, "knowledge_base_id": gaudi2_kb_id},
        )
        assert response.status_code == 200
        assert response.json()['status'] == True

    async def test_append_existing_kb(self):
        # create oneapi knowledge base
        with open(self.oneapi_doc, "rb") as file:
            response = await client.post(
                "/v1/askdoc/create",
                files={"file": ("./oneapi.txt", file, "multipart/form-data")},
            )
        assert response.status_code == 200
        assert "knowledge_base_id" in response.json()
        oneapi_kb_id = response.json()["knowledge_base_id"]
        with open("./gaudi2.txt", "rb") as file:
            response = client.post(
                "/v1/askdoc/append",
                files={"files": ("./gaudi2.txt", file, "multipart/form-data")},
                data={"knowledge_base_id": oneapi_kb_id},
            )
        assert response.status_code == 200
        assert "Succeed" in response.json()

    async def test_non_stream_chat(self):
        # create gaudi2 knowledge base
        with open(self.gaudi2_doc, "rb") as file:
            response = await client.post(
                "/v1/askdoc/create",
                files={"file": ("./gaudi2.txt", file, "multipart/form-data")},
            )
        assert response.status_code == 200
        assert "knowledge_base_id" in response.json()
        gaudi2_kb_id = response.json()["knowledge_base_id"]
        query_params = {
            "query": "How about the benchmark test of Habana Gaudi2?",
            "knowledge_base_id": gaudi2_kb_id,
            "stream": False,
            "max_new_tokens": 64,
            "return_link": False
        }
        response = client.post("/v1/askdoc/chat", json=query_params)
        assert response.status_code == 200

    async def test_stream_chat(self):
        # create gaudi2 knowledge base
        with open(self.gaudi2_doc, "rb") as file:
            response = await client.post(
                "/v1/askdoc/create",
                files={"file": ("./gaudi2.txt", file, "multipart/form-data")},
            )
        assert response.status_code == 200
        assert "knowledge_base_id" in response.json()
        gaudi2_kb_id = response.json()["knowledge_base_id"]
        query_params = {
            "query": "How about the benchmark test of Habana Gaudi2?",
            "knowledge_base_id": gaudi2_kb_id,
            "stream": True,
            "max_new_tokens": 64,
            "return_link": False
        }
        response = client.post("/v1/askdoc/chat", json=query_params)
        assert response.status_code == 200

    def test_save_feedback_to_db(self):
        feedback_data = {
            "question": "When is CES 2024?",
            "answer": "CES 2024 taking place Jan. 9-12, in Las Vegas.",
            "feedback": "1",  # Feedback can be '1' for like or '0' for dislike
            "comments": "Good answer."
        }
        # Mocking the MysqlDb class
        with patch('intel_extension_for_transformers.neural_chat.server.restful.retrieval_api.MysqlDb') as mock_mysql_db:
            mock_instance = mock_mysql_db.return_value
            mock_instance.insert.return_value = None
            response = client.post("/v1/askdoc/feedback", json=feedback_data)

        assert response.status_code == 200
        assert response.json() == "Succeed"

    def test_get_feedback_from_db(self):
        feedback_data = [
            {'feedback_id': 1, 'question': 'Question 1', 'answer': 'Answer 1', 'feedback_result': 1, 'feedback_time': '2023-01-01', "comments": "Comments 1"},
            {'feedback_id': 2, 'question': 'Question 2', 'answer': 'Answer 2', 'feedback_result': 0, 'feedback_time': '2023-01-02', "comments": "Comments 2"},
        ]

        # Mocking the MysqlDb class and fetch_all method
        with patch('intel_extension_for_transformers.neural_chat.server.restful.retrieval_api.MysqlDb') as mock_mysql_db:
            mock_instance = mock_mysql_db.return_value
            mock_instance.fetch_all.return_value = feedback_data

            response = client.get("/v1/askdoc/downloadFeedback")
            assert response.status_code == 200
            assert response.headers['content-type'] == 'text/csv; charset=utf-8'
            assert 'attachment;filename=feedback' in response.headers['content-disposition']

if __name__ == "__main__":
    unittest.main()
