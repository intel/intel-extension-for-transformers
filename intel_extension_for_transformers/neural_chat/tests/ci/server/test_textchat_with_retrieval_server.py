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
from fastapi import FastAPI
from fastapi.testclient import TestClient
from intel_extension_for_transformers.neural_chat import build_chatbot, plugins
from intel_extension_for_transformers.neural_chat import PipelineConfig
from intel_extension_for_transformers.neural_chat.server.restful.textchat_api import router
from intel_extension_for_transformers.neural_chat.server.restful.openai_protocol import ChatCompletionRequest
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

class UnitTest(unittest.TestCase):
    def setUp(self) -> None:
        config = PipelineConfig(model_name_or_path="facebook/opt-125m")
        chatbot = build_chatbot(config)
        router.set_chatbot(chatbot)

        self.oneapi_doc = "oneapi.txt"
        with open(self.oneapi_doc, "w") as file:
            file.write(oneapi_content)
        print(f"File created at {self.oneapi_doc}")

        plugins["retrieval"]['class'] = Agent_QA
        plugins["retrieval"]["instance"] = plugins["retrieval"]['class'](input_path="./oneapi.txt")

    def tearDown(self) -> None:
        # delete created resources
        import shutil
        if os.path.exists("./output"):
            shutil.rmtree("./output")
        if os.path.exists("./oneapi.txt"):
            os.remove("./oneapi.txt")

    def test_text_chat_with_retrieval(self):
        # Create a sample chat completion request object
        chat_request = ChatCompletionRequest(
            model="facebook/opt-125m",
            messages=[{"role": "user", "content": "Tell me about Intel Xeon processors."}],
        )
        response = client.post("/v1/chat/completions", json=chat_request.dict())
        assert response.status_code == 200

if __name__ == "__main__":
    unittest.main()
