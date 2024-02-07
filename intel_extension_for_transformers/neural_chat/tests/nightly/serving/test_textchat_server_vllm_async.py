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
from fastapi import FastAPI
from fastapi.testclient import TestClient
from intel_extension_for_transformers.neural_chat import build_chatbot
from intel_extension_for_transformers.neural_chat import PipelineConfig
from intel_extension_for_transformers.neural_chat.config import ServingConfig, VllmEngineParams
from intel_extension_for_transformers.neural_chat.server.restful.textchat_api import router
from intel_extension_for_transformers.neural_chat.server.restful.openai_protocol import ChatCompletionRequest
import torch

app = FastAPI()
app.include_router(router)
client = TestClient(app)

class UnitTest(unittest.TestCase):
    def setUp(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("Only test this UT case on Nvidia GPU.")
        serving_config = ServingConfig(
                            framework="vllm", framework_config=VllmEngineParams(
                                use_async_engine=True,  # Start the Async vllm Engine
                                tensor_parallel_size = 1,
                                quantization=None,
                                gpu_memory_utilization=0.9,
                                swap_space=4,
                                enforce_eager=False,
                                max_context_len_to_capture=8192
                            ))
        config = PipelineConfig(model_name_or_path="facebook/opt-125m", serving_config=serving_config)
        chatbot = build_chatbot(config)
        router.set_chatbot(chatbot)

    def test_text_chat_async(self):
        # Create a sample chat completion request object
        chat_request = ChatCompletionRequest(
            model="facebook/opt-125m",
            messages=[{"role": "user", "content": "Tell me about Intel Xeon processors."}],
            temperature=0.8,
            top_p=0.95,
        )
        response =  client.post("/v1/chat/completions", json=chat_request.dict())
        assert response.status_code == 200

if __name__ == "__main__":
    unittest.main()
