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
from intel_extension_for_transformers.neural_chat import build_chatbot
from intel_extension_for_transformers.neural_chat import PipelineConfig
from intel_extension_for_transformers.neural_chat.config import LoadingModelConfig
from intel_extension_for_transformers.transformers import WeightOnlyQuantConfig
from intel_extension_for_transformers.neural_chat.utils.common import get_device_type
from intel_extension_for_transformers.neural_chat.server.restful.textchat_api import router
from intel_extension_for_transformers.neural_chat.server.restful.openai_protocol import ChatCompletionRequest, ChatCompletionResponse

app = FastAPI()
app.include_router(router)
client = TestClient(app)

class UnitTest(unittest.TestCase):
    def setUp(self) -> None:
        device = get_device_type()
        if device != "cpu":
            self.skipTest("Only test this UT case on Intel CPU.")
        loading_config = LoadingModelConfig(use_neural_speed=False)
        optimization_config = WeightOnlyQuantConfig(compute_dtype="int8", weight_dtype="int4_fullrange")
        config = PipelineConfig(model_name_or_path="facebook/opt-125m", device="cpu",
                                loading_config=loading_config,
                                optimization_config=optimization_config)
        chatbot = build_chatbot(config)
        router.set_chatbot(chatbot)

    def tearDown(self) -> None:
        # delete created resources
        import shutil
        if os.path.exists("./nc_workspace"):
            shutil.rmtree("./nc_workspace")
        return super().tearDown()

    def test_text_chat_with_woq_int4(self):
        # Create a sample chat completion request object
        chat_request = ChatCompletionRequest(
            model="facebook/opt-125m",
            messages=[{"role": "user", "content": "Tell me about Intel Xeon processors."}],
        )
        response = client.post("/v1/chat/completions", json=chat_request.dict())
        assert response.status_code == 200


if __name__ == "__main__":
    unittest.main()
