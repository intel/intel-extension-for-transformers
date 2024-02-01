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

import os
import unittest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import patch
from intel_extension_for_transformers.neural_chat.server.restful.tgi_api import router


app = FastAPI()
app.include_router(router)
client = TestClient(app)


@patch('intel_extension_for_transformers.neural_chat.server.restful.tgi_api.InferenceClient.text_generation')
class TestTGI(unittest.TestCase):

    def setUp(self):
        return super().setUp()

    def tearDown(self) -> None:
        return super().tearDown()

    def test_tgi_root(self, mock_text_generation):
        mock_text_generation.return_value = "Mocked text generation result."
        request_data = {
            "inputs": "Test text generation inputs.",
            "parameters": {"max_new_tokens":10},
            "stream": False
        }
        response = client.post("/v1/tgi", json=request_data)

        mock_text_generation.assert_called_once_with(
            prompt="Test text generation inputs.",
            best_of=1,
            do_sample=True,
            max_new_tokens=10,
            repetition_penalty=1.03,
            temperature=0.5,
            top_k=10,
            top_p=0.95,
            typical_p=0.95,
            stream=False
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), "Mocked text generation result.")

    def test_tgi_generate(self, mock_text_generation):
        mock_text_generation.return_value = "Mocked text generation result."
        request_data = {
            "inputs": "Test text generation inputs.",
            "parameters": {"max_new_tokens":10}
        }
        response = client.post("/v1/tgi/generate", json=request_data)

        mock_text_generation.assert_called_once_with(
            prompt="Test text generation inputs.",
            best_of=1,
            do_sample=True,
            max_new_tokens=10,
            repetition_penalty=1.03,
            temperature=0.5,
            top_k=10,
            top_p=0.95,
            typical_p=0.95,
            stream=False
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), "Mocked text generation result.")

    def test_tgi_generate_stream(self, mock_text_generation):
        def mock_generator():
            yield "Mocked text generation output part 1."
            yield "Mocked text generation output part 2."
        mock_text_generation.return_value = mock_generator()
        request_data = {
            "inputs": "Test text generation inputs.",
            "parameters": {"max_new_tokens":10}
        }
        response = client.post("/v1/tgi/generate_stream", json=request_data)

        self.assertEqual(response.status_code, 200)
        lines = response.content.decode()
        self.assertIn("Mocked text generation output part 1.", lines)
        self.assertIn("Mocked text generation output part 2.", lines)


if __name__ == "__main__":
    unittest.main()
