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
from intel_extension_for_transformers.neural_chat import build_chatbot
from intel_extension_for_transformers.neural_chat import PipelineConfig, GenerationConfig
from intel_extension_for_transformers.neural_chat import plugins
from unittest.mock import Mock, patch
from intel_extension_for_transformers.neural_chat.constants import ResponseCodes

# All UT cases use 'facebook/opt-125m' to reduce test time.
class TestErrorCodeBuilder(unittest.TestCase):
    def setUp(self):
        return super().setUp()

    def tearDown(self) -> None:
        return super().tearDown()

    def test_build_chatbot_success():
        config = PipelineConfig()
        chatbot = build_chatbot(config)
        unittest.assertIsNotInstance(chatbot, str)

    def test_build_chatbot_out_of_memory():
        config = PipelineConfig()
        # Mock psutil.virtual_memory().available to return less available memory
        with patch('psutil.virtual_memory') as mock_virtual_memory:
            mock_virtual_memory.return_value.available = 7 * 1024 ** 3  # 7GB
            result = build_chatbot(config)
        breakpoint()
        assert result == ResponseCodes.ERROR_OUT_OF_MEMORY