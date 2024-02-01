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
import torch
from intel_extension_for_transformers.neural_chat import build_chatbot
from intel_extension_for_transformers.neural_chat import PipelineConfig
from intel_extension_for_transformers.neural_chat import plugins
from unittest.mock import patch
from intel_extension_for_transformers.neural_chat.errorcode import ErrorCodes
from intel_extension_for_transformers.neural_chat.utils.error_utils import get_latest_error

# All UT cases use 'facebook/opt-125m' to reduce test time.
class TestErrorCodeBuilder(unittest.TestCase):
    def setUp(self):
        return super().setUp()

    def tearDown(self) -> None:
        return super().tearDown()

    def test_build_chatbot_success(self):
        config = PipelineConfig(model_name_or_path="facebook/opt-125m")
        chatbot = build_chatbot(config)
        self.assertIsNot(chatbot, None)

    @patch('transformers.AutoModelForCausalLM.from_pretrained')
    def test_build_chatbot_out_of_storage(self, mock_from_pretrained):
        # Simulate out of storage scenario
        config = PipelineConfig(model_name_or_path="facebook/opt-125m")
        mock_from_pretrained.side_effect = Exception("No space left on device")
        result = build_chatbot(config)
        self.assertIsNone(result)
        assert get_latest_error() == ErrorCodes.ERROR_OUT_OF_STORAGE

    def test_build_chatbot_unsupported_device(self):
        config = PipelineConfig(model_name_or_path="facebook/opt-125m")
        config.device = "unsupported_device"
        chatbot = build_chatbot(config)
        assert chatbot == None
        assert get_latest_error() == ErrorCodes.ERROR_DEVICE_NOT_SUPPORTED

    def test_build_chatbot_out_of_gpu_memory(self):
        config = PipelineConfig(model_name_or_path="facebook/opt-125m")
        config.device = "cuda"
        if torch.cuda.is_available():
            # Mock torch.cuda.get_device_properties to return less GPU memory
            with patch('torch.cuda.get_device_properties') as mock_get_device_properties:
                mock_get_device_properties.return_value.total_memory = 8 * 1024 ** 3  # 8GB
                mock_get_device_properties.return_value.memory_allocated = 3 * 1024 ** 3  # 3GB
                chatbot = build_chatbot(config)
            assert chatbot == None
            assert get_latest_error() == ErrorCodes.ERROR_OUT_OF_MEMORY

    def test_build_chatbot_unsupported_model(self):
        plugins["unsupported_plugin"] = {
            'enable': True,
            'class': None,
            'args': {},
            'instance': None
        }
        config = PipelineConfig(model_name_or_path="unsupported_model", plugins=plugins)
        chatbot = build_chatbot(config)
        assert chatbot == None
        assert get_latest_error() == ErrorCodes.ERROR_MODEL_NOT_SUPPORTED

if __name__ == '__main__':
    unittest.main()
