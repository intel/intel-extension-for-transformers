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
from unittest.mock import MagicMock, patch
from intel_extension_for_transformers.neural_chat import build_chatbot, PipelineConfig
from intel_extension_for_transformers.transformers import MixedPrecisionConfig
from intel_extension_for_transformers.neural_chat.utils.common import get_device_type
class TestBuildChatbotExceptions(unittest.TestCase):

    @unittest.skipIf(get_device_type() != 'cpu', "Only run this test on CPU")
    def test_out_of_storage(self):
        # Simulate out of storage scenario
        config = PipelineConfig(model_name_or_path="facebook/opt-125m")
        with unittest.mock.patch('psutil.disk_usage') as mock_disk_usage:
            mock_disk_usage.return_value.free = 0  # Set free storage to 0
            result = build_chatbot(config)
            self.assertIsNone(result)

    @unittest.skipIf(get_device_type() != 'cpu', "Only run this test on CPU")
    def test_invalid_device(self):
        # Test when an invalid device is provided
        config = PipelineConfig(model_name_or_path="facebook/opt-125m")
        config.device = "invalid_device"
        result = build_chatbot(config)
        self.assertIsNone(result)

    @unittest.skipIf(get_device_type() != 'cpu', "Only run this test on CPU")
    def test_cuda_device_not_found(self):
        # Test when CUDA device is not found
        config = PipelineConfig(model_name_or_path="facebook/opt-125m")
        config.device = "cuda"
        torch.cuda.is_available = MagicMock(return_value=False)
        result = build_chatbot(config)
        self.assertIsNone(result)

    @unittest.skipIf(get_device_type() != 'cpu', "Only run this test on CPU")
    def test_xpu_device_not_found(self):
        # Test when XPU device is not found
        config = PipelineConfig(model_name_or_path="facebook/opt-125m")
        config.device = "xpu"
        torch.xpu.is_available = MagicMock(return_value=False)
        result = build_chatbot(config)
        self.assertIsNone(result)

    @unittest.skipIf(get_device_type() != 'cpu', "Only run this test on CPU")
    def test_invalid_model_name(self):
        # Test with an unsupported model name
        config = PipelineConfig(model_name_or_path="facebook/opt-125m")
        config.model_name_or_path = "InvalidModel"
        result = build_chatbot(config)
        self.assertIsNone(result)

    @unittest.skipIf(get_device_type() != 'cpu', "Only run this test on CPU")
    @patch('intel_extension_for_transformers.neural_chat.models.model_utils.load_model')
    def test_adapter_load_model_out_of_memory(self, mock_load_model):
        # Test out of memory exception handling
        config = PipelineConfig(model_name_or_path="facebook/opt-125m")
        mock_load_model.side_effect = RuntimeError("Mocked exception: out of memory")
        result = build_chatbot(config=config)
        self.assertIsNone(result)

    @unittest.skipIf(get_device_type() != 'cpu', "Only run this test on CPU")
    @patch('intel_extension_for_transformers.neural_chat.models.model_utils.load_model')
    def test_adapter_load_model_device_busy(self, mock_load_model):
        # Test device busy or unavailable exception handling
        config = PipelineConfig(model_name_or_path="facebook/opt-125m")
        mock_load_model.side_effect = RuntimeError("devices are busy or unavailable")
        result = build_chatbot(config)
        self.assertIsNone(result)

    @unittest.skipIf(get_device_type() != 'cpu', "Only run this test on CPU")
    @patch('intel_extension_for_transformers.neural_chat.models.model_utils.load_model')
    def test_adapter_load_model_device_not_found(self, mock_load_model):
        # Test device not found exception handling
        mock_load_model.side_effect = RuntimeError("tensor does not have a device")
        config = PipelineConfig(model_name_or_path="facebook/opt-125m")
        result = build_chatbot(config)
        self.assertIsNone(result)

    @unittest.skipIf(get_device_type() != 'cpu', "Only run this test on CPU")
    @patch('intel_extension_for_transformers.neural_chat.models.model_utils.load_model')
    def test_adapter_load_model_runtime_exception(self, mock_load_model):
        # Test generic exception handling
        config = PipelineConfig(model_name_or_path="facebook/opt-125m")
        mock_load_model.side_effect = RuntimeError("Some generic error")
        result = build_chatbot(config)
        self.assertIsNone(result)

    @unittest.skipIf(get_device_type() != 'cpu', "Only run this test on CPU")
    @patch('intel_extension_for_transformers.neural_chat.models.model_utils.load_model')
    def test_adapter_load_model_value_error_unsupported_device(self, mock_load_model):
        # Test value error exception handling
        config = PipelineConfig(model_name_or_path="facebook/opt-125m")
        mock_load_model.side_effect = ValueError("load_model: unsupported device")
        result = build_chatbot(config)
        self.assertIsNone(result)

    @unittest.skipIf(get_device_type() != 'cpu', "Only run this test on CPU")
    @patch('intel_extension_for_transformers.neural_chat.models.model_utils.load_model')
    def test_adapter_load_model_value_error_unsupported_model(self, mock_load_model):
        # Test value error exception handling
        config = PipelineConfig(model_name_or_path="facebook/opt-125m")
        mock_load_model.side_effect = ValueError("load_model: unsupported model")
        result = build_chatbot(config)
        self.assertIsNone(result)

    @unittest.skipIf(get_device_type() != 'cpu', "Only run this test on CPU")
    @patch('intel_extension_for_transformers.neural_chat.models.model_utils.load_model')
    def test_adapter_load_model_value_error_tokenizer_not_found(self, mock_load_model):
        # Test value error exception handling
        config = PipelineConfig(model_name_or_path="facebook/opt-125m")
        mock_load_model.side_effect = ValueError("load_model: tokenizer is not found")
        result = build_chatbot(config)
        self.assertIsNone(result)

    @unittest.skipIf(get_device_type() != 'cpu', "Only run this test on CPU")
    @patch('intel_extension_for_transformers.neural_chat.models.model_utils.load_model')
    def test_adapter_load_model_value_error_model_not_found(self, mock_load_model):
        # Test value error exception handling
        config = PipelineConfig(model_name_or_path="facebook/opt-125m")
        mock_load_model.side_effect = ValueError("load_model: model name or path is not found")
        result = build_chatbot(config)
        self.assertIsNone(result)

    @unittest.skipIf(get_device_type() != 'cpu', "Only run this test on CPU")
    @patch('intel_extension_for_transformers.neural_chat.models.model_utils.load_model')
    def test_adapter_load_model_value_error_model_config_not_found(self, mock_load_model):
        # Test value error exception handling
        config = PipelineConfig(model_name_or_path="facebook/opt-125m")
        mock_load_model.side_effect = ValueError("load_model: model config is not found")
        result = build_chatbot(config)
        self.assertIsNone(result)

    @unittest.skipIf(get_device_type() != 'cpu', "Only run this test on CPU")
    @patch('intel_extension_for_transformers.neural_chat.models.model_utils.load_model')
    def test_adapter_load_model_value_error_generic(self, mock_load_model):
        # Test value error exception handling
        config = PipelineConfig(model_name_or_path="facebook/opt-125m")
        mock_load_model.side_effect = ValueError("load_model: some generic error")
        result = build_chatbot(config)
        self.assertIsNone(result)

    @unittest.skipIf(get_device_type() != 'cpu', "Only run this test on CPU")
    @patch('intel_extension_for_transformers.neural_chat.models.model_utils.load_model')
    def test_adapter_load_model_value_error_unknown(self, mock_load_model):
        # Test value error exception handling
        config = PipelineConfig(model_name_or_path="facebook/opt-125m")
        mock_load_model.side_effect = ValueError("load_model: unknown ValueError occurred")
        result = build_chatbot(config)
        self.assertIsNone(result)

    @unittest.skipIf(get_device_type() != 'cpu', "Only run this test on CPU")
    @patch('intel_extension_for_transformers.neural_chat.models.model_utils.load_model')
    def test_adapter_load_model_environmentvalue_error_unknown(self, mock_load_model):
        # Test value error exception handling
        config = PipelineConfig(model_name_or_path="facebook/opt-125m")
        mock_load_model.side_effect = ValueError("load_model: unknown EnvironmentError occurred")
        result = build_chatbot(config)
        self.assertIsNone(result)

    @unittest.skipIf(get_device_type() != 'cpu', "Only run this test on CPU")
    @patch('intel_extension_for_transformers.neural_chat.models.model_utils.load_model')
    def test_adapter_load_model_error_unexpected(self, mock_load_model):
        # Test value error exception handling
        config = PipelineConfig(model_name_or_path="facebook/opt-125m")
        mock_load_model.side_effect = ValueError("load_model: an unexpected error occurred")
        result = build_chatbot(config)
        self.assertIsNone(result)

    @unittest.skipIf(get_device_type() != 'cpu', "Only run this test on CPU")
    @patch('intel_extension_for_transformers.neural_chat.models.model_utils.load_model')
    def test_adapter_load_model_unsupported_model(self, mock_load_model):
        # Test value error exception handling
        config = PipelineConfig(model_name_or_path="facebook/opt-125m")
        mock_load_model.side_effect = ValueError("unsupported model name or path {model_name}")
        result = build_chatbot(config)
        self.assertIsNone(result)

    @unittest.skipIf(get_device_type() != 'cpu', "Only run this test on CPU")
    @patch('intel_extension_for_transformers.neural_chat.models.model_utils.load_model')
    def test_adapter_load_model_unsupported_device(self, mock_load_model):
        # Test value error exception handling
        config = PipelineConfig(model_name_or_path="facebook/opt-125m")
        mock_load_model.side_effect = ValueError("unsupported device {device}, only supports cpu, xpu, cuda and hpu now.")
        result = build_chatbot(config)
        self.assertIsNone(result)

    @unittest.skipIf(get_device_type() != 'cpu', "Only run this test on CPU")
    @patch('intel_extension_for_transformers.neural_chat.models.model_utils.load_model')
    def test_adapter_load_model_exception(self, mock_load_model):
        # Test generic exception handling
        config = PipelineConfig(model_name_or_path="facebook/opt-125m")
        mock_load_model.side_effect = Exception("Some generic error")
        result = build_chatbot(config)
        self.assertIsNone(result)

if __name__ == '__main__':
    unittest.main()
