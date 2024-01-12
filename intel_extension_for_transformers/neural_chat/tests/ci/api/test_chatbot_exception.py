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
from intel_extension_for_transformers.neural_chat.utils.common import get_device_type
from transformers import TrainingArguments, Seq2SeqTrainingArguments
from intel_extension_for_transformers.neural_chat.config import (
    ModelArguments,
    DataArguments,
    FinetuningArguments,
    TextGenerationFinetuningConfig,
)
from intel_extension_for_transformers.neural_chat.chatbot import finetune_model
import unittest
import shutil
import os

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
    def test_adapter_load_model_exception(self, mock_load_model):
        # Test generic exception handling
        config = PipelineConfig(model_name_or_path="facebook/opt-125m")
        mock_load_model.side_effect = Exception("Some generic error")
        result = build_chatbot(config)
        self.assertIsNone(result)

json_data = \
"""
[
    {"instruction": "Generate a slogan for a software company", "input": "", "output": "The Future of Software is Here"},
    {"instruction": "Provide the word that comes immediately after the.", "input": "He threw the ball over the fence.", "output": "fence."}
]
"""
test_data_file = './alpaca_test.json'

class TestFinetuneExceptions(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.device = get_device_type()
        with open(test_data_file, mode='w') as f:
            f.write(json_data)

            self.training_args = TrainingArguments(
                    output_dir='./tmp',
                    do_train=True,
                    max_steps=3,
                    overwrite_output_dir=True)

            self.seq2seq_training_args = Seq2SeqTrainingArguments(
                    output_dir='./tmp',
                    do_train=True,
                    max_steps=3,
                    overwrite_output_dir=True)


    @classmethod
    def tearDownClass(self):
        shutil.rmtree('./tmp', ignore_errors=True)
        os.remove(test_data_file)

    @unittest.skipIf(get_device_type() != 'cpu', "Only run this test on CPU")
    @patch('intel_extension_for_transformers.llm.finetuning.finetuning.Finetuning')
    def test_finetune_error_dataset_not_found(self, mock_finetune):
        model_args = ModelArguments(model_name_or_path="facebook/opt-125m")
        mock_finetune.side_effect = FileNotFoundError("Couldn't find a dataset script")
        data_args = DataArguments(train_file=test_data_file)
        finetune_args = FinetuningArguments(device=self.device, do_lm_eval=False)
        finetune_cfg = TextGenerationFinetuningConfig(
            model_args=model_args,
            data_args=data_args,
            training_args=self.training_args,
            finetune_args=finetune_args
        )
        self.assertRaises(FileNotFoundError,finetune_model,finetune_cfg)


if __name__ == '__main__':
    unittest.main()
