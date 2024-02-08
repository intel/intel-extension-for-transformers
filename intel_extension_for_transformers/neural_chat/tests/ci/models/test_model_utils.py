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
from unittest import mock
from intel_extension_for_transformers.neural_chat.models.model_utils import load_model, MODELS, predict
from intel_extension_for_transformers.transformers import MixedPrecisionConfig, BitsAndBytesConfig, WeightOnlyQuantConfig
from intel_extension_for_transformers.neural_chat.utils.common import get_device_type
from intel_extension_for_transformers.neural_chat.utils.error_utils import clear_latest_error, get_latest_error
from intel_extension_for_transformers.neural_chat.errorcode import ErrorCodes
class TestModelUtils(unittest.TestCase):
    def setUp(self) -> None:
        clear_latest_error()
        return super().setUpClass()

    def tearDown(self) -> None:
        if os.path.exists("nc_workspace"):
            shutil.rmtree("nc_workspace")
        if os.path.exists("runtime_outs"):
            shutil.rmtree("runtime_outs")
        return super().tearDown()

    @unittest.skipIf(get_device_type() != 'cpu', "Only run this test on CPU")
    def test_load_model_on_cpu(self):
        load_model(model_name="facebook/opt-125m", tokenizer_name="facebook/opt-125m", device="cpu")
        self.assertTrue("facebook/opt-125m" in MODELS)
        self.assertTrue(MODELS["facebook/opt-125m"]["model"] is not None)

    @unittest.skipIf(get_device_type() != 'cpu', "Only run this test on CPU")
    def test_load_model_on_gpu_with_mock(self):
        with mock.patch('torch.cuda.is_available', return_value=True):
            with mock.patch('torch.load', return_value={'model_state_dict': {}}):
                with mock.patch('transformers.AutoModelForCausalLM.from_pretrained') as mock_from_pretrained:
                    model_mock = mock.Mock()
                    model_mock.eval.return_value = model_mock
                    model_mock.to.return_value = model_mock
                    model_mock.config = mock.Mock()
                    model_mock.config.architectures = ['OPTForCausalLM']
                    model_mock.generation_config.pad_token_id = 1
                    model_mock.generation_config.eos_token_id = 2
                    model_mock.generation_config.bos_token_id = 2
                    # Mock the convert_ids_to_tokens method
                    model_mock.tokenizer.convert_ids_to_tokens.return_value = "some_token"
                    mock_from_pretrained.return_value = model_mock
                    load_model(model_name="facebook/opt-125m", tokenizer_name="facebook/opt-125m", device="cuda")
                    mock_from_pretrained.assert_called_once()
                    model_mock.eval.assert_called_once()
                    model_mock.to.assert_called_once_with('cuda')

    @unittest.skipIf(get_device_type() != 'cpu', "Only run this test on CPU")
    def test_load_model_on_xpu_with_mock(self):
        with mock.patch('torch.xpu.is_available', return_value=True):
            with mock.patch('torch.load', return_value={'model_state_dict': {}}):
                with mock.patch('transformers.AutoModelForCausalLM.from_pretrained') as mock_from_pretrained:
                    model_mock = mock.Mock()
                    model_mock.eval.return_value = model_mock
                    model_mock.to.return_value = model_mock
                    model_mock.config = mock.Mock()
                    model_mock.config.architectures = ['OPTForCausalLM']
                    model_mock.generation_config.pad_token_id = 1
                    model_mock.generation_config.eos_token_id = 2
                    model_mock.generation_config.bos_token_id = 2
                    # Mock the convert_ids_to_tokens method to return a list or other iterable object
                    model_mock.tokenizer.convert_ids_to_tokens.return_value = ["some_token"]
                    mock_from_pretrained.return_value = model_mock
                    load_model(model_name="facebook/opt-125m", tokenizer_name="facebook/opt-125m", device="xpu")
                    mock_from_pretrained.assert_called_once()
                    model_mock.eval.assert_called_once()
                    model_mock.to.assert_called_once_with('xpu')

    @unittest.skipIf(get_device_type() != 'cpu', "Only run this test on CPU")
    def test_load_model_with_assistant_on_cpu(self):
        load_model(model_name="facebook/opt-350m", tokenizer_name="facebook/opt-350m",
                   device="cpu", assistant_model="facebook/opt-125m")
        self.assertTrue("facebook/opt-350m" in MODELS)
        self.assertTrue(MODELS["facebook/opt-350m"]["assistant_model"] is not None)

    @unittest.skipIf(get_device_type() != 'cpu', "Only run this test on CPU")
    def test_load_nonexistent_model(self):
        load_model("non-existent-model", "non-existent-model", device="cpu")
        self.assertEqual(get_latest_error(), ErrorCodes.ERROR_MODEL_NOT_FOUND)

    @unittest.skipIf(get_device_type() != 'cpu', "Only run this test on CPU")
    def test_model_optimization_mix_precision(self):
        config = MixedPrecisionConfig(dtype="bfloat16")
        load_model(model_name="facebook/opt-125m", tokenizer_name="facebook/opt-125m", device="cpu", optimization_config=config)
        self.assertTrue("facebook/opt-125m" in MODELS)
        self.assertTrue(MODELS["facebook/opt-125m"]["model"] is not None)

    @unittest.skipIf(get_device_type() != 'cpu', "Only run this test on CPU")
    def test_model_optimization_bitsandbytes(self):
        with mock.patch('torch.cuda.is_available', return_value=True):
            with mock.patch('torch.load', return_value={'model_state_dict': {}}):
                with mock.patch('transformers.AutoModelForCausalLM.from_pretrained') as mock_from_pretrained:
                    model_mock = mock.Mock()
                    model_mock.eval.return_value = model_mock
                    model_mock.to.return_value = model_mock
                    model_mock.config = mock.Mock()
                    model_mock.config.architectures = ['OPTForCausalLM']
                    model_mock.generation_config.pad_token_id = 1
                    model_mock.generation_config.eos_token_id = 2
                    model_mock.generation_config.bos_token_id = 2
                    # Mock the convert_ids_to_tokens method
                    model_mock.tokenizer.convert_ids_to_tokens.return_value = "some_token"
                    mock_from_pretrained.return_value = model_mock
                    load_model(model_name="facebook/opt-125m", tokenizer_name="facebook/opt-125m",
                                device="cuda", optimization_config=BitsAndBytesConfig())
                    mock_from_pretrained.assert_called_once()
                    model_mock.eval.assert_called_once()
                    model_mock.to.assert_called_once_with('cuda')

    @unittest.skipIf(get_device_type() != 'cpu', "Only run this test on CPU")
    def test_model_optimization_weightonly_llmruntime(self):
        config = WeightOnlyQuantConfig(compute_dtype="int8", weight_dtype="int4")
        load_model(model_name="facebook/opt-125m", tokenizer_name="facebook/opt-125m", device="cpu", optimization_config=config, use_llm_runtime=True)
        self.assertTrue("facebook/opt-125m" in MODELS)
        self.assertTrue(MODELS["facebook/opt-125m"]["model"] is not None)

    @unittest.skipIf(get_device_type() != 'cpu', "Only run this test on CPU")
    def test_model_optimization_weightonly(self):
        config = WeightOnlyQuantConfig(compute_dtype="int8", weight_dtype="int4_fullrange")
        load_model(model_name="facebook/opt-125m", tokenizer_name="facebook/opt-125m", device="cpu", optimization_config=config)
        self.assertTrue("facebook/opt-125m" in MODELS)
        self.assertTrue(MODELS["facebook/opt-125m"]["model"] is not None)

    @unittest.skipIf(get_device_type() != 'cpu', "Only run this test on CPU")
    def test_model_predict(self):
        load_model(model_name="facebook/opt-125m", tokenizer_name="facebook/opt-125m", device="cpu")
        self.assertTrue("facebook/opt-125m" in MODELS)
        self.assertTrue(MODELS["facebook/opt-125m"]["model"] is not None)

        params = {
            "model_name": "facebook/opt-125m",
            "prompt": "hi"
        }
        output = predict(**params)
        self.assertIn("hi", output)
        self.assertNotIn("[/INST]", output)

if __name__ == '__main__':
    unittest.main()
