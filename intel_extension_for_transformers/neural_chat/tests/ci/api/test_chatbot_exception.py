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
from transformers import TrainingArguments
from intel_extension_for_transformers.neural_chat.config import (
    ModelArguments,
    DataArguments,
    FinetuningArguments,
    TextGenerationFinetuningConfig,
)
from intel_extension_for_transformers.neural_chat.chatbot import finetune_model, optimize_model
from intel_extension_for_transformers.transformers import MixedPrecisionConfig, WeightOnlyQuantConfig
from intel_extension_for_transformers.transformers import BitsAndBytesConfig
from intel_extension_for_transformers.neural_chat.errorcode import ErrorCodes
from intel_extension_for_transformers.neural_chat.utils.error_utils import get_latest_error
from intel_extension_for_transformers.neural_chat import plugins
from transformers import AutoModelForCausalLM
import unittest
import shutil
import os

class TestBuildChatbotExceptions(unittest.TestCase):

    @unittest.skipIf(get_device_type() != 'cpu', "Only run this test on CPU")
    @patch('transformers.AutoModelForCausalLM.from_pretrained')
    def test_out_of_storage(self, mock_from_pretrained):
        # Simulate out of storage scenario
        config = PipelineConfig(model_name_or_path="facebook/opt-125m")
        mock_from_pretrained.side_effect = Exception("No space left on device")
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
    @patch('transformers.AutoModelForCausalLM.from_pretrained')
    def test_adapter_load_model_out_of_memory(self, mock_from_pretrained):
        # Test out of memory exception handling
        config = PipelineConfig(model_name_or_path="facebook/opt-125m")
        mock_from_pretrained.side_effect = Exception("out of memory")
        result = build_chatbot(config=config)
        self.assertIsNone(result)

    @unittest.skipIf(get_device_type() != 'cpu', "Only run this test on CPU")
    @patch('transformers.AutoModelForCausalLM.from_pretrained')
    def test_adapter_load_model_cache_dir_no_write_permission(self, mock_from_pretrained):
        # Test out of memory exception handling
        config = PipelineConfig(model_name_or_path="facebook/opt-125m")
        mock_from_pretrained.side_effect = Exception("Permission denied")
        result = build_chatbot(config=config)
        self.assertIsNone(result)

    @unittest.skipIf(get_device_type() != 'cpu', "Only run this test on CPU")
    @patch('transformers.AutoModelForCausalLM.from_pretrained')
    def test_adapter_load_model_device_busy(self, mock_from_pretrained):
        # Test device busy or unavailable exception handling
        config = PipelineConfig(model_name_or_path="facebook/opt-125m")
        mock_from_pretrained.side_effect = RuntimeError("devices are busy or unavailable")
        result = build_chatbot(config)
        self.assertIsNone(result)

    @unittest.skipIf(get_device_type() != 'cpu', "Only run this test on CPU")
    @patch('transformers.AutoModelForCausalLM.from_pretrained')
    def test_adapter_load_model_device_not_found(self, mock_from_pretrained):
        # Test device not found exception handling
        mock_from_pretrained.side_effect = RuntimeError("tensor does not have a device")
        config = PipelineConfig(model_name_or_path="facebook/opt-125m")
        result = build_chatbot(config)
        self.assertIsNone(result)

    @unittest.skipIf(get_device_type() != 'cpu', "Only run this test on CPU")
    @patch('transformers.AutoModelForCausalLM.from_pretrained')
    def test_adapter_load_model_runtime_exception(self, mock_from_pretrained):
        # Test generic exception handling
        config = PipelineConfig(model_name_or_path="facebook/opt-125m")
        mock_from_pretrained.side_effect = RuntimeError("Some generic error")
        result = build_chatbot(config)
        self.assertIsNone(result)

    @unittest.skipIf(get_device_type() != 'cpu', "Only run this test on CPU")
    @patch('transformers.AutoModelForCausalLM.from_pretrained')
    def test_adapter_load_model_value_error_unsupported_device(self, mock_from_pretrained):
        # Test value error exception handling
        config = PipelineConfig(model_name_or_path="facebook/opt-125m")
        mock_from_pretrained.side_effect = ValueError("load_model: unsupported device")
        result = build_chatbot(config)
        self.assertIsNone(result)

    @unittest.skipIf(get_device_type() != 'cpu', "Only run this test on CPU")
    @patch('transformers.AutoModelForCausalLM.from_pretrained')
    def test_adapter_load_model_value_error_unsupported_model(self, mock_from_pretrained):
        # Test value error exception handling
        config = PipelineConfig(model_name_or_path="facebook/opt-125m")
        mock_from_pretrained.side_effect = ValueError("load_model: unsupported model")
        result = build_chatbot(config)
        self.assertIsNone(result)

    @unittest.skipIf(get_device_type() != 'cpu', "Only run this test on CPU")
    def test_adapter_unknown_plugin(self):
        # Test unknown plugin exception handling
        config = PipelineConfig(model_name_or_path="facebook/opt-125m",
                plugins={"unknown_plugin": {"enable": True, "class": None, "args": {}, "instance": None}})
        result = build_chatbot(config)
        self.assertIsNone(result)

    @unittest.skipIf(get_device_type() != 'cpu', "Only run this test on CPU")
    @patch('transformers.AutoModelForCausalLM.from_pretrained')
    def test_adapter_load_model_value_error_tokenizer_not_found(self, mock_from_pretrained):
        # Test value error exception handling
        config = PipelineConfig(model_name_or_path="facebook/opt-125m")
        mock_from_pretrained.side_effect = ValueError("load_model: tokenizer is not found")
        result = build_chatbot(config)
        self.assertIsNone(result)

    @unittest.skipIf(get_device_type() != 'cpu', "Only run this test on CPU")
    @patch('transformers.AutoModelForCausalLM.from_pretrained')
    def test_adapter_load_model_value_error_model_not_found(self, mock_from_pretrained):
        # Test value error exception handling
        config = PipelineConfig(model_name_or_path="facebook/opt-125m")
        mock_from_pretrained.side_effect = ValueError("load_model: model name or path is not found")
        result = build_chatbot(config)
        self.assertIsNone(result)

    @unittest.skipIf(get_device_type() != 'cpu', "Only run this test on CPU")
    @patch('transformers.AutoModelForCausalLM.from_pretrained')
    def test_adapter_load_model_value_error_model_config_not_found(self, mock_from_pretrained):
        # Test value error exception handling
        config = PipelineConfig(model_name_or_path="facebook/opt-125m")
        mock_from_pretrained.side_effect = ValueError("load_model: model config is not found")
        result = build_chatbot(config)
        self.assertIsNone(result)

    @unittest.skipIf(get_device_type() != 'cpu', "Only run this test on CPU")
    @patch('transformers.AutoModelForCausalLM.from_pretrained')
    def test_adapter_load_model_value_error_generic(self, mock_from_pretrained):
        # Test value error exception handling
        config = PipelineConfig(model_name_or_path="facebook/opt-125m")
        mock_from_pretrained.side_effect = ValueError("load_model: some generic error")
        result = build_chatbot(config)
        self.assertIsNone(result)

    @unittest.skipIf(get_device_type() != 'cpu', "Only run this test on CPU")
    @patch('transformers.AutoModelForCausalLM.from_pretrained')
    def test_adapter_load_model_exception(self, mock_from_pretrained):
        # Test generic exception handling
        config = PipelineConfig(model_name_or_path="facebook/opt-125m")
        mock_from_pretrained.side_effect = Exception("Some generic error")
        result = build_chatbot(config)
        self.assertIsNone(result)

    @unittest.skipIf(get_device_type() != 'cpu', "Only run this test on CPU")
    @patch('transformers.models.opt.modeling_opt.OPTForCausalLM.forward')
    def test_model_generate_exception(self, mock_forward):
        # Test model inference exception handling
        config = PipelineConfig(model_name_or_path="facebook/opt-125m")
        chatbot = build_chatbot(config)
        mock_forward.side_effect = Exception("Model inference error")
        result = chatbot.predict("Tell me about Intel.")
        self.assertIsNone(result)

    @unittest.skipIf(get_device_type() != 'cpu', "Only run this test on CPU")
    @patch('intel_extension_for_transformers.neural_chat.pipeline.plugins.retrieval.detector.intent_detection.IntentDetector.intent_detection')
    def test_model_intent_detection_exception(self, mock_intent_detection):
        # test intent detection exception handling
        plugins.retrieval.enable = True
        plugins.retrieval.args["input_path"] = "../assets/docs/sample.txt"
        plugins.retrieval.args["persist_directory"] = "./test_txt"
        plugins.retrieval.args["retrieval_type"] = 'default'
        config = PipelineConfig(model_name_or_path="facebook/opt-125m")
        chatbot = build_chatbot(config)
        mock_intent_detection.side_effect = Exception("intent detect error")
        result = chatbot.predict("Tell me about Intel.")
        self.assertIsNone(result)
        plugins.retrieval.enable = False

    @unittest.skipIf(get_device_type() != 'cpu', "Only run this test on CPU")
    @patch('intel_extension_for_transformers.neural_chat.pipeline.plugins.retrieval.detector.intent_detection.IntentDetector.intent_detection')
    def test_model_intent_detection_stream_exception(self, mock_intent_detection):
        # test intent detection exception handling
        plugins.retrieval.enable = True
        plugins.retrieval.args["input_path"] = "../assets/docs/sample.txt"
        plugins.retrieval.args["persist_directory"] = "./test_txt"
        plugins.retrieval.args["retrieval_type"] = 'default'
        config = PipelineConfig(model_name_or_path="facebook/opt-125m")
        chatbot = build_chatbot(config)
        mock_intent_detection.side_effect = Exception("intent detect error")
        result = chatbot.predict_stream("Tell me about Intel.")
        self.assertIsNone(result)
        plugins.retrieval.enable = False

    @unittest.skipIf(get_device_type() != 'cpu', "Only run this test on CPU")
    def test_asr_audio_format_exception(self):
        plugins.asr.enable = True
        plugins.tts.enable = True
        plugins.tts.args["output_audio_path"] = "./response.wav"
        pipeline_config = PipelineConfig(model_name_or_path="facebook/opt-125m", plugins=plugins)
        chatbot = build_chatbot(pipeline_config)
        audio_path = "/intel-extension-for-transformers/intel_extension_for_transformers/neural_chat/assets/video/intel.mp4"
        if not os.path.exists(audio_path):
            audio_path = "../assets/video/intel.mp4"
        response = chatbot.predict(query=audio_path)
        self.assertIsNone(response)
        plugins.asr.enable = False
        plugins.tts.enable = False

    @unittest.skipIf(get_device_type() != 'cpu', "Only run this test on CPU")
    def test_model_doc_format_not_supported_exception(self):
        # test retrieval doc format= exception handling
        doc_path = "/intel-extension-for-transformers/intel_extension_for_transformers/neural_chat/assets/video/intel.mp4"
        if not os.path.exists(doc_path):
            doc_path = "../assets/video/intel.mp4"
        plugins.retrieval.enable = True
        plugins.retrieval.args["input_path"] = doc_path
        plugins.retrieval.args["persist_directory"] = "./test_txt"
        plugins.retrieval.args["retrieval_type"] = 'default'
        config = PipelineConfig(model_name_or_path="facebook/opt-125m")
        chatbot = build_chatbot(config)
        self.assertIsNone(chatbot)
        plugins.retrieval.enable = False

    @unittest.skipIf(get_device_type() != 'cpu', "Only run this test on CPU")
    def test_model_safety_checker_exception(self):
        # test safety checker exception handling
        from intel_extension_for_transformers.neural_chat.pipeline.plugins.security.safety_checker import SafetyChecker
        with self.assertRaises(Exception) as context:
            safety_checker = SafetyChecker(dict_path="./")
        self.assertEqual(str(context.exception), "[SafetyChecker ERROR] Sensitive check file not found!")

    @unittest.skipIf(get_device_type() != 'cpu', "Only run this test on CPU")
    def test_asr_audio_format_stream_exception(self):
        plugins.asr.enable = True
        plugins.tts.enable = True
        plugins.tts.args["output_audio_path"] = "./response.wav"
        pipeline_config = PipelineConfig(model_name_or_path="facebook/opt-125m", plugins=plugins)
        chatbot = build_chatbot(pipeline_config)
        audio_path = "/intel-extension-for-transformers/intel_extension_for_transformers/neural_chat/assets/video/intel.mp4"
        if not os.path.exists(audio_path):
            audio_path = os.path.abspath("../assets/video/intel.mp4")
        response = chatbot.predict_stream(query=audio_path)
        self.assertIsNone(response)
        plugins.asr.enable = False
        plugins.tts.enable = False

json_data = \
"""
[
    {"instruction": "Generate a slogan for a software company", "input": "", "output": "The Future of Software is Here"},
    {"instruction": "Provide the word that comes immediately after the.", "input": "He threw the ball over the fence.", "output": "fence."}
]
"""
test_data_file = './alpaca_test.json'

class TestFinetuneModelExceptions(unittest.TestCase):
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

    @classmethod
    def tearDownClass(self):
        shutil.rmtree('./tmp', ignore_errors=True)
        os.remove(test_data_file)

    @unittest.skipIf(get_device_type() != 'cpu', "Only run this test on CPU")
    @patch('intel_extension_for_transformers.llm.finetuning.finetuning.Finetuning.finetune')
    def test_finetune_error_dataset_not_found(self, mock_finetune):
        model_args = ModelArguments(model_name_or_path="facebook/opt-125m")
        data_args = DataArguments(train_file=test_data_file)
        finetune_args = FinetuningArguments(device=self.device, do_lm_eval=False)
        finetune_cfg = TextGenerationFinetuningConfig(
            model_args=model_args,
            data_args=data_args,
            training_args=self.training_args,
            finetune_args=finetune_args
        )
        mock_finetune.side_effect = FileNotFoundError("Couldn't find a dataset script")
        finetune_model(finetune_cfg)
        self.assertEqual(get_latest_error(), ErrorCodes.ERROR_DATASET_NOT_FOUND)

    @unittest.skipIf(get_device_type() != 'cpu', "Only run this test on CPU")
    @patch('intel_extension_for_transformers.llm.finetuning.finetuning.Finetuning.finetune')
    def test_finetune_error_validation_file_not_found(self, mock_finetune):
        model_args = ModelArguments(model_name_or_path="facebook/opt-125m")
        data_args = DataArguments(train_file=test_data_file)
        finetune_args = FinetuningArguments(device=self.device, do_lm_eval=False)
        finetune_cfg = TextGenerationFinetuningConfig(
            model_args=model_args,
            data_args=data_args,
            training_args=self.training_args,
            finetune_args=finetune_args
        )
        mock_finetune.side_effect = ValueError("--do_eval requires a validation dataset")
        finetune_model(finetune_cfg)
        self.assertEqual(get_latest_error(), ErrorCodes.ERROR_VALIDATION_FILE_NOT_FOUND)

    @unittest.skipIf(get_device_type() != 'cpu', "Only run this test on CPU")
    @patch('intel_extension_for_transformers.llm.finetuning.finetuning.Finetuning.finetune')
    def test_finetune_error_train_file_not_found(self, mock_finetune):
        model_args = ModelArguments(model_name_or_path="facebook/opt-125m")
        data_args = DataArguments(train_file=test_data_file)
        finetune_args = FinetuningArguments(device=self.device, do_lm_eval=False)
        finetune_cfg = TextGenerationFinetuningConfig(
            model_args=model_args,
            data_args=data_args,
            training_args=self.training_args,
            finetune_args=finetune_args
        )
        mock_finetune.side_effect = ValueError("--do_train requires a train dataset")
        finetune_model(finetune_cfg)
        self.assertEqual(get_latest_error(), ErrorCodes.ERROR_TRAIN_FILE_NOT_FOUND)

    @unittest.skipIf(get_device_type() != 'cpu', "Only run this test on CPU")
    @patch('intel_extension_for_transformers.llm.finetuning.finetuning.Finetuning.finetune')
    def test_finetune_lora_finetune_fail(self, mock_finetune):
        model_args = ModelArguments(model_name_or_path="facebook/opt-125m")
        data_args = DataArguments(train_file=test_data_file)
        finetune_args = FinetuningArguments(device=self.device, do_lm_eval=False, peft="lora")
        finetune_cfg = TextGenerationFinetuningConfig(
            model_args=model_args,
            data_args=data_args,
            training_args=self.training_args,
            finetune_args=finetune_args
        )
        mock_finetune.side_effect = Exception
        finetune_model(finetune_cfg)
        self.assertEqual(get_latest_error(), ErrorCodes.ERROR_LORA_FINETUNE_FAIL)

    @unittest.skipIf(get_device_type() != 'cpu', "Only run this test on CPU")
    @patch('intel_extension_for_transformers.llm.finetuning.finetuning.Finetuning.finetune')
    def test_finetune_llama_adapter_finetune_fail(self, mock_finetune):
        model_args = ModelArguments(model_name_or_path="facebook/opt-125m")
        data_args = DataArguments(train_file=test_data_file)
        finetune_args = FinetuningArguments(device=self.device, do_lm_eval=False, peft="llama_adapter")
        finetune_cfg = TextGenerationFinetuningConfig(
            model_args=model_args,
            data_args=data_args,
            training_args=self.training_args,
            finetune_args=finetune_args
        )
        mock_finetune.side_effect = Exception
        finetune_model(finetune_cfg)
        self.assertEqual(get_latest_error(), ErrorCodes.ERROR_LLAMA_ADAPTOR_FINETUNE_FAIL)

    @unittest.skipIf(get_device_type() != 'cpu', "Only run this test on CPU")
    @patch('intel_extension_for_transformers.llm.finetuning.finetuning.Finetuning.finetune')
    def test_finetune_ptun_finetune_fail(self, mock_finetune):
        model_args = ModelArguments(model_name_or_path="facebook/opt-125m")
        data_args = DataArguments(train_file=test_data_file)
        finetune_args = FinetuningArguments(device=self.device, do_lm_eval=False, peft="ptun")
        finetune_cfg = TextGenerationFinetuningConfig(
            model_args=model_args,
            data_args=data_args,
            training_args=self.training_args,
            finetune_args=finetune_args
        )
        mock_finetune.side_effect = Exception
        finetune_model(finetune_cfg)
        self.assertEqual(get_latest_error(), ErrorCodes.ERROR_PTUN_FINETUNE_FAIL)

    @unittest.skipIf(get_device_type() != 'cpu', "Only run this test on CPU")
    @patch('intel_extension_for_transformers.llm.finetuning.finetuning.Finetuning.finetune')
    def test_finetune_prefix_finetune_fail(self, mock_finetune):
        model_args = ModelArguments(model_name_or_path="facebook/opt-125m")
        data_args = DataArguments(train_file=test_data_file)
        finetune_args = FinetuningArguments(device=self.device, do_lm_eval=False, peft="prefix")
        finetune_cfg = TextGenerationFinetuningConfig(
            model_args=model_args,
            data_args=data_args,
            training_args=self.training_args,
            finetune_args=finetune_args
        )
        mock_finetune.side_effect = Exception
        finetune_model(finetune_cfg)
        self.assertEqual(get_latest_error(), ErrorCodes.ERROR_PREFIX_FINETUNE_FAIL)

    @unittest.skipIf(get_device_type() != 'cpu', "Only run this test on CPU")
    @patch('intel_extension_for_transformers.llm.finetuning.finetuning.Finetuning.finetune')
    def test_finetune_prompt_finetune_fail(self, mock_finetune):
        model_args = ModelArguments(model_name_or_path="facebook/opt-125m")
        data_args = DataArguments(train_file=test_data_file)
        finetune_args = FinetuningArguments(device=self.device, do_lm_eval=False, peft="prompt")
        finetune_cfg = TextGenerationFinetuningConfig(
            model_args=model_args,
            data_args=data_args,
            training_args=self.training_args,
            finetune_args=finetune_args
        )
        mock_finetune.side_effect = Exception
        finetune_model(finetune_cfg)
        self.assertEqual(get_latest_error(), ErrorCodes.ERROR_PROMPT_FINETUNE_FAIL)

    @unittest.skipIf(get_device_type() != 'cpu', "Only run this test on CPU")
    @patch('intel_extension_for_transformers.llm.finetuning.finetuning.Finetuning.finetune')
    def test_finetune_data_no_permission_finetune_fail(self, mock_finetune):
        model_args = ModelArguments(model_name_or_path="facebook/opt-125m")
        data_args = DataArguments(train_file=test_data_file)
        finetune_args = FinetuningArguments(device=self.device, do_lm_eval=False, peft="lora")
        finetune_cfg = TextGenerationFinetuningConfig(
            model_args=model_args,
            data_args=data_args,
            training_args=self.training_args,
            finetune_args=finetune_args
        )
        mock_finetune.side_effect = Exception("Permission denied")
        finetune_model(finetune_cfg)
        self.assertEqual(get_latest_error(), ErrorCodes.ERROR_DATASET_CACHE_DIR_NO_WRITE_PERMISSION)

    @unittest.skipIf(get_device_type() != 'cpu', "Only run this test on CPU")
    @patch('intel_extension_for_transformers.llm.finetuning.finetuning.Finetuning.finetune')
    def test_finetune_error_generic(self, mock_finetune):
        model_args = ModelArguments(model_name_or_path="facebook/opt-125m")
        data_args = DataArguments(train_file=test_data_file)
        finetune_args = FinetuningArguments(device=self.device, do_lm_eval=False, peft=None)
        finetune_cfg = TextGenerationFinetuningConfig(
            model_args=model_args,
            data_args=data_args,
            training_args=self.training_args,
            finetune_args=finetune_args
        )
        mock_finetune.side_effect = Exception("Some generic error")
        finetune_model(finetune_cfg)
        self.assertEqual(get_latest_error(), ErrorCodes.ERROR_GENERIC)


class TestOptimizeModelExceptions(unittest.TestCase):
    def setUp(self):
        return super().setUp()

    def tearDown(self) -> None:
        return super().tearDown()

    @unittest.skipIf(get_device_type() != 'cpu', "Only run this test on CPU")
    @patch('intel_extension_for_transformers.llm.quantization.optimization.Optimization.optimize')
    def test_amp_optimize_fail(self,mock_optimize):
        config = MixedPrecisionConfig(dtype="float16" if torch.cuda.is_available() else "bfloat16")
        model = AutoModelForCausalLM.from_pretrained(
                "facebook/opt-125m",
                low_cpu_mem_usage=True,
            )
        mock_optimize.side_effect = Exception("AMP optimization")
        optimize_model(model, config)
        self.assertEqual(get_latest_error(), ErrorCodes.ERROR_AMP_OPTIMIZATION_FAIL)

    @unittest.skipIf(get_device_type() != 'cpu', "Only run this test on CPU")
    @patch('intel_extension_for_transformers.llm.quantization.optimization.Optimization.optimize')
    def test_weight_only_quant_optimize_fail(self,mock_optimize):
        config = WeightOnlyQuantConfig(compute_dtype="int8", weight_dtype="int4")
        model = AutoModelForCausalLM.from_pretrained(
                "facebook/opt-125m",
                low_cpu_mem_usage=True,
            )
        mock_optimize.side_effect = Exception("WOQ optimization")
        optimize_model(model, config)
        self.assertEqual(get_latest_error(), ErrorCodes.ERROR_WEIGHT_ONLY_QUANT_OPTIMIZATION_FAIL)

    @unittest.skipIf(get_device_type() != 'cuda', "Only run this test on CUDA")
    @patch('intel_extension_for_transformers.llm.quantization.optimization.Optimization.optimize')
    def test_bitsandbytes_quant_optimize_fail(self,mock_optimize):
        config = BitsAndBytesConfig(load_in_4bit=True,
                        bnb_4bit_quant_type='nf4',
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_compute_dtype="bfloat16")
        model = AutoModelForCausalLM.from_pretrained(
                "facebook/opt-125m",
                low_cpu_mem_usage=True,
            )
        mock_optimize.side_effect = Exception("BitsAndBytes optimization")
        optimize_model(model, config)
        self.assertEqual(get_latest_error(), ErrorCodes.ERROR_BITS_AND_BYTES_OPTIMIZATION_FAIL)

if __name__ == '__main__':
    unittest.main()
