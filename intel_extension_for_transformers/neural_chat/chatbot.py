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
"""Neural Chat Chatbot API."""

import os
from intel_extension_for_transformers.llm.finetuning.finetuning import Finetuning
from intel_extension_for_transformers.llm.quantization.optimization import Optimization
from .config import PipelineConfig
from .config import BaseFinetuningConfig
from .config import DeviceOptions
from .plugins import plugins

from intel_extension_for_transformers.utils import logger
from .constants import ResponseCodes, MEMORY_THRESHOLD_GB, STORAGE_THRESHOLD_GB, GPU_MEMORY_THRESHOLD_MB
import psutil
import torch


def build_chatbot(config: PipelineConfig=None):
    """Build the chatbot with a given configuration.

    Args:
        config (PipelineConfig): Configuration for building the chatbot.

    Returns:
        adapter: The chatbot model adapter.

    Example:
        from intel_extension_for_transformers.neural_chat import build_chatbot
        pipeline = build_chatbot()
        response = pipeline.predict(query="Tell me about Intel Xeon Scalable Processors.")
    """
    # Check for out of memory
    available_memory = psutil.virtual_memory().available / (1024 ** 3)
    if available_memory < MEMORY_THRESHOLD_GB: # The 4-bit 7B model requires a minimum of 7GB of memory
        logger.error("LLM requires a minimum of 8GB of free system memory, \
                   but the current available memory is insufficient.")
        return ResponseCodes.ERROR_OUT_OF_MEMORY

    # Check for out of storage
    available_storage = psutil.disk_usage('/').free
    available_storage_gb = available_storage / (1024 ** 3)
    if available_storage_gb < STORAGE_THRESHOLD_GB:
        logger.error("LLM requires a minimum of 30GB of free system storage, \
                   but the current available storage is insufficient.")
        return ResponseCodes.ERROR_OUT_OF_STORAGE

    global plugins
    if not config:
        config = PipelineConfig()
    # Validate input parameters
    if config.device not in [option.name.lower() for option in DeviceOptions]:
        valid_options = ", ".join([option.name.lower() for option in DeviceOptions])
        logger.error(f"Invalid device value '{config.device}'. Must be one of {valid_options}")
        return ResponseCodes.ERROR_DEVICE_NOT_SUPPORTED

    if config.device == "gpu":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            remaining_memory = torch.cuda.get_device_properties(device).total_memory - \
                               torch.cuda.memory_allocated(device)
            remaining_memory_gb = remaining_memory / (1024 ** 3)
            if remaining_memory_gb < GPU_MEMORY_THRESHOLD_MB:
                log.error("LLM requires a minimum of 6GB of free GPU memory, \
                           but the current available GPU memory is insufficient.")
                return ResponseCodes.ERROR_OUT_OF_MEMORY

    # create model adapter
    if "llama" in config.model_name_or_path.lower():
        from .models.llama_model import LlamaModel
        adapter = LlamaModel()
    elif "mpt" in config.model_name_or_path:
        from .models.mpt_model import MptModel
        adapter = MptModel()
    elif "neural-chat" in config.model_name_or_path:
        from .models.neuralchat_model import NeuralChatModel
        adapter = NeuralChatModel()
    elif "chatglm" in config.model_name_or_path:
        from .models.chatglm_model import ChatGlmModel
        adapter = ChatGlmModel()
    elif "Qwen" in config.model_name_or_path:
        from .models.qwen_model import QwenModel
        adapter = QwenModel()
    elif "opt" in config.model_name_or_path or \
         "gpt" in config.model_name_or_path or \
         "flan-t5" in config.model_name_or_path or \
         "bloom" in config.model_name_or_path or \
         "starcoder" in config.model_name_or_path:
        from .models.base_model import BaseModel
        adapter = BaseModel()
    else:
        logger.error(f"Unsupported model name or path {config.model_name_or_path}, \
                   only supports FLAN-T5/LLAMA/MPT/GPT/BLOOM/OPT/QWEN/NEURAL-CHAT now.")
        return ResponseCodes.ERROR_MODEL_NOT_SUPPORTED

    # register plugin instance in model adaptor
    if config.plugins:
        for plugin_name, plugin_value in config.plugins.items():
            enable_plugin = plugin_value.get('enable', False)
            if enable_plugin:
                if plugin_name == "tts":
                    from .pipeline.plugins.audio.tts import TextToSpeech
                    plugins[plugin_name]['class'] = TextToSpeech
                elif plugin_name == "tts_chinese":
                    from .pipeline.plugins.audio.tts_chinese import ChineseTextToSpeech
                    plugins[plugin_name]['class'] = ChineseTextToSpeech
                elif plugin_name == "asr":
                    from .pipeline.plugins.audio.asr import AudioSpeechRecognition
                    plugins[plugin_name]['class'] = AudioSpeechRecognition
                elif plugin_name == "asr_chinese":
                    from .pipeline.plugins.audio.asr_chinese import ChineseAudioSpeechRecognition
                    plugins[plugin_name]['class'] = ChineseAudioSpeechRecognition
                elif plugin_name == "retrieval":
                    from .pipeline.plugins.retrieval.retrieval_agent import Agent_QA
                    plugins[plugin_name]['class'] = Agent_QA
                elif plugin_name == "cache":
                    from .pipeline.plugins.caching.cache import ChatCache
                    plugins[plugin_name]['class'] = ChatCache
                elif plugin_name == "safety_checker":
                    from .pipeline.plugins.security.safety_checker import SafetyChecker
                    plugins[plugin_name]['class'] = SafetyChecker
                elif plugin_name == "ner":
                    from .pipeline.plugins.ner.ner import NamedEntityRecognition
                    plugins[plugin_name]['class'] = NamedEntityRecognition
                elif plugin_name == "ner_int":
                    from .pipeline.plugins.ner.ner_int import NamedEntityRecognitionINT
                    plugins[plugin_name]['class'] = NamedEntityRecognitionINT
                else:
                    logger.error(f"Unsupported plugin: {plugin_name}")
                    return ResponseCodes.ERROR_PLUGIN_NOT_SUPPORTED
                print(f"create {plugin_name} plugin instance...")
                print(f"plugin parameters: ", plugin_value['args'])
                plugins[plugin_name]["instance"] = plugins[plugin_name]['class'](**plugin_value['args'])
                adapter.register_plugin_instance(plugin_name, plugins[plugin_name]["instance"])

    parameters = {}
    parameters["model_name"] = config.model_name_or_path
    if config.tokenizer_name_or_path:
        parameters["tokenizer_name"] = config.tokenizer_name_or_path
    else:
        parameters["tokenizer_name"] = config.model_name_or_path
    parameters["device"] = config.device
    parameters["use_hpu_graphs"] = config.loading_config.use_hpu_graphs
    parameters["cpu_jit"] = config.loading_config.cpu_jit
    parameters["ipex_int8"] = config.loading_config.ipex_int8
    parameters["use_cache"] = config.loading_config.use_cache
    parameters["peft_path"] = config.loading_config.peft_path
    parameters["use_deepspeed"] = config.loading_config.use_deepspeed
    parameters["use_llm_runtime"] = config.loading_config.use_llm_runtime
    parameters["optimization_config"] = config.optimization_config
    parameters["hf_access_token"] = config.hf_access_token

    adapter.load_model(parameters)

    return adapter

def finetune_model(config: BaseFinetuningConfig):
    """Finetune the model based on the provided configuration.

    Args:
        config (BaseFinetuningConfig): Configuration for finetuning the model.
    """

    assert config is not None, "BaseFinetuningConfig is needed for finetuning."
    finetuning = Finetuning(config)
    finetuning.finetune()

def optimize_model(model, config, use_llm_runtime=False):
    """Optimize the model based on the provided configuration.

    Args:
        model: large language model
        config (OptimizationConfig): The configuration required for optimizing the model.
        use_llm_runtime (bool): A boolean indicating whether to use the LLM runtime graph optimization.
    """
    optimization = Optimization(optimization_config=config)
    model = optimization.optimize(model, use_llm_runtime)
    return model
