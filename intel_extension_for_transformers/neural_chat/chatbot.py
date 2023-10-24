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

def prepare_env(device):
    if device == "hpu":
        os.environ.setdefault("PT_HPU_LAZY_ACC_PAR_MODE", "0")
        os.environ.setdefault("PT_HPU_ENABLE_LAZY_COLLECTIVES", "true")

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
    global plugins
    if not config:
        config = PipelineConfig()
    # Validate input parameters
    if config.device not in [option.name.lower() for option in DeviceOptions]:
        valid_options = ", ".join([option.name.lower() for option in DeviceOptions])
        raise ValueError(f"Invalid device value '{config.device}'. Must be one of {valid_options}")

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
        raise ValueError("NeuralChat Error: Unsupported model name or path, \
                         only supports FLAN-T5/LLAMA/MPT/GPT/BLOOM/OPT/QWEN/NEURAL-CHAT now.")

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
                    raise ValueError("NeuralChat Error: Unsupported plugin")
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
    parameters["optimization_config"] = config.optimization_config
    parameters["hf_access_token"] = config.hf_access_token

    # Set necessary env variables
    prepare_env(config.device)
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

def optimize_model(model, config):
    """Optimize the model based on the provided configuration.

    Args:
        config (OptimizationConfig): Configuration for optimizing the model.
    """
    optimization = Optimization(optimization_config=config)
    model = optimization.optimize(model)
    return model
