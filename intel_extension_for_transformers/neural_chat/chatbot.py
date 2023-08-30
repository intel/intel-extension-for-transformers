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

from intel_extension_for_transformers.llm.finetuning.finetuning import Finetuning
from intel_extension_for_transformers.llm.quantization.optimization import Optimization
from .config import PipelineConfig
from .config import OptimizationConfig
from .config import FinetuningConfig
from .plugins import is_plugin_enabled, get_plugin_instance, get_registered_plugins
from .config import DeviceOptions
from .models.base_model import get_model_adapter
from .utils.common import get_device_type
from .pipeline.plugins.caching.cache import init_similar_cache_from_config
from .pipeline.plugins.audio.asr import AudioSpeechRecognition
from .pipeline.plugins.audio.asr_chinese import ChineseAudioSpeechRecognition
from .pipeline.plugins.audio.tts import TextToSpeech
from .pipeline.plugins.audio.tts_chinese import ChineseTextToSpeech
from .pipeline.plugins.retrievals import QA_Client
from .pipeline.plugins.security.safety_checker import SafetyChecker
from .pipeline.plugins.intent_detector import IntentDetector
from .models.llama_model import LlamaModel
from .models.mpt_model import MptModel
from .models.chatglm_model import ChatGlmModel


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
    if not config:
        config = PipelineConfig()
    # Validate input parameters
    if config.device not in [option.name.lower() for option in DeviceOptions]:
        valid_options = ", ".join([option.name.lower() for option in DeviceOptions])
        raise ValueError(f"Invalid device value '{config.device}'. Must be one of {valid_options}")

    if config.device == "auto":
        config.device = get_device_type()

    # get model adapter
    adapter = get_model_adapter(config.model_name_or_path)

    # register plugin instance in model adaptor
    for plugin_name in get_registered_plugins():
        if is_plugin_enabled(plugin_name):
            plugin_instance = get_plugin_instance(plugin_name)
            if plugin_instance:
                adapter.register_plugin_instance(plugin_name, plugin_instance)

    parameters = {}
    parameters["model_name"] = config.model_name_or_path
    if config.tokenizer_name_or_path:
        parameters["tokenizer_name"] = config.tokenizer_name_or_path
    else:
        parameters["tokenizer_name"] = config.model_name_or_path
    parameters["device"] = config.device
    parameters["use_hpu_graphs"] = config.loading_config.use_hpu_graphs
    parameters["cpu_jit"] = config.loading_config.cpu_jit
    parameters["use_cache"] = config.loading_config.use_cache
    parameters["peft_path"] = config.loading_config.peft_path
    parameters["use_deepspeed"] = config.loading_config.use_deepspeed
    parameters["optimization_config"] = config.optimization_config
    adapter.load_model(parameters)

    return adapter

def finetune_model(config: FinetuningConfig):
    """Finetune the model based on the provided configuration.

    Args:
        config (FinetuningConfig): Configuration for finetuning the model.
    """

    assert config is not None, "FinetuningConfig is needed for finetuning."
    finetuning = Finetuning(config)
    finetuning.finetune()

def optimize_model(model, config: OptimizationConfig):
    """Optimize the model based on the provided configuration.

    Args:
        config (OptimizationConfig): Configuration for optimizing the model.
    """
    optimization = Optimization(optimization_config=config)
    model = optimization.optimize(model)
    return model
