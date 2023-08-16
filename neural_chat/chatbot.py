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
from .config import PipelineConfig
from .config import OptimizationConfig
from .config import FinetuningConfig
from .pipeline.finetuning.finetuning import Finetuning
from .config import DeviceOptions, BackendOptions, AudioLanguageOptions, RetrievalTypeOptions
from .models.base_model import get_model_adapter
from .utils.common import get_device_type, get_backend_type
from .pipeline.plugins.caching.cache import init_similar_cache_from_config
from .pipeline.plugins.audio.asr import AudioSpeechRecognition
from .pipeline.plugins.audio.asr_chinese import ChineseAudioSpeechRecognition
from .pipeline.plugins.audio.tts import TextToSpeech
from .pipeline.plugins.audio.tts_chinese_tts import ChineseTextToSpeech
from .pipeline.plugins.retrievers.indexing.DocumentParser import DocumentIndexing
from .pipeline.plugins.retrievers.retriever.langchain_retriever import ChromaRetriever
from .pipeline.plugins.retrievers.retriever import BM25Retriever
from .pipeline.plugins.security.SensitiveChecker import SensitiveChecker
from .models.llama_model import LlamaModel
from .models.mpt_model import MptModel
from .models.chatglm_model import ChatGlmModel


def build_chatbot(config: PipelineConfig):
    """Build the chatbot with a given configuration.

    Args:
        config (PipelineConfig): Configuration for building the chatbot.

    Returns:
        adapter: The chatbot model adapter.

    Example:
        from neural_chat.config import PipelineConfig
        from neural_chat.chatbot import build_chatbot
        config = PipelineConfig()
        pipeline = build_chatbot(config)
        response = pipeline.predict(query="Tell me about Intel Xeon Scalable Processors.")
    """
    # Validate input parameters
    if config.device not in [option.name.lower() for option in DeviceOptions]:
        valid_options = ", ".join([option.name.lower() for option in DeviceOptions])
        raise ValueError(f"Invalid device value '{config.device}'. Must be one of {valid_options}")

    if config.backend not in [option.name.lower() for option in BackendOptions]:
        valid_options = ", ".join([option.name.lower() for option in BackendOptions])
        raise ValueError(f"Invalid backend value '{config.backend}'. Must be one of {valid_options}")

    if config.device == "auto":
        config.device = get_device_type()

    if config.backend == "auto":
        config.backend = get_backend_type()

    # get model adapter
    adapter = get_model_adapter(config.model_name_or_path)

    # construct document retrieval using retrieval plugin
    if config.retrieval:
        if config.retrieval_type not in [option.name.lower() for option in RetrievalTypeOptions]:
            valid_options = ", ".join([option.name.lower() for option in RetrievalTypeOptions])
            raise ValueError(f"Invalid retrieval type value '{config.retrieval_type}'. Must be one of {valid_options}")
        if not config.retrieval_document_path:
            raise ValueError("Must provide a retrieval document path")
        if not os.path.exists(config.retrieval_document_path):
            raise ValueError(f"The retrieval document path {config.retrieval_document_path} is not exist.")
        db = DocumentIndexing(config.retrieval_type).KB_construct(config.retrieval_document_path)
        if config.retrieval_type == "dense":
            retriever = ChromaRetriever(db).retriever
        else:
            retriever = BM25Retriever(document_store = db)
        adapter.register_retriever(retriever, config.retrieval_type)

    # construct audio plugin
    if config.audio_input or config.audio_output:
        if config.audio_lang not in [option.name.lower() for option in AudioLanguageOptions]:
            valid_options = ", ".join([option.name.lower() for option in AudioLanguageOptions])
            raise ValueError(f"Invalid audio language value '{config.audio_lang}'. Must be one of {valid_options}")
        if config.audio_input:
            if config.audio_lang == AudioLanguageOptions.CHINESE.name.lower():
                asr = ChineseAudioSpeechRecognition()
            else:
                asr = AudioSpeechRecognition()
            adapter.register_asr(asr)
        if config.audio_output:
            if config.audio_lang == AudioLanguageOptions.CHINESE.name.lower():
                tts = ChineseTextToSpeech()
            else:
                tts = TextToSpeech()
            adapter.register_tts(tts)

    # construct response caching
    if config.cache_chat:
        if not config.cache_chat_config_file:
            cache_chat_config_file = "./pipeline/plugins/caching/cache_config.yaml"
        else:
            cache_chat_config_file = config.cache_chat_config_file
        if not config.cache_embedding_model_dir:
            cache_embedding_model_dir = "hkunlp/instructor-large"
        else:
            cache_embedding_model_dir = config.cache_embedding_model_dir
        init_similar_cache_from_config(config_dir=cache_chat_config_file,
                                       embedding_model_dir=cache_embedding_model_dir)

    # construct safety checker
    if config.safety_checker:
        safety_checker = SensitiveChecker()
        adapter.register_safety_checker(safety_checker)

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

def optimize_model(config: OptimizationConfig):
    """Optimize the model based on the provided configuration.

    Args:
        config (OptimizationConfig): Configuration for optimizing the model.
    """
    pass
