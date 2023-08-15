
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

from abc import ABC, abstractmethod
from typing import List
import os
from fastchat.conversation import get_conv_template, Conversation
from neural_chat.pipeline.inference.inference import load_model, predict, predict_stream
from neural_chat.config import GenerationConfig

def construct_parameters(query, model_name, config):
    params = {}
    params["prompt"] = query
    params["temperature"] = config.temperature
    params["top_k"] = config.top_k
    params["top_p"] = config.top_p
    params["repetition_penalty"] = config.repetition_penalty
    params["max_new_tokens"] = config.max_new_tokens
    params["do_sample"] = config.do_sample
    params["num_beams"] = config.num_beams
    params["model_name"] = model_name
    params["num_return_sequences"] = config.num_return_sequences
    params["bad_words_ids"] = config.bad_words_ids
    params["force_words_ids"] = config.force_words_ids
    params["use_hpu_graphs"] = config.use_hpu_graphs
    params["use_cache"] = config.use_cache
    return params

class BaseModel(ABC):
    """
    A base class for LLM.
    """

    def __init__(self):
        """
        Initializes the BaseModel class.
        """
        self.model_name = ""
        self.asr = None
        self.tts = None
        self.audio_input_path = None
        self.audio_output_path = None

    def match(self, model_path: str):
        """
        Check if the provided model_path matches the current model.

        Args:
            model_path (str): Path to a model.

        Returns:
            bool: True if the model_path matches, False otherwise.
        """
        return True

    def load_model(self, kwargs: dict):
        """
        Load the model using the provided arguments.

        Args:
            kwargs (dict): A dictionary containing the configuration parameters for model loading.

        Example 'kwargs' dictionary:
        {
            "model_name": "my_model",
            "tokenizer_name": "my_tokenizer",
            "device": "cuda",
            "use_hpu_graphs": True,
            "cpu_jit": False,
            "use_cache": True,
            "peft_path": "/path/to/peft",
            "use_deepspeed": False
        }
        """
        self.model_name = kwargs["model_name"]
        load_model(model_name=kwargs["model_name"],
                   tokenizer_name=kwargs["tokenizer_name"],
                   device=kwargs["device"],
                   use_hpu_graphs=kwargs["use_hpu_graphs"],
                   cpu_jit=kwargs["cpu_jit"],
                   use_cache=kwargs["use_cache"],
                   peft_path=kwargs["peft_path"],
                   use_deepspeed=kwargs["use_deepspeed"])

    def predict_stream(self, query, config=None):
        """
        Predict using a streaming approach.

        Args:
            query: The input query for prediction.
            config: Configuration for prediction.
        """
        if not config:
            config = GenerationConfig()
        return predict_stream(**construct_parameters(query, self.model_name, config))

    def predict(self, query=None, config=None):
        """
        Predict using a non-streaming approach.

        Args:
            query: The input query for prediction.
            config: Configuration for prediction.
        """
        if not config:
            config = GenerationConfig()

        if self.asr and self.audio_input_path:
            query = self.asr.audio2text(config.audio_input_path)
        assert query is not None, "Query cannot be None."
        response = predict(**construct_parameters(query, self.model_name, config))
        if self.tts and self.audio_output_path:
            self.tts.text2audio(response, config.audio_output_path)
            response = config.audio_output_path
        return response

    def chat_stream(self, query, config=None):
        """
        Chat using a streaming approach.

        Args:
            query: The input query for prediction.
            config: Configuration for prediction.
        """
        if not config:
            config = GenerationConfig()
        return predict_stream(**construct_parameters(query, self.model_name, config))

    def chat(self, query, config=None):
        """
        Chat using a non-streaming approach.

        Args:
            query: The input query for conversation.
            config: Configuration for conversation.
        """
        if not config:
            config = GenerationConfig()

        if self.asr and self.audio_input_path:
            query = self.asr.audio2text(config.audio_input_path)
        assert query is not None, "Query cannot be None."
        response = predict(**construct_parameters(query, self.model_name, config))
        if self.tts and self.audio_output_path:
            self.tts.text2audio(response, config.audio_output_path)
            response = config.audio_output_path
        return response

    def get_default_conv_template(self, model_path: str) -> Conversation:
        """
        Get the default conversation template for the given model path.

        Args:
            model_path (str): Path to the model.

        Returns:
            Conversation: A default conversation template.
        """
        return get_conv_template("one_shot")

    def register_tts(self, instance, audio_output_path):
        """
        Register a text-to-speech (TTS) instance.

        Args:
            instance: An instance of a TTS module.
        """
        self.tts = instance
        self.audio_output_path = audio_output_path

    def register_asr(self, instance, audio_input_path):
        """
        Register an automatic speech recognition (ASR) instance.

        Args:
            instance: An instance of an ASR module.
        """
        self.asr = instance
        self.audio_input_path = audio_input_path

    def register_safety_checker(self, instance):
        """
        Register a safety checker instance.

        Args:
            instance: An instance of a safety checker module.
        """
        self.safety_checker = instance


# A global registry for all model adapters
model_adapters: List[BaseModel] = []

def register_model_adapter(cls):
    """Register a model adapter."""
    model_adapters.append(cls())


def get_model_adapter(model_name_path: str) -> BaseModel:
    """Get a model adapter for a model_name_path."""
    model_path_basename = os.path.basename(os.path.normpath(model_name_path)).lower()

    for adapter in model_adapters:
        if adapter.match(model_path_basename) and type(adapter) != BaseModel:
            return adapter

    raise ValueError(f"No valid model adapter for {model_name_path}")
