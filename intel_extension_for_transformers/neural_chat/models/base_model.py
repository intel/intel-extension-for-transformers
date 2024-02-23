# !/usr/bin/env python
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

from abc import ABC
from typing import List
import os, types
from fastchat.conversation import get_conv_template, Conversation
from ..config import GenerationConfig
from ..plugins import is_plugin_enabled, get_plugin_instance, get_registered_plugins
from ..utils.common import is_audio_file
from .model_utils import load_model, predict, predict_stream, MODELS
from ..prompts import PromptTemplate
from ..prompts.prompt import MAGICODER_PROMPT, generate_sqlcoder_prompt
from ..utils.error_utils import set_latest_error
from ..errorcode import ErrorCodes
import logging
logging.basicConfig(
    format="%(asctime)s %(name)s:%(levelname)s:%(message)s",
    datefmt="%d-%M-%Y %H:%M:%S",
    level=logging.INFO
)


def construct_parameters(query, model_name, device, assistant_model, config):
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
    params["ipex_int8"] = config.ipex_int8
    params["return_stats"] = config.return_stats
    params["format_version"] = config.format_version
    params["assistant_model"] = assistant_model
    params["device"] = device
    return params

class BaseModel(ABC):
    """
    A base class for LLM.
    """

    def __init__(self, model_name, task="chat"):
        """
        Initializes the BaseModel class.
        """
        self.model_name = model_name
        self.asr = None
        self.tts = None
        self.face_animation = None
        self.audio_input_path = None
        self.audio_output_path = None
        self.retriever = None
        self.retrieval_type = None
        self.safety_checker = None
        self.intent_detection = False
        self.cache = None
        self.device = None
        self.conv_template = None
        self.ipex_int8 = None
        self.get_conv_template(task)

    def match(self):
        """
        Check if the provided model_name matches the current model.

        Returns:
            bool: True if the model_name matches, False otherwise.
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
            "ipex_int8": False,
            "use_cache": True,
            "peft_path": "/path/to/peft",
            "use_deepspeed": False,
            "hf_access_token": "user's huggingface access token",
            "assistant_model": "assistant model name to speed up inference",
            "use_vllm": "whether to use vllm for serving",
            "vllm_engine_params": "vllm engine parameters if use_vllm is true",
        }
        """
        self.model_name = kwargs["model_name"]
        self.device = kwargs["device"]
        self.use_hpu_graphs = kwargs["use_hpu_graphs"]
        self.cpu_jit = kwargs["cpu_jit"]
        self.use_cache = kwargs["use_cache"]
        self.ipex_int8 = kwargs["ipex_int8"]
        self.assistant_model = kwargs["assistant_model"]
        load_model(model_name=kwargs["model_name"],
                   tokenizer_name=kwargs["tokenizer_name"],
                   device=kwargs["device"],
                   use_hpu_graphs=kwargs["use_hpu_graphs"],
                   cpu_jit=kwargs["cpu_jit"],
                   ipex_int8=kwargs["ipex_int8"],
                   use_cache=kwargs["use_cache"],
                   peft_path=kwargs["peft_path"],
                   use_deepspeed=kwargs["use_deepspeed"],
                   optimization_config=kwargs["optimization_config"],
                   hf_access_token=kwargs["hf_access_token"],
                   use_neural_speed=kwargs["use_neural_speed"],
                   assistant_model=kwargs["assistant_model"],
                   use_vllm=kwargs["use_vllm"],
                   vllm_engine_params=kwargs["vllm_engine_params"],
                   gguf_model_path=kwargs["gguf_model_path"])

    def predict_stream(self, query, origin_query="", config=None):
        """
        Predict using a streaming approach.

        Args:
            query: The input query for prediction.
            origin_query: The origin Chinese query for safety checker.
            config: Configuration for prediction.
        """
        if not config:
            config = GenerationConfig()

        config.device = self.device
        config.use_hpu_graphs = self.use_hpu_graphs
        config.cpu_jit = self.cpu_jit
        config.use_cache = self.use_cache
        config.ipex_int8 = self.ipex_int8

        my_query = query
        my_origin_query = origin_query

        if is_audio_file(query):
            if not os.path.exists(query):
                raise ValueError(f"The audio file path {query} is invalid.")

        query_include_prompt = False
        if (self.conv_template.roles[0] in query and self.conv_template.roles[1] in query) or \
              "starcoder" in self.model_name.lower() or "codellama" in self.model_name.lower() or \
              "codegen" in self.model_name.lower() or "magicoder" in self.model_name.lower() or \
              "phi-2" in self.model_name.lower() or "sqlcoder" in self.model_name.lower():
            query_include_prompt = True

        # plugin pre actions
        link = []
        for plugin_name in get_registered_plugins():
            if is_plugin_enabled(plugin_name):
                plugin_instance = get_plugin_instance(plugin_name)
                if plugin_instance:
                    if hasattr(plugin_instance, 'pre_llm_inference_actions'):
                        if plugin_name == "cache":
                            response = plugin_instance.pre_llm_inference_actions(query)
                            if response:
                                logging.info("Get response: %s from cache", response)
                                return response['choices'][0]['text'], link
                        if plugin_name == "asr" and not os.path.exists(query):
                            continue
                        if plugin_name == "retrieval":
                            try:
                                response, link = plugin_instance.pre_llm_inference_actions(self.model_name, query)
                                if response == "Response with template.":
                                    return plugin_instance.response_template, link
                            except Exception as e:
                                if "[Rereieval ERROR] intent detection failed" in str(e):
                                    set_latest_error(ErrorCodes.ERROR_INTENT_DETECT_FAIL)
                                return
                        else:
                            try:
                                response = plugin_instance.pre_llm_inference_actions(query)
                            except Exception as e:
                                if plugin_name == "asr":
                                    if "[ASR ERROR] Audio format not supported" in str(e):
                                        set_latest_error(ErrorCodes.ERROR_AUDIO_FORMAT_NOT_SUPPORTED)
                                return
                        if plugin_name == "safety_checker":
                            sign1=plugin_instance.pre_llm_inference_actions(my_query)
                            if sign1:
                                return "Your query contains sensitive words, please try another query.", link
                            if not my_origin_query=="":
                                sign2=plugin_instance.pre_llm_inference_actions(my_origin_query)
                                if sign2:
                                    return "Your query contains sensitive words, please try another query.", link
                        else:
                            if response != None and response != False:
                                query = response
        assert query is not None, "Query cannot be None."

        if not query_include_prompt and not is_plugin_enabled("retrieval"):
            query = self.prepare_prompt(query, config.task)

        # Phind/Phind-CodeLlama-34B-v2 model accpects Alpaca/Vicuna instruction format.
        if "phind" in self.model_name.lower():
            conv_template = PromptTemplate(name="phind")
            conv_template.append_message(conv_template.roles[0], query)
            conv_template.append_message(conv_template.roles[1], None)
            query = conv_template.get_prompt()

        if "magicoder" in self.model_name.lower():
            query = MAGICODER_PROMPT.format(instruction=query)

        if "sqlcoder" in self.model_name.lower():
            query = generate_sqlcoder_prompt(query, config.sql_metadata)

        try:
            response = predict_stream(
                **construct_parameters(query, self.model_name, self.device, self.assistant_model, config))
        except Exception as e:
            set_latest_error(ErrorCodes.ERROR_MODEL_INFERENCE_FAIL)
            return

        def is_generator(obj):
            return isinstance(obj, types.GeneratorType)

        # plugin post actions
        for plugin_name in get_registered_plugins():
            if is_plugin_enabled(plugin_name):
                plugin_instance = get_plugin_instance(plugin_name)
                if plugin_instance:
                    if hasattr(plugin_instance, 'post_llm_inference_actions'):
                        if (plugin_name == "safety_checker" and is_generator(response)) or \
                           plugin_name == "cache":
                            continue
                        response = plugin_instance.post_llm_inference_actions(response)

        return response, link

    def predict(self, query, origin_query="", config=None):
        """
        Predict using a non-streaming approach.

        Args:
            query: The input query for prediction.
            origin_query: The origin Chinese query for safety checker.
            config: Configuration for prediction.
        """
        if not config:
            config = GenerationConfig()

        original_query = query
        config.device = self.device
        config.use_hpu_graphs = self.use_hpu_graphs
        config.cpu_jit = self.cpu_jit
        config.use_cache = self.use_cache
        config.ipex_int8 = self.ipex_int8

        if is_audio_file(query):
            if not os.path.exists(query):
                raise ValueError(f"The audio file path {query} is invalid.")

        query_include_prompt = False
        if (self.conv_template.roles[0] in query and self.conv_template.roles[1] in query) or \
               "starcoder" in self.model_name.lower() or "codellama" in self.model_name.lower() or \
               "codegen" in self.model_name.lower() or "magicoder" in self.model_name.lower() or \
               "sqlcoder" in self.model_name.lower():
            query_include_prompt = True

        # plugin pre actions
        for plugin_name in get_registered_plugins():
            if is_plugin_enabled(plugin_name):
                plugin_instance = get_plugin_instance(plugin_name)
                if plugin_instance:
                    if hasattr(plugin_instance, 'pre_llm_inference_actions'):
                        if plugin_name == "cache":
                            response = plugin_instance.pre_llm_inference_actions(query)
                            if response:
                                logging.info("Get response: %s from cache", response)
                                return response['choices'][0]['text']
                        if plugin_name == "asr" and not os.path.exists(query):
                            continue
                        if plugin_name == "retrieval":
                            try:
                                response, link = plugin_instance.pre_llm_inference_actions(self.model_name, query)
                                if response == "Response with template.":
                                    return plugin_instance.response_template
                            except Exception as e:
                                if "[Rereieval ERROR] intent detection failed" in str(e):
                                    set_latest_error(ErrorCodes.ERROR_INTENT_DETECT_FAIL)
                                return
                        else:
                            try:
                                response = plugin_instance.pre_llm_inference_actions(query)
                            except Exception as e:
                                if plugin_name == "asr":
                                    if "[ASR ERROR] Audio format not supported" in str(e):
                                        set_latest_error(ErrorCodes.ERROR_AUDIO_FORMAT_NOT_SUPPORTED)
                                return
                        if plugin_name == "safety_checker" and response:
                            if response:
                                return "Your query contains sensitive words, please try another query."
                            elif origin_query and plugin_instance.pre_llm_inference_actions(origin_query):
                                return "Your query contains sensitive words, please try another query."
                        else:
                            if response != None and response != False:
                                query = response
        assert query is not None, "Query cannot be None."

        if not query_include_prompt and not is_plugin_enabled("retrieval") \
            and not 'vllm' in str(MODELS[self.model_name]['model']):
            query = self.prepare_prompt(query, config.task)

        # Phind/Phind-CodeLlama-34B-v2 model accpects Alpaca/Vicuna instruction format.
        if "phind" in self.model_name.lower():
            conv_template = PromptTemplate(name="phind")
            conv_template.append_message(conv_template.roles[0], query)
            conv_template.append_message(conv_template.roles[1], None)
            query = conv_template.get_prompt()

        if "magicoder" in self.model_name.lower():
            query = MAGICODER_PROMPT.format(instruction=query)

        if "sqlcoder" in self.model_name.lower():
            query = generate_sqlcoder_prompt(query, config.sql_metadata)

        # LLM inference
        try:
            response = predict(
                **construct_parameters(query, self.model_name, self.device, self.assistant_model, config))
        except Exception as e:
            set_latest_error(ErrorCodes.ERROR_MODEL_INFERENCE_FAIL)
            return

        # plugin post actions
        for plugin_name in get_registered_plugins():
            if is_plugin_enabled(plugin_name):
                plugin_instance = get_plugin_instance(plugin_name)
                if plugin_instance:
                    if hasattr(plugin_instance, 'post_llm_inference_actions'):
                        if plugin_name == "cache":
                            plugin_instance.post_llm_inference_actions(original_query, response)
                        else:
                            response = plugin_instance.post_llm_inference_actions(response)

        return response

    def chat_stream(self, query, origin_query="", config=None):
        """
        Chat using a streaming approach.

        Args:
            query: The input query for prediction.
            origin_query: The origin Chinese query for safety checker.
            config: Configuration for prediction.
        """
        return self.predict_stream(query=query, origin_query=origin_query, config=config)

    def chat(self, query, origin_query="", config=None):
        """
        Chat using a non-streaming approach.

        Args:
            query: The input query for conversation.
            origin_query: The origin Chinese query for safety checker.
            config: Configuration for conversation.
        """
        return self.predict(query=query, origin_query=origin_query, config=config)

    def face_animate(self, image_path, audio_path=None, text=None, voice=None) -> str:  # pragma: no cover
        # 1) if there is a driven audio, then image + audio
        # 2) if there is no driven audio but there is a input text, then first TTS and then image + audio
        if audio_path:
            plugin_name = "face_animation"
            if is_plugin_enabled(plugin_name):
                plugin_instance = get_plugin_instance(plugin_name)
                video_path = plugin_instance.convert(source_image=image_path, driven_audio=audio_path)
            else:
                raise Exception("Please specify the face_animation plugin!")
        elif text:
            plugin_name = "tts"
            if is_plugin_enabled("tts"):
                plugin_name = "tts"
            elif  is_plugin_enabled("tts_chinese"):
                plugin_name = "tts_chinese"
            else:
                raise Exception("Please specify the TTS plugin!")
            plugin_instance = get_plugin_instance(plugin_name)
            audio_path = plugin_instance.text2speech(text, "tmp_audio.wav", voice=voice)
            plugin_instance = get_plugin_instance("face_animation")
            video_path = plugin_instance.convert(source_image=image_path, driven_audio=audio_path)
            os.remove(audio_path)
        return video_path

    def get_default_conv_template(self) -> Conversation:
        """
        Get the default conversation template for the given model path.

        Args:
            model_path (str): Path to the model.

        Returns:
            Conversation: A default conversation template.
        """
        return get_conv_template("zero_shot")

    def get_conv_template(self, model_path: str, task: str = "") -> Conversation:
        """
        Get the conversation template for the given model path or given task.

        Args:
            model_path (str): Path to the model.
            task (str): Task type, one of [completion, chat, summarization].

        Returns:
            Conversation: A conversation template.
        """
        if self.conv_template:
            return
        if not task:
            self.conv_template = PromptTemplate(self.get_default_conv_template().name, clear_history=True)
        else:
            clear_history = True
            if task == "completion":
                name = "alpaca_without_input"
            elif task == "chat":
                clear_history = False
                name = self.get_default_conv_template().name
            elif task == "summarization":
                name = "summarization"
            else:
                raise NotImplementedError(f"Unsupported task {task}.")
            self.conv_template = PromptTemplate(name, clear_history=clear_history)

    def prepare_prompt(self, prompt: str, task: str = ""):
        self.get_conv_template(task)
        self.conv_template.append_message(self.conv_template.roles[0], prompt)
        self.conv_template.append_message(self.conv_template.roles[1], None)
        return self.conv_template.get_prompt()

    def set_customized_system_prompts(self, system_prompts, model_path: str, task: str = ""):
        """Override the system prompts of the model path and the task."""
        if system_prompts is None or len(system_prompts) == 0:
            raise Exception("Please check the model system prompts, should not be None!")
        else:
            self.get_conv_template(model_path, task)
            self.conv_template.conv.system_message = system_prompts

    def register_plugin_instance(self, plugin_name, instance):
        """
        Register a plugin instance.

        Args:
            instance: An instance of a plugin.
        """
        if plugin_name == "tts":
            self.tts = instance
        if plugin_name == "tts_chinese":
            self.tts_chinese = instance
        if plugin_name == "asr":
            self.asr = instance
        if plugin_name == "asr_chinese":
            self.asr_chinese = instance
        if plugin_name == "retrieval":
            self.retrieval = instance
        if plugin_name == "cache":
            self.cache = instance
        if plugin_name == "safety_checker":
            self.safety_checker = instance
        if plugin_name == "face_animation": # pragma: no cover
            self.face_animation = instance
        if plugin_name == "image2image": # pragma: no cover
            self.image2image = instance


# A global registry for all model adapters
model_adapters: List[BaseModel] = []


def register_model_adapter(cls):
    """Register a model adapter."""
    model_adapters.append(cls)


def get_model_adapter(model_name_path: str) -> BaseModel:
    """Get a model adapter for a model_name_path."""
    model_path_basename = os.path.basename(os.path.normpath(model_name_path)).lower()

    for adapter in model_adapters:
        if adapter.match(model_path_basename):
            return adapter

    raise ValueError(f"No valid model adapter for {model_name_path}")
