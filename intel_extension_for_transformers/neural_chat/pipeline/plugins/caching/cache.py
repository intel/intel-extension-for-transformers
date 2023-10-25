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

# pylint: disable=wrong-import-position
from typing import Any
import os
import gptcache.processor.post
import gptcache.processor.pre
from gptcache import Cache, cache, Config
from gptcache.adapter.adapter import adapt
from gptcache.embedding import (
    Onnx,
    Huggingface,
    SBERT,
    FastText,
    Data2VecAudio,
    Timm,
    ViT,
    OpenAI,
    Cohere,
    Rwkv,
    PaddleNLP,
    UForm,
)
from gptcache.manager import manager_factory
from gptcache.processor.context import (
    SummarizationContextProcess,
    SelectiveContextProcess,
    ConcatContextProcess,
)
from gptcache.processor.post import temperature_softmax
from gptcache.processor.pre import get_prompt
from gptcache.similarity_evaluation import (
    SearchDistanceEvaluation,
    NumpyNormEvaluation,
    OnnxModelEvaluation,
    ExactMatchEvaluation,
    KReciprocalEvaluation
)
from gptcache.utils import import_ruamel
import time

class ChatCache:
    def __init__(self, config_dir: str=os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "./cache_config.yaml"),
            embedding_model_dir: str="hkunlp/instructor-large"):
        self.cache_obj = cache
        self.init_similar_cache_from_config(config_dir, embedding_model_dir)

    def _cache_data_converter(self, cache_data):
        return self._construct_resp_from_cache(cache_data)

    def _construct_resp_from_cache(self, return_message):
        return {
            "gptcache": True,
            "choices": [
                {
                    "text": return_message,
                    "finish_reason": "stop",
                    "index": 0,
                }
            ],
            "created": int(time.time()),
            "usage": {"completion_tokens": 0, "prompt_tokens": 0, "total_tokens": 0},
            "object": "chat.completion",
        }

    def _update_cache_callback_none(self, llm_data, update_cache_func, *args, **kwargs): # pylint: disable=W0613
        return None

    def _llm_handle_none(self, *llm_args, **llm_kwargs): # pylint: disable=W0613
        return None

    def _update_cache_callback(self, llm_data, update_cache_func, *args, **kwargs):
        update_cache_func(llm_data)

    def put(self, prompt: str, data: Any, **kwargs):
        def llm_handle(*llm_args, **llm_kwargs): # pylint: disable=W0613
            return data

        adapt(
            llm_handle,
            self._cache_data_converter,
            self._update_cache_callback,
            cache_skip=False,
            prompt=prompt,
            **kwargs,
        )

    def get(self, prompt: str, **kwargs):
        res = adapt(
            self._llm_handle_none,
            self._cache_data_converter,
            self._update_cache_callback_none,
            prompt=prompt,
            **kwargs,
        )
        return res

    def init_similar_cache(self, data_dir: str = "api_cache", pre_func=get_prompt,
                           embedding=None, data_manager=None, evaluation=None,
                           post_func=temperature_softmax, config=Config()):
        if not embedding:
            embedding = Onnx()
        if not data_manager:
            data_manager = manager_factory(
                "sqlite,faiss",
                data_dir=data_dir,
                vector_params={"dimension": embedding.dimension},
            )
        if not evaluation:
            evaluation = SearchDistanceEvaluation()
        self.cache_obj.init(
            pre_embedding_func=pre_func,
            embedding_func=embedding.to_embeddings,
            data_manager=data_manager,
            similarity_evaluation=evaluation,
            post_process_messages_func=post_func,
            config=config,
        )

    def init_similar_cache_from_config(self, config_dir: str=os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "./cache_config.yaml"),
            embedding_model_dir: str="hkunlp/instructor-large"):
        import_ruamel()
        from ruamel.yaml import YAML # pylint: disable=C0415

        if config_dir:
            with open(config_dir, "r", encoding="utf-8") as f:
                yaml = YAML(typ="unsafe", pure=True)
                init_conf = yaml.load(f)
        else:
            init_conf = {}

        # Due to the problem with the first naming, it is reserved to ensure compatibility
        embedding = init_conf.get("model_source", "")
        if not embedding:
            embedding = init_conf.get("embedding", "onnx")
        # ditto
        embedding_config = init_conf.get("model_config", {})
        # if not embedding_config:
        #     embedding_config = init_conf.get("embedding_config", {})
        embedding_config = {'model': embedding_model_dir}
        embedding_model = self._get_model(embedding, embedding_config)

        storage_config = init_conf.get("storage_config", {})
        storage_config.setdefault("manager", "sqlite,faiss")
        storage_config.setdefault("data_dir", "gptcache_data")
        storage_config.setdefault("vector_params", {})
        storage_config["vector_params"] = storage_config["vector_params"] or {}
        storage_config["vector_params"]["dimension"] = embedding_model.dimension
        data_manager = manager_factory(**storage_config)

        eval_strategy = init_conf.get("evaluation", "distance")
        # Due to the problem with the first naming, it is reserved to ensure compatibility
        eval_config = init_conf.get("evaluation_kws", {})
        if not eval_config:
            eval_config = init_conf.get("evaluation_config", {})
        evaluation = self._get_eval(eval_strategy, eval_config)

        pre_process = init_conf.get("pre_context_function")
        if pre_process:
            pre_func = self._get_pre_context_function(
                pre_process, init_conf.get("pre_context_config")
            )
            pre_func = pre_func.pre_process
        else:
            pre_process = init_conf.get("pre_function", "get_prompt")
            pre_func = self._get_pre_func(pre_process)

        post_process = init_conf.get("post_function", "first")
        post_func = self._get_post_func(post_process)

        config_kws = init_conf.get("config", {}) or {}
        config = Config(**config_kws)

        self.cache_obj.init(
            pre_embedding_func=pre_func,
            embedding_func=embedding_model.to_embeddings,
            data_manager=data_manager,
            similarity_evaluation=evaluation,
            post_process_messages_func=post_func,
            config=config,
        )

    def _get_model(self, model_src, model_config=None):
        model_src = model_src.lower()
        model_config = model_config or {}

        if model_src == "onnx":
            return Onnx(**model_config)
        if model_src == "huggingface":
            return Huggingface(**model_config)
        if model_src == "sbert":
            return SBERT(**model_config)
        if model_src == "fasttext":
            return FastText(**model_config)
        if model_src == "data2vecaudio":
            return Data2VecAudio(**model_config)
        if model_src == "timm":
            return Timm(**model_config)
        if model_src == "vit":
            return ViT(**model_config)
        if model_src == "openai":
            return OpenAI(**model_config)
        if model_src == "cohere":
            return Cohere(**model_config)
        if model_src == "rwkv":
            return Rwkv(**model_config)
        if model_src == "paddlenlp":
            return PaddleNLP(**model_config)
        if model_src == "uform":
            return UForm(**model_config)

    def _get_eval(self, strategy, kws=None):
        strategy = strategy.lower()
        kws = kws or {}
        if "distance" in strategy:
            return SearchDistanceEvaluation(**kws)
        if "np" in strategy:
            return NumpyNormEvaluation(**kws)
        if "exact" in strategy:
            return ExactMatchEvaluation()
        if "onnx" in strategy:
            return OnnxModelEvaluation(**kws)
        if "kreciprocal" in strategy:
            return KReciprocalEvaluation(**kws)

    def _get_pre_func(self, pre_process):
        return getattr(gptcache.processor.pre, pre_process)

    def _get_pre_context_function(self, pre_context_process, kws=None):
        pre_context_process = pre_context_process.lower()
        kws = kws or {}
        if pre_context_process in "summarization":
            return SummarizationContextProcess(**kws)
        if pre_context_process in "selective":
            return SelectiveContextProcess(**kws)
        if pre_context_process in "concat":
            return ConcatContextProcess()

    def _get_post_func(self, post_process):
        return getattr(gptcache.processor.post, post_process)

    def pre_llm_inference_actions(self, prompt):
        return self.get(prompt)

    def post_llm_inference_actions(self, prompt, response):
        self.put(prompt, response)
