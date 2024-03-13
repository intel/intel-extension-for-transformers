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

from .base_model import BaseModel
import logging
from huggingface_hub import InferenceClient
from ..config import GenerationConfig
from ..errorcode import ErrorCodes
from ..utils.error_utils import set_latest_error
from ..plugins import is_plugin_enabled, get_plugin_instance

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

class HuggingfaceModel(BaseModel):
    def __init__(self, hf_endpoint_url, hf_access_token):
        self.hf_client = InferenceClient(model=hf_endpoint_url, token=hf_access_token)
        self.model_name = hf_endpoint_url
        logger.info(f"HuggingfaceModel initialized.")

    def _predict(self, query, config: GenerationConfig = None, stream=False):
        if not config:
            config = GenerationConfig()

        plugin_name = "retrieval"
        new_user_prompt = query
        if is_plugin_enabled(plugin_name):
            plugin_instance = get_plugin_instance(plugin_name)
            try:
                new_user_prompt, link = plugin_instance.pre_llm_inference_actions(self.model_name, query)
            except Exception as e:
                if "[Rereieval ERROR] intent detection failed" in str(e):
                    set_latest_error(ErrorCodes.ERROR_INTENT_DETECT_FAIL)
                return
        assert new_user_prompt is not None, "Query cannot be None."

        response = self.hf_client.text_generation(prompt=new_user_prompt,
            max_new_tokens=config.max_new_tokens,
            do_sample=config.do_sample,
            repetition_penalty=config.repetition_penalty,
            temperature=config.temperature,
            top_k=config.top_k,
            top_p=config.top_p,
            stream=stream)

        if stream:
            for token in response:
                yield token
        else:
            return response


    def predict(self, query, config: GenerationConfig = None):
        """Customized Huggingface endpoint predict.

        Args:
            query: string, user query
            config: GenerationConfig, provides the needed inference parameters such as top_p, max_tokens.

        Returns:
            the predict string of endpoint
        """
        return self._predict(query=query, config=config, stream=False)

    def predict_stream(self, query, config: GenerationConfig = None):
        """Customized Huggingface endpoint predict_stream.

        Args:
            query: string, user query
            config: GenerationConfig, provides the needed inference parameters such as top_p, max_tokens.

        Returns:
            the predict string of endpoint
        """
        return self._predict(query=query, config=config, stream=True)
