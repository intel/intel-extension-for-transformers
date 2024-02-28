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
from openai import OpenAI
import logging
from ..config import GenerationConfig
from ..plugins import is_plugin_enabled, get_plugin_instance, get_registered_plugins
import os
from ..utils.error_utils import set_latest_error
from ..errorcode import ErrorCodes


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

class OpenAIModel(BaseModel):
    """
    Customized class to operate on OpenAI models in the pipeline.
    """
    def __init__(self, model_name, task, openai_config):
        self.api_key = openai_config.api_key
        self.organization = openai_config.organization
        self.model_name = model_name
        self.task = task

    def load_model(self, kwargs: dict):
        self.client = OpenAI(api_key=self.api_key, organization=self.organization)

    def predict(self, query, config: GenerationConfig = None):
        """Customized OpenAI model predict.

        Args:
            query: List[Dict], usually contains system prompt + user prompt.
            config: GenerationConfig, provides the needed inference parameters such as top_p, max_tokens.

        Returns:
            the result string of one single choice
        """
        if not config:
            config = GenerationConfig()

        # Only supported retrieval plugin for now
        plugin_name = "retrieval"
        if is_plugin_enabled(plugin_name):
            plugin_instance = get_plugin_instance(plugin_name)
            try:
                new_user_prompt, link = plugin_instance.pre_llm_inference_actions(self.model_name,
                                                                                  self.find_user_prompt(query))
                self.update_user_prompt(query, new_user_prompt)
            except Exception as e:
                if "[Rereieval ERROR] intent detection failed" in str(e):
                    set_latest_error(ErrorCodes.ERROR_INTENT_DETECT_FAIL)
                return
        assert query is not None, "Query cannot be None."
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=query,
            temperature=config.temperature,
            top_p=config.top_p,
            max_tokens=config.max_new_tokens,
        )
        return response.choices[0].message.content

    def find_user_prompt(self, query):
        """Find in the query List[Dict] the user prompt."""
        return [i['content'] for i in query if 'role' in i and i['role'] == 'user'][0]

    def update_user_prompt(self, query, new_user_prompt):
        """Update the user prompt in the query List[Dict]."""
        for i in query:
            if 'role' in i and i['role'] == 'user':
                i['content'] = new_user_prompt
        return query

    def predict_stream(self, query, config: GenerationConfig = None):
        raise Exception("Currently not support streaming! Will fix this in the future.")
