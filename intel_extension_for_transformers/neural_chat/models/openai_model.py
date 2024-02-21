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
        if not config:
            config = GenerationConfig()
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=query,
            temperature=config.temperature,
            top_p=config.top_p,
            max_tokens=config.max_new_tokens,
        )
        return response

    def predict_stream(self, query, config: GenerationConfig = None):
        pass