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

import logging

from fastchat.conversation import Conversation, get_conv_template

from .base_model import BaseModel, register_model_adapter

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class ChatGlmModel(BaseModel):
    def match(self, model_path: str):
        """Check if the provided model_path matches the current model.

        Args:
            model_path (str): Path to a model.

        Returns:
            bool: True if the model_path matches, False otherwise.
        """
        return "chatglm" in model_path.lower()

    def get_default_conv_template(self, model_path: str) -> Conversation:
        """Get the default conversation template for the given model path.

        Args:
            model_path (str): Path to the model.

        Returns:
            Conversation: A default conversation template.
        """
        model_path = model_path.lower()
        if "chatglm2" in model_path.lower():
            return get_conv_template("chatglm2")
        return get_conv_template("chatglm")


register_model_adapter(ChatGlmModel)
