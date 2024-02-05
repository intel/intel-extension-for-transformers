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
from fastchat.conversation import get_conv_template, Conversation

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

class NeuralChatModel(BaseModel):
    def match(self):
        """
        Check if the provided model_path matches the current model.

        Returns:
            bool: True if the model_path matches, False otherwise.
        """
        return "neural-chat" in self.model_name.lower()

    def get_default_conv_template(self) -> Conversation:
        """
        Get the default conversation template for the given model path.

        Returns:
            Conversation: A default conversation template.
        """
        if "neural-chat-7b-v2" in self.model_name.lower():
            return get_conv_template("neural-chat-7b-v2")
        elif "neural-chat-7b-v3" in self.model_name.lower():
            return get_conv_template("neural-chat-7b-v3")
        else:
            return get_conv_template("neural-chat-7b-v1-1")
