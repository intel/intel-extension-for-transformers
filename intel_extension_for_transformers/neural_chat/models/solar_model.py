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
from fastchat.conversation import get_conv_template, Conversation, register_conv_template, SeparatorStyle

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# Solar-10.7B Chat Template
# Reference: https://huggingface.co/upstage/SOLAR-10.7B-Instruct-v1.0/blob/main/tokenizer_config.json
register_conv_template(
    Conversation(
        name="solar",
        system_message="",
        roles=("### User", "### Assistant"),
        sep_style=SeparatorStyle.ADD_COLON_SPACE_SINGLE,
        sep="\n\n",
        stop_str="</s>",
    )
)

class SolarModel(BaseModel):
    def match(self, model_path: str):
        """
        Check if the provided model_path matches the current model.

        Args:
            model_path (str): Path to a model.

        Returns:
            bool: True if the model_path matches, False otherwise.
        """
        return "solar-" in self.model_name.lower() and "instruct" in self.model_name.lower()

    def get_default_conv_template(self) -> Conversation:
        """
        Get the default conversation template for the given model path.

        Args:
            model_path (str): Path to the model.

        Returns:
            Conversation: A default conversation template.
        """
        return get_conv_template("solar")
