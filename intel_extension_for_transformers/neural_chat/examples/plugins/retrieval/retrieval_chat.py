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

from intel_extension_for_transformers.neural_chat.config import PipelineConfig
from intel_extension_for_transformers.neural_chat.chatbot import build_chatbot
from intel_extension_for_transformers.neural_chat.plugins import plugins
import logging
logging.basicConfig(
    format="%(asctime)s %(name)s:%(levelname)s:%(message)s",
    datefmt="%d-%M-%Y %H:%M:%S",
    level=logging.INFO
)

def main():
    plugins.retrieval.enable = True
    plugins.retrieval.args["input_path"] = "./Annual_Report.pdf"
    pipeline_args = PipelineConfig(model_name_or_path="Intel/neural-chat-7b-v3-1",
                                   plugins=plugins)
    chatbot = build_chatbot(pipeline_args)

    response = chatbot.predict(query="What is IDM 2.0?")
    logging.info(response)

if __name__ == "__main__":
    main()
