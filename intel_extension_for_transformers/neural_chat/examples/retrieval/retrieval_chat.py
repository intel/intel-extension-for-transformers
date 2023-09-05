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

import os
import sys
from transformers import TrainingArguments, HfArgumentParser
from intel_extension_for_transformers.neural_chat.config import PipelineConfig
from intel_extension_for_transformers.neural_chat.chatbot import build_chatbot
from intel_extension_for_transformers.neural_chat.plugins import plugins


def main():
    # See all possible arguments in config.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    parser = HfArgumentParser(PipelineConfig)
    
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        pipeline_args= parser.parse_json_file(json_file = os.path.abspath(sys.argv[1]))
    else:
        pipeline_args= parser.parse_args_into_dataclasses()

    plugins.retrieval.enable = True
    plugins.retrieval.args["retrieval_input_path"] = "./Annual_report.pdf"
    pipeline_args.plugins = plugins
    chatbot = build_chatbot(pipeline_args)

    response = chatbot.predict(query="What is IDM 2.0?", config=pipeline_args)
    print(response.choices[0]["text"])

if __name__ == "__main__":
    main()