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


import argparse
from intel_extension_for_transformers.neural_chat import GenerationConfig
from intel_extension_for_transformers.neural_chat.chatbot import build_chatbot

def main():

    parser = argparse.ArgumentParser(prog='text chat example', add_help=True)
    parser.add_argument(
        "--prompt",
        action="store",
        help="text prompt for neuralchat",
        default=None,
        required=True)

    args = parser.parse_args()
    chatbot = build_chatbot()
    config = GenerationConfig(max_new_tokens=64)
    response = chatbot.predict(
        query=args.prompt, 
        config=config
    )
    print(response)

if __name__ == "__main__":
    main()
