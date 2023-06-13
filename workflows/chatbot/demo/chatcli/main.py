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

import argparse
import sys

import model

parser = argparse.ArgumentParser(prog='neural chat command line tool')
parser.add_argument("--model", choices=['gpt-3.5-turbo', 'llama-7b-hf'], default='gpt-3.5-turbo')
parser.add_argument("--endpoint", default=None, type=str)
parser.add_argument("--key", default=None, type=str)

args = parser.parse_args()

bot: model.ChatModel = None

if args.model == 'gpt-3.5-turbo':
    if args.endpoint is not None:
        raise "OpenAI model does not need provide endpoint"
    if args.key is None:
        raise "OpenAI model need provide api key"
    bot = model.OpenAIModel(args.key)
else:
    if args.endpoint is None:
        raise "intel model need provide endpoint"
    if args.key is not None:
        raise "intel model does not need provide api key"
    bot = model.IntelModel(args.endpoint, args.model, "Human", "Assistant")

while True:
    sys.stdout.write("user > ")
    msg = input()
    sys.stdout.write("\nbot  > ")
    bot.interact(msg)
    sys.stdout.write("\n\n")
