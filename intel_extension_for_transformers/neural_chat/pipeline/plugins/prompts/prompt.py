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

"""The function for generating the target prompt."""

def generate_qa_prompt(query, context=None, history=None):
    if context and history:
        prompt = """Have a conversation with a human, answer the following questions as best you can. 
        You can refer to the following document and context.\n\n### Question: {}\n\n### Context: {}\n\n### Chat History:{}\n\n### Response:""".format(query, context, history)
    elif context:
        prompt = """Have a conversation with a human, answer the following questions as best you can. 
        You can refer to the following document and context.\n\n### Question: {}\n\n### Context: {}\n\n### Response:""".format(query, context)
    else:
        prompt = """Have a conversation with a human. You are required to generate suitable response to the user input.
            \n\n### Input: {}\n\n### Response:""".format(query)
    
    return prompt


def generate_prompt(query, history=None):
    if history:
        prompt = """Have a conversation with a human. You are required to generate suitable response to the user input.
                        \n\n### Input:{} \n\n###Chat History: {}\n\n### Response:""".format(query, history)
    else:
        prompt = """Have a conversation with a human. You are required to generate suitable response to the user input.
                        \n\n### Input: {}\n\n### Response:""".format(query)
    return prompt


def generate_intent_prompt(query):
    prompt = """Please identify the intent of the provided context. 
        You may only respond with "chitchat" or "QA" without explanations or engaging in conversation.\nContext:{}\nIntent:""".format(query)
    return prompt
