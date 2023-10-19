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

from fastchat.conversation import get_conv_template, register_conv_template, Conversation, SeparatorStyle

# neuralchat-v2 prompt template
register_conv_template(
    Conversation(
        name="neural-chat-7b-v2",
        system_message="""### System:
- You are a helpful assistant chatbot trained by Intel.
- You answer questions.
- You are excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
- You are more than just an information source, you are also able to write poetry, \
short stories, and make jokes.</s>\n""",
        roles=("### User:", "### Assistant:"),
        sep_style=SeparatorStyle.NO_COLON_TWO,
        sep="\n",
        sep2="</s>",
    )
)

# neuralchat-v1.1 prompt template
register_conv_template(
    Conversation(
        name="neural-chat-7b-v1-1",
        system_template="""<|im_start|>system
{system_message}""",
        system_message="""- You are a helpful assistant chatbot trained by Intel.
- You answer questions.
- You are excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
- You are more than just an information source, you are also able to write poetry, short stories, and make jokes.""",
        roles=("<|im_start|>user", "<|im_start|>assistant"),
        sep_style=SeparatorStyle.CHATML,
        sep="<|im_end|>",
        stop_token_ids=[50278, 0],
    )
)

# Alpaca template without input
register_conv_template(
    Conversation(
        name="alpaca_without_input",
        system_message="Below is an instruction that describes a task. " + \
            "Write a response that appropriately completes the request.",
        roles=("### Instruction", "### Response"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ROBIN,
        sep="\n\n",
    )
)

# Alpaca template with input
register_conv_template(
    Conversation(
        name="alpaca_with_input",
        system_message="Below is an instruction that describes a task, " + \
            "paired with an input that provides further context. " + \
            "Write a response that appropriately completes the request.",
        roles=("### Instruction", "### Input", "### Response"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ROBIN,
        sep="\n\n",
    )
)

# Summarization template
register_conv_template(
    Conversation(
        name="summarization",
        system_message="",
        roles=("", "Summarize the highlights of this article.\n"),
        sep_style=SeparatorStyle.NO_COLON_SINGLE,
        sep="\n",
    )
)

# Rag with context and memory template
register_conv_template(
    Conversation(
        name="rag_with_context_memory",
        system_message="Have a conversation with a human, answer the following questions as best you can." + \
            " You can refer to the following document and context.\n",
        roles=("### Question: ", "### Context: ", "### Chat History: ", "### Response: "),
        sep_style=SeparatorStyle.NO_COLON_SINGLE,
        sep="\n",
    )
)

# Rag without context template
register_conv_template(
    Conversation(
        name="rag_without_context",
        system_message="Have a conversation with a human. " + \
            "You are required to generate suitable response to the user input.\n",
        roles=("### Input: ", "### Response: "),
        sep_style=SeparatorStyle.NO_COLON_SINGLE,
        sep="\n",
    )
)

# Intent template
register_conv_template(
    Conversation(
        name="intent",
        system_message="Please identify the intent of the provided context." + \
            " You may only respond with \"chitchat\" or \"QA\" without explanations" + \
            " or engaging in conversation.\n",
        roles=("Context:", "Intent:"),
        sep_style=SeparatorStyle.NO_COLON_SINGLE,
        sep="\n",
    )
)

# NER template
register_conv_template(
    Conversation(
        name="ner",
        system_message="""Please determine the precise time mentioned in the user's query. 
            Your response should consist only of an accurate time in the format 
            'Time: YYYY-MM-DD' or 'Period: YYYY-MM-DD to YYYY-MM-DD.' 
            If the user query does not include any time reference, please reply with 'None'.\n""",
        roles=("Current Time: ", "User Query: "),
        sep_style=SeparatorStyle.NO_COLON_SINGLE,
        sep="\n",
    )
)

class PromptTemplate:
    def __init__(self, name="one_shot"):
        self.conv = get_conv_template(name)

    @property
    def roles(self):
        return self.conv.roles

    def append_message(self, role: str, message: str):
        self.conv.append_message(role, message)

    def get_prompt(self) -> str:
        return self.conv.get_prompt()
    
