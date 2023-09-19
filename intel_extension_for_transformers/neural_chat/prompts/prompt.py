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
        name="neural-chat-7b-v1.1",
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

instruction_template = {
    "prompt_with_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
    ),
    "prompt_without_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:\n"
    ),
}

chat_template = """### System:
- You are a helpful assistant chatbot trained by Intel.
- You answer questions.
- You are excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
- You are more than just an information source, you are also able to write poetry, short stories, and make jokes.{eos_token}
### User:\n{instruction}{eos_token}
### Assistant:
"""

summarization_template = "{instruction}\nSummarize the highlights of this article.\n"


template_maps = {
        "completion": instruction_template,
        "chat": chat_template,
        "summarization": summarization_template
}

def apply_template(example, template):
    if "prompt_with_input" in template:
        prompt_template = (
                template["prompt_with_input"]
                if example["input"] != ""
                else template["prompt_without_input"]
                )
    else:
        prompt_template = template
    prompt = prompt_template.format_map(example)
    return prompt

def prepare_prompt(prompt, task, tokenizer):
    template = "{instruction}"
    if template_maps.get(task) is not None:
        template = template_maps.get(task)
    else:
        NotImplementedError(f'task template is not exist.')
    prompt = apply_template(
        {"instruction": prompt, "input": "", "eos_token": tokenizer.eos_token},
        template
    )
    return prompt

if __name__ == "__main__":
    prompt = "aa aaa"
    tmp = PromptTemplate(name='summarization')
    tmp.append_message(tmp.roles[0], prompt)
    tmp.append_message(tmp.roles[1], "")
    print(tmp.get_prompt() == summarization_template.format(instruction=prompt))
    tmp = PromptTemplate(name='alpaca_with_input')
    tmp.append_message(tmp.roles[0], prompt)
    tmp.append_message(tmp.roles[1], prompt)
    tmp.append_message(tmp.roles[2], "")
    print(tmp.get_prompt() == instruction_template["prompt_with_input"].format(instruction=prompt,input=prompt))
    tmp = PromptTemplate(name='alpaca_without_input')
    tmp.append_message(tmp.roles[0], prompt)
    tmp.append_message(tmp.roles[1], "")
    print(tmp.get_prompt() == instruction_template["prompt_without_input"].format(instruction=prompt))