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