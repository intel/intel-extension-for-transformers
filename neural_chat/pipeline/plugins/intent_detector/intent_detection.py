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

"""Function to check the intent of the input user query with LLM."""
import transformers
import torch
from ..prompts.prompt import generate_intent_prompt
from neural_chat.plugins import register_plugin

@register_plugin("intent_detection")
class IntentDetector:
    def __init__(self):
        self.model = None
        self.tokenizer = None

    def intent_detection(self, query):
        """Using the LLM to detect the intent of the user query."""
        prompt = generate_intent_prompt(query)
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        input_ids = input_ids.to(self.model.device)
        generate_ids = self.model.generate(input_ids, max_new_tokens=5, top_k=1, temperature=0.001)
        intent = self.tokenizer.batch_decode(generate_ids[:, input_ids.shape[1]:], skip_special_tokens=False,
                                        clean_up_tokenization_spaces=False)[0]
        return intent

    def pre_llm_inference_actions(self, query, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        return self.intent_detection(query)
