# !/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2024 Intel Corporation
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

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

MODEL_NAME = "spycsh/shanghainese-opus-zh-sh-4000"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)

def generate_translation(model, tokenizer, example):
    """Mandarin => Shanghainese."""
    input_ids = example['input_ids']
    input_ids = torch.LongTensor(input_ids).view(1, -1).to(model.device)
    generated_ids = model.generate(input_ids, max_new_tokens=64)
    prediction = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    print('prediction: ', prediction)

text = '你是公派出去还是因私出国'
model_inputs = tokenizer(text, max_length=64, truncation=True)
example = {}

example['input_ids'] = model_inputs['input_ids']
generate_translation(model, tokenizer, example)
