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

from transformers import AutoTokenizer, TextStreamer
from intel_extension_for_transformers.transformers import AutoModelForCausalLM, WeightOnlyQuantConfig

model_name = "/mnt/disk1/data2/zhenweil/models/llama/Llama-2-7b-chat-hf"  # or local path to model # meta-llama/Llama-2-7b-chat-hf
woq_config = WeightOnlyQuantConfig(compute_dtype="int8", weight_dtype="int4")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
streamer = TextStreamer(tokenizer)
model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=woq_config, trust_remote_code=True)

history = []
def build_prompt(h):
    out_prompt = ""
    for idx in range(0, len(h), 2):
        out_prompt += "[INST]{}[/INST]".format(h[idx])
        if idx < len(h) - 1:
            out_prompt += "{}".format(h[idx + 1])
    return out_prompt

while True:
    print("> ", end="")
    prompt = input().strip()
    history.append(prompt)
    b_prompt = build_prompt(history)
    inputs = tokenizer(b_prompt, return_tensors="pt").input_ids
    outputs = model.generate(inputs, streamer=streamer, interactive=False, ignore_prompt=True,
                                     num_beams=1, max_new_tokens=1024, ctx_size = 2048, do_sample=True, threads=28, repetition_penalty=1.1)
    history.append(tokenizer.batch_decode(outputs)[0])
