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

model_name = "/mnt/disk1/data2/zhenweil/models/chatglm2-6b"  # or local path to model
woq_config = WeightOnlyQuantConfig(compute_dtype="int8", weight_dtype="int4")
prompt = "one +one +one is what"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
# import pdb; pdb.set_trace()
inputs = tokenizer(prompt, return_tensors="pt").input_ids
streamer = TextStreamer(tokenizer)

# model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=woq_config, trust_remote_code=True)
# # import pdb; pdb.set_trace()
# outputs = model.generate(inputs, streamer=streamer, max_new_tokens=300)

prompt = "My name is Jerry, I love to learn new things."
# prompt = "tell me about Intel."
inputs = tokenizer(prompt, return_tensors="pt").input_ids
# outputs = model.generate(inputs, streamer=streamer, max_new_tokens=300)

from intel_extension_for_transformers.llm.runtime.graph import Model
model = Model()
model.init_from_bin("chatglm2", "ne_chatglm2_q.bin", 
                    num_beams=1, max_new_tokens=512, ctx_size = 512, do_sample=True, threads=28, repetition_penalty=1.1) # n_keep=4, ctx_size = 15, n_discard=1 temperature=0.001, top_k=1, top_p=0.95,

count = 1
while True:
    print(">", end="")
    prompt = input()
    b_prompt = "[Round {}]\n\n问：{}\n\n答：".format(count, prompt)
    inputs = tokenizer(b_prompt, return_tensors="pt").input_ids
    outputs = model.generate(inputs, streamer=streamer, interactive=True, ignore_prompt=True)
    count += 1