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
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
streamer = TextStreamer(tokenizer)
# model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=woq_config, trust_remote_code=True)
from intel_extension_for_transformers.llm.runtime.graph import Model
model = Model()
model.init_from_bin("chatglm2", "ne_chatglm2_q.bin", 
                    num_beams=1, max_new_tokens=1024, ctx_size = 128, do_sample=True, threads=28, repetition_penalty=1.1, seed=1, n_keep=2)


# prompt = "[Round 1]\n\n问:你好\n\n答：你好👋！我是人工智能助手 ChatGLM2-6B，很高兴见到你，欢迎问我任何问题。\n\n[Round 2]\n\n问:介绍下Intel公司\n\n答:"
# inputs = tokenizer(prompt, return_tensors="pt").input_ids
# # import pdb; pdb.set_trace()
# inputs_tokens = """
# 64790 64792 30995 30951 517 30910 30939 19364 30916 30974 30916 54761 30954 39701 30974 30916 30974 30916 55437 31211 39701 243 162 148 142 31404 33030 34797 4248 
# 1 22011 10461 30944 30943 30941 30978 30949 31123 48895 35214 54622 31123 32616 39905 31901 31639 31155 30974 30916 30974 30916 30995 30951 517 30910 30943 19364 
# 30916 30974 30916 54761 30954 32025 54578 49030 31629 30974 30916 30974 30916 55437 30954
# """
# inputs_tokens = inputs_tokens.split(" ")
# inputs_tokens = [int(item.strip()) for item in inputs_tokens]
# import torch
# inputs = torch.tensor([inputs_tokens])
# outputs = model.generate(inputs, streamer=streamer, interactive=False, ignore_prompt=True,
#             num_beams=1, max_new_tokens=1024, ctx_size = 1024, do_sample=True, threads=28, repetition_penalty=1.1)
# inputs = tokenizer("once upon a time, there existed a little girl", return_tensors="pt").input_ids
# outputs = model.generate(inputs, streamer=streamer, interactive=False, ignore_prompt=True,
#             num_beams=1, max_new_tokens=1024, ctx_size = 1024, do_sample=True, threads=28, repetition_penalty=1.1)


def process_response(response):
    response = response.strip()
    response = response.replace("[[训练时间]]", "2023年")
    return response
history = []
count = 0
while True:
    print("> ", end="")
    query = input().strip()
    count += 1
    if query == "quit":
        break
    # if count == 1:
    #     b_prompt = "[Round {}]\n\n问：{}\n\n答：".format(count, prompt)  # prompt template for llama2
    # else:
    #     b_prompt = "\n\n[Round {}]\n\n问：{}\n\n答：".format(count, prompt)  # prompt template for llama2
    prompt = tokenizer.build_prompt(query)
    inputs = tokenizer([prompt], return_tensors="pt").input_ids
    outputs = model.generate(inputs, streamer=streamer, interactive=True, ignore_prompt=True,
                num_beams=1, max_new_tokens=1024, ctx_size = 1024, do_sample=True, threads=28, repetition_penalty=1.1)
    response = tokenizer.decode(outputs[0])
    response = process_response(response)
    history = history + [(query, response)]

"""
> 你好
你好👋！我是人工智能助手 ChatGLM2-6B，很高兴见到你，欢迎问我任何问题。
> 介绍下Intel公司
Intel简介:

英特尔(intel)成立于1968年，总部位于美国加利福curr_input_ids: ！我是人工智能助手 ChatGLM2-6B，很高兴见到你，欢迎问我任何问题。

[Round 2]

问：介绍下Intel公司

答： Intel简介:

英特尔(intel)成立于1968年，总部位于美国加利福
看守
> 
"""