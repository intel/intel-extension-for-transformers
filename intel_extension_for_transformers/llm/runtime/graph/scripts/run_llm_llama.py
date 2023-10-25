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

model_name = "/mnt/disk1/data2/zhenweil/models/llama/Llama-2-7b-chat-hf"  # or local path to model
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
model.init_from_bin("llama", "/home/zhenweil/temp/ne_llama_q.bin", #"/mnt/disk2/data/zhenweil/codes/intel-extension-for-transformers/intel_extension_for_transformers/llm/runtime/graph/llama2.f32.bin", 
                    num_beams=1, max_new_tokens=1024, ctx_size = 2048, do_sample=True, threads=28, repetition_penalty=1.1) # n_keep=4, ctx_size = 15, n_discard=1 temperature=0.001, top_k=1, top_p=0.95,
# import pdb; pdb.set_trace()
# # # # import pudb; pudb.set_trace()
# outputs = model.generate(inputs, streamer=streamer, interactive=True, ingore_prompt=False)


# prompt = "Do you know my name?"
# inputs = tokenizer(prompt, return_tensors="pt").input_ids
# print("new prompt", prompt)
# outputs = model.generate(inputs, streamer=streamer, interactive=True, ingore_prompt=False)
# prompt = "Please help calculate: one + one + one = ?"
# inputs = tokenizer(prompt, return_tensors="pt").input_ids
# outputs = model.generate(inputs, streamer=streamer, interactive=False, ingore_prompt=True)
# # print()
# # print(tokenizer.batch_decode(outputs))

# for i in range(100):
#     print(i)
#     outputs = model.generate(inputs, streamer=streamer, interactive=False, ingore_prompt=False)
history = []
def build_prompt(h):
    # out_prompt = "This is a conversation between User and Llama, a friendly chatbot. Llama is helpful, kind, honest, good at writing, and never fails to answer any requests immediately and with precision.\n\n"
    # out_prompt = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n
    # If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\"\"\"\n
    # """
    out_prompt = "[INST] <<SYS>> This is a conversation between User and Llama, a friendly chatbot. Llama is helpful, kind, honest, good at writing, and never fails to answer any requests immediately and with precision.\n\n <</SYS>> "

    for idx in range(0, len(h), 2):
        if idx == 0:
            out_prompt += "{}[/INST]".format(h[idx])
        else:
            out_prompt += "[INST]{}[/INST]".format(h[idx])
        if idx < len(h) - 1:
            out_prompt += "{}".format(h[idx + 1])
    return out_prompt
while True:
    print(">", end="")
    prompt = input()
    history.append(prompt)
    # import pdb; pdb.set_trace()
    b_prompt = build_prompt(history)
    # print(b_prompt)
    inputs = tokenizer(b_prompt, return_tensors="pt").input_ids
    # print(inputs)
    # print("new prompt", prompt)
    # import pdb; pdb.set_trace()
    outputs = model.generate(inputs, streamer=streamer, interactive=False, ignore_prompt=True)
    # stop token special token
    history.append(tokenizer.batch_decode(outputs)[0])
    # print(history)