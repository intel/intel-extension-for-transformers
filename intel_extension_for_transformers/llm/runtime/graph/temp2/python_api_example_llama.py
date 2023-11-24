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

# model_name = "/mnt/disk1/data2/zhenweil/models/gptq/Llama-2-7B-Chat-GPTQ"  # or local path to model
# model_name = "/mnt/disk1/data2/zhenweil/models/mistral/neural-chat-7b-v3-1"
model_name = "/mnt/disk1/data2/zhenweil/models/llama/Llama-2-7b-chat-hf"
woq_config = WeightOnlyQuantConfig(compute_dtype="int8", weight_dtype="int4", use_gptq=False, use_cache=False)
prompt = "Once upon a time, a little girl"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
inputs = tokenizer(prompt, return_tensors="pt").input_ids
streamer = TextStreamer(tokenizer)

model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=woq_config, trust_remote_code=True)
outputs = model.generate(inputs, streamer=streamer)
# while True:
#     prompt = input("> ").strip()
#     if prompt == "quit":
#         break
#     b_prompt = "[INST]{}[/INST]".format(prompt)  # prompt template for llama2
#     inputs = tokenizer(b_prompt, return_tensors="pt").input_ids
#     outputs = model.generate(inputs, streamer=streamer, interactive=True, ignore_prompt=True,
#                 num_beams=1, max_new_tokens=-1, ctx_size = 1024, do_sample=True, threads=28, repetition_penalty=1.1)

