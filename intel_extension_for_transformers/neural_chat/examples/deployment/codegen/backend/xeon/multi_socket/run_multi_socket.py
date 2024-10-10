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


import os
import torch
import deepspeed
from deepspeed.accelerator import get_accelerator
from transformers import pipeline, AutoTokenizer


dtype = torch.bfloat16
local_rank = int(os.getenv("LOCAL_RANK", "0"))
model = "facebook/opt-125m"
task = "text-generation"
world_size = 2

# We have to load these large models on CPU with pipeline because not
# enough GPU memory
tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
pipe = pipeline(task,
                model=model,
                tokenizer=tokenizer,
                torch_dtype=dtype,
                trust_remote_code=True,
                device=torch.device("cpu"),
                framework="pt")

pipe.model = deepspeed.init_inference(pipe.model, dtype=dtype, mp_size=world_size, replace_with_kernel_inject=False)
ds_config = {
    "weight_quantization": {
        "post_init_quant": {
            '*': {
                'num_bits': 4,
                'group_size': 32,
                'group_dim': 1,
                'symmetric': False
            },
        }
    }
}
from deepspeed.inference.quantization.quantization import _init_group_wise_weight_quantization
pipe.model = _init_group_wise_weight_quantization(pipe.model, ds_config)
pipe.device = torch.device(get_accelerator().device_name(local_rank))

query = "DeepSpeed is the greatest"
inf_kwargs = {"do_sample": False, "temperature": 1.0, "max_length": 20}
ds_output = pipe(query, **inf_kwargs)

print(local_rank, "deepspeed", ds_output)