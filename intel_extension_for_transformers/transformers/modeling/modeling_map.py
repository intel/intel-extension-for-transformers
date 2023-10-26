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

# coding=utf-8
# Copyright 2021 The EleutherAI and HuggingFace Teams. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import transformers
from .gptj.modeling_gptj import GPTJForCausalLM
from .llama.modeling_llama import LlamaForCausalLM
from .bloom.modeling_bloom import BloomForCausalLM
from .gpt_neox.modeling_gpt_neox import GPTNeoXForCausalLM
from .opt.modeling_opt import OPTForCausalLM
from .gpt_bigcode.modeling_gpt_bigcode import GPTBigCodeForCausalLM
from .mistral.modeling_mistral import MistralForCausalLM
# to use modeling modification base transformers 4.30.2:
transformers.models.gpt_bigcode.modeling_gpt_bigcode.GPTBigCodeForCausalLM = GPTBigCodeForCausalLM
# to use modeling modification base transformers 4.28.1:
transformers.models.gptj.modeling_gptj.GPTJForCausalLM = GPTJForCausalLM
transformers.models.llama.modeling_llama.LlamaForCausalLM = LlamaForCausalLM
transformers.models.bloom.modeling_bloom.BloomForCausalLM = BloomForCausalLM
transformers.models.gpt_neox.modeling_gpt_neox.GPTNeoXForCausalLM = GPTNeoXForCausalLM
transformers.models.opt.modeling_opt.OPTForCausalLM = OPTForCausalLM
try:
    transformers.models.mistral.modeling_mistral.MistralForCausalLM = MistralForCausalLM
except:
    print("Please install transformers >=4.34.0 if you want to run Mistral model.")
