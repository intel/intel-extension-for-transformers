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

import re
from typing import Union
from intel_extension_for_transformers.transformers import (
    AutoModel,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    GPTBigCodeForCausalLM,
    MixedPrecisionConfig,
    WeightOnlyQuantConfig,
    BitsAndBytesConfig
)

class Optimization:
    def __init__(
            self,
            optimization_config: Union[MixedPrecisionConfig, WeightOnlyQuantConfig, BitsAndBytesConfig]
        ):
        self.optimization_config = optimization_config

    def optimize(self, model, use_llm_runtime=False):
        optimized_model = model
        config = self.optimization_config
        if re.search("flan-t5", model.config._name_or_path, re.IGNORECASE):
                optimized_model = AutoModelForSeq2SeqLM.from_pretrained(
                    model.config._name_or_path,
                    quantization_config=config,
                    use_llm_runtime=use_llm_runtime,
                    trust_remote_code=True)
        elif (
            re.search("gpt", model.config._name_or_path, re.IGNORECASE)
            or re.search("mpt", model.config._name_or_path, re.IGNORECASE)
            or re.search("bloom", model.config._name_or_path, re.IGNORECASE)
            or re.search("llama", model.config._name_or_path, re.IGNORECASE)
            or re.search("opt", model.config._name_or_path, re.IGNORECASE)
            or re.search("neural-chat-7b-v1", model.config._name_or_path, re.IGNORECASE)
            or re.search("neural-chat-7b-v2", model.config._name_or_path, re.IGNORECASE)
        ):
            optimized_model = AutoModelForCausalLM.from_pretrained(
                model.config._name_or_path,
                quantization_config=config,
                use_llm_runtime=use_llm_runtime,
                trust_remote_code=True)
        elif re.search("starcoder", model.config._name_or_path, re.IGNORECASE):
            optimized_model = GPTBigCodeForCausalLM.from_pretrained(
                model.config._name_or_path,
                quantization_config=config,
                use_llm_runtime=use_llm_runtime,
                trust_remote_code=True)
        elif re.search("chatglm", model.config._name_or_path, re.IGNORECASE):
            optimized_model = AutoModel.from_pretrained(
                model.config._name_or_path,
                quantization_config=config,
                use_llm_runtime=use_llm_runtime,
                trust_remote_code=True)
        return optimized_model
