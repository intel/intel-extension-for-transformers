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
        if isinstance(config, WeightOnlyQuantConfig):
            print("Applying Weight Only Quantization.")
            if use_llm_runtime:
                if re.search("flan-t5", model.config._name_or_path, re.IGNORECASE):
                        optimized_model = AutoModelForSeq2SeqLM.from_pretrained(
                            model.config._name_or_path,
                            quantization_config=config,
                            use_llm_runtime=True,
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
                        use_llm_runtime=True,
                        trust_remote_code=True)
            else:
                from neural_compressor import PostTrainingQuantConfig, quantization
                if config.weight_dtype is None:
                    config.weight_dtype = 'int4_fullrange'
                bits = 1  # only for int8
                if config.weight_dtype == "int8":
                    dtype = "int8"
                    bits = 8
                elif "int4" in config.weight_dtype:
                    dtype = "int4"
                else:
                    dtype = config.weight_dtype
                op_type_dict = {
                    '.*':{ # re.match
                        "weight": {
                            'bits': bits, # 1-8 bits
                            "dtype":dtype,
                            'group_size': config.group_size,  # -1 (per-channel)
                            'scheme': config.scheme, # sym/asym
                            'algorithm': config.algorithm, # RTN/AWQ/TEQ
                        },
                    },
                }
                recipes = {"enable_full_range": True if "fullrange" in config.weight_dtype else False,
                        "enable_mse_search": config.mse_range}
                conf = PostTrainingQuantConfig(
                    approach='weight_only',
                    op_type_dict=op_type_dict,
                    recipes=recipes,
                )
                optimized_model = quantization.fit(
                    model,
                    conf,
                ).model
        return optimized_model
