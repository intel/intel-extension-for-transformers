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

class Optimization:
    def __init__(
            self,
            optimization_config
        ):
        self.optimization_config = optimization_config

    def optimize(self, model, use_neural_speed=False):
        if isinstance(model, str):
            model_name = model
        else:
            model_name = model.config._name_or_path
            optimized_model = model
        from intel_extension_for_transformers.transformers import (
            MixedPrecisionConfig,
            WeightOnlyQuantConfig,
            BitsAndBytesConfig
        )
        assert type(self.optimization_config) in [MixedPrecisionConfig, WeightOnlyQuantConfig, BitsAndBytesConfig], \
            f"Expect optimization_config be an object of MixedPrecisionConfig, WeightOnlyQuantConfig" + \
            " or BitsAndBytesConfig,got {type(self.optimization_config)}."
        config = self.optimization_config
        if re.search("flan-t5", model_name, re.IGNORECASE):
            from intel_extension_for_transformers.transformers import AutoModelForSeq2SeqLM
            optimized_model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_name,
                    quantization_config=config,
                    use_neural_speed=use_neural_speed,
                    trust_remote_code=True)
        elif (
            re.search("gpt", model_name, re.IGNORECASE)
            or re.search("mpt", model_name, re.IGNORECASE)
            or re.search("bloom", model_name, re.IGNORECASE)
            or re.search("llama", model_name, re.IGNORECASE)
            or re.search("opt", model_name, re.IGNORECASE)
            or re.search("neural-chat", model_name, re.IGNORECASE)
            or re.search("starcoder", model_name, re.IGNORECASE)
            or re.search("codegen", model_name, re.IGNORECASE)
            or re.search("mistral", model_name, re.IGNORECASE)
            or re.search("magicoder", model_name, re.IGNORECASE)
            or re.search("solar", model_name, re.IGNORECASE)
        ):
            from intel_extension_for_transformers.transformers import AutoModelForCausalLM
            optimized_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=config,
                use_neural_speed=use_neural_speed,
                trust_remote_code=True)
        elif re.search("chatglm", model_name, re.IGNORECASE):
            from intel_extension_for_transformers.transformers import AutoModel
            optimized_model = AutoModel.from_pretrained(
                model_name,
                quantization_config=config,
                use_neural_speed=use_neural_speed,
                trust_remote_code=True)
        return optimized_model
