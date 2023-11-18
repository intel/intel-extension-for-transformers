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

    def optimize(self, model, use_llm_runtime=False):
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
        try:
            if re.search("flan-t5", model.config._name_or_path, re.IGNORECASE):
                from intel_extension_for_transformers.transformers import AutoModelForSeq2SeqLM
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
                from intel_extension_for_transformers.transformers import AutoModelForCausalLM
                optimized_model = AutoModelForCausalLM.from_pretrained(
                    model.config._name_or_path,
                    quantization_config=config,
                    use_llm_runtime=use_llm_runtime,
                    trust_remote_code=True)
            elif re.search("starcoder", model.config._name_or_path, re.IGNORECASE):
                from intel_extension_for_transformers.transformers import GPTBigCodeForCausalLM
                optimized_model = GPTBigCodeForCausalLM.from_pretrained(
                    model.config._name_or_path,
                    quantization_config=config,
                    use_llm_runtime=use_llm_runtime,
                    trust_remote_code=True)
            elif re.search("chatglm", model.config._name_or_path, re.IGNORECASE):
                from intel_extension_for_transformers.transformers import AutoModel
                optimized_model = AutoModel.from_pretrained(
                    model.config._name_or_path,
                    quantization_config=config,
                    use_llm_runtime=use_llm_runtime,
                    trust_remote_code=True)
            return optimized_model
        except Exception as e:
            from intel_extension_for_transformers.neural_chat.constants import ResponseCodes
            from intel_extension_for_transformers.utils import logger
            if type(self.optimization_config) == MixedPrecisionConfig:
                logger.error(f"Optimize model {model.config._name_or_path} with mixed precision failed, {e}")
                return ResponseCodes.ERROR_AMP_OPTIMIZATION_FAIL
            elif type(self.optimization_config) == WeightOnlyQuantConfig:
                logger.error(f"Optimize model {model.config._name_or_path} with weight only quantization failed, {e}")
                return ResponseCodes.ERROR_WEIGHT_ONLY_QUANT_OPTIMIZATION_FAIL
            elif type(self.optimization_config) == BitsAndBytesConfig:
                logger.error(f"Optimize model {model.config._name_or_path} with bits and bytes failed, {e}")
                return ResponseCodes.ERROR_BITS_AND_BYTES_OPTIMIZATION_FAIL
