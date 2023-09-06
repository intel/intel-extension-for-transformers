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


import logging
import torch
import transformers
from ...llm.quantization.config import WeightOnlyConfig
from ...llm.quantization.utils import convert_to_quantized_model, convert_dtype_2_str


logger = logging.getLogger(__name__)


class _BaseQBitsAutoModelClass:
    ORIG_MODEL = None

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        import intel_extension_for_transformers.transformers.modeling.modeling_map
        load_in_8bit = kwargs.pop("load_in_8bit", False)
        load_in_4bit = kwargs.pop("load_in_4bit", False)
        quantization_config = kwargs.pop("quantization_config", None)
        if load_in_8bit or load_in_4bit or quantization_config is not None:
            torch_dtype = kwargs.pop("torch_dtype", torch.float32)
        if load_in_4bit:
            if quantization_config is None:
                quantization_config = WeightOnlyConfig(compute_dtype=torch_dtype, weight_dtype="nf4")
            else:
                assert "4" in quantization_config.weight_dtype and quantization_config.compute_dtype == torch_dtype, \
                f"Quantization_config.weight_dtype should be 'nf4', 'int4_fullrange', 'int4_clip',"
                f"'fp4_e2m1' or 'fp4_e2m1_bnb' and compute_dtype should be {torch_dtype}."
        elif load_in_8bit:
            if quantization_config is None:
                quantization_config = WeightOnlyConfig(compute_dtype=torch_dtype, weight_dtype="int8")
            else:
                assert quantization_config.weight_dtype == "int8" and quantization_config.compute_dtype == torch_dtype, \
                f"Quantization_config.weight_dtype should be 'int8' and compute_dtype should be {torch_dtype}."
        elif quantization_config is not None:
            if quantization_config.compute_dtype != convert_dtype_2_str(torch_dtype):
                logger.warning(f"Quantization_config.compute_dtype should be align with {torch_dtype}.")

        model = cls.ORIG_MODEL.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        if quantization_config is not None:
            return convert_to_quantized_model(model, quantization_config)
        else:
            return model


class AutoModelForCausalLM(_BaseQBitsAutoModelClass):
    ORIG_MODEL = transformers.AutoModelForCausalLM


class AutoModel(_BaseQBitsAutoModelClass):
    ORIG_MODEL = transformers.AutoModel


class AutoModelForSeq2SeqLM(_BaseQBitsAutoModelClass):
    ORIG_MODEL = transformers.AutoModelForSeq2SeqLM

