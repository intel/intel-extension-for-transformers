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

from typing import Union
from intel_extension_for_transformers.neural_chat.config import (
    AMPConfig,
    WeightOnlyQuantizationConfig,
    BitsAndBytesConfig
)

class Optimization:
    def __init__(
            self,
            optimization_config: Union[AMPConfig, WeightOnlyQuantizationConfig, BitsAndBytesConfig]
        ):
        self.optimization_config = optimization_config

    def optimize(self, model):
        optimized_model = model
        config = self.optimization_config
        if isinstance(config, WeightOnlyQuantizationConfig):
            print("Applying Weight Only Quantization.")
            from neural_compressor import PostTrainingQuantConfig, quantization
            op_type_dict = {
                '.*':{ 	# re.match
                    "weight": {
                        'bits': config.bits, # 1-8 bits
                        'group_size': config.group_size,  # -1 (per-channel)
                        'scheme': config.scheme, # sym/asym
                        'algorithm': config.algorithm, # RTN/AWQ/TEQ
                    },
                },
            }
            recipes = {"rtn_args": {"enable_full_range": config.enable_full_range}}
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
