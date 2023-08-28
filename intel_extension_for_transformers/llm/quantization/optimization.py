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

from intel_extension_for_transformers.neural_chat.config import OptimizationConfig

class Optimization:
    def __init__(self, optimization_config: OptimizationConfig):
        self.optimization_config = optimization_config

    def optimize(self, model):
        optimized_model = model
        config = self.optimization_config
        if config.weight_only_quant_config:
            print("Applying Weight Only Quantization.")
            from neural_compressor import PostTrainingQuantConfig, quantization
            op_type_dict = {
                '.*':{ 	# re.match
                    "weight": {
                        'bits': config.weight_only_quant_config.bits, # 1-8 bits
                        'group_size': config.weight_only_quant_config.group_size,  # -1 (per-channel)
                        'scheme': config.weight_only_quant_config.scheme, # sym/asym
                        'algorithm': config.weight_only_quant_config.algorithm, # RTN/AWQ/TEQ
                    },
                },
            }
            recipes = {"rtn_args": {"sym_full_range": config.weight_only_quant_config.sym_full_range}}
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
