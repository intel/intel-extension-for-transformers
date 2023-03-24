#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Intel Corporation
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
"""The StableDiffusion_bf16Convert Pattern."""

from .pattern import Pattern, pattern_registry
from collections import OrderedDict
from .. import graph_utils as util
from ..ops import Tensor
from .subgraph_matcher import EXECUTOR_TYPE
import numpy as np


@pattern_registry(pattern_type='StableDiffusion_bf16Convert')
class StableDiffusion_bf16Convert(Pattern):
    """The StableDiffusion_bf16Convert pattern.

    Fuse the original sub-graph into the custom acceleration 'StableDiffusion_bf16Convert' graph.
    The search strategy is based on the following pattern mapping configs for different models.
    """
    def __call__(self, model):
        """The __call__ function of this pattern class."""
        def fp32_to_bf16(fp32_np):
            assert (fp32_np.dtype == np.float32)
            int32_np = fp32_np.view(dtype=np.int32)
            int32_np = int32_np >> 16
            bf16_np = int32_np.astype(np.uint16)
            return bf16_np

        for node in model.nodes:
            if node.op_type in EXECUTOR_TYPE:
                if (EXECUTOR_TYPE[node.op_type] == "InnerProduct" and node.input_tensors[1].dtype == "fp16") or \
                    (EXECUTOR_TYPE[node.op_type] == "Convolution" and node.input_tensors[1].dtype == "fp16"):
                    node.input_tensors[1].dtype = 'bf16'
                    if len(node.input_tensors) > 2:
                        bias_fp32 = node.input_tensors[2].data
                        if node.input_tensors[2].dtype == 'fp32':
                            node.input_tensors[2].data = fp32_to_bf16(bias_fp32)
                        node.input_tensors[2].dtype = 'bf16'
                    input_tensor = node.input_tensors[0]
                    input_name = input_tensor.name
                    quant_output = Tensor(name=input_name + "_quant",
                                          source_op=[node.name + "_quant"],
                                          dest_op=[node.name],
                                          dtype='bf16')
                    quantize_op = util.construct_node(node_name=node.name + "_quant",
                                                      op_type='Quantize',
                                                      input_tensors=[input_tensor],
                                                      output_tensors=[quant_output],
                                                      attr=OrderedDict({'output_dtype': 'bf16'}))
                    insert_idx = model.get_node_id(node.name)

                    model.insert_nodes(insert_idx, [quantize_op])
                    node.input_tensors[0] = quant_output
                    dest_op = node.output_tensors[0].dest_op
                    if dest_op and model.get_node_by_name(dest_op[0]).op_type != 'LayerNorm':
                        next_node = model.get_node_by_name(dest_op[0])
                        next_dest_op = next_node.output_tensors[0].dest_op
                        # when next_dest_op is Ouput op, output_type should be fp32
                        if not next_dest_op:
                            continue

        remove_duplicate_set = set()
        duplicate_list = []
        for node in model.nodes:
            sz = len(remove_duplicate_set)
            remove_duplicate_set.add(node.output_tensors[0].name)
            new_sz = len(remove_duplicate_set)
            if new_sz == sz:
                duplicate_list.append(node.name)
        model.remove_nodes(duplicate_list)

        return model
