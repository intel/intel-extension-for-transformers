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

from .pattern import Pattern, pattern_registry
from collections import namedtuple, OrderedDict
from .. import graph_utils as util
from ..ops import Tensor
from .subgraph_matcher import EXECUTOR_TYPE


@pattern_registry(pattern_type='QunatizeFusion')
class QunatizeFusion(Pattern):

    def __call__(self, model):

        def search_quant_fusion(node):
            if node.input_tensors[0].source_op == []:
                return (None, False)
            pre_node = model.get_node_by_name(node.input_tensors[0].source_op[0])
            if pre_node.name in quant_info or (pre_node.op_type == "Softmax") \
               or (not quant_info and pre_node.op_type in EXECUTOR_TYPE \
                   and (EXECUTOR_TYPE[pre_node.op_type] == "InnerProduct" or \
                        EXECUTOR_TYPE[pre_node.op_type] == "Matmul")):
                return (pre_node, True)
            elif pre_node.op_type == "Reshape":
                return search_quant_fusion(pre_node)
            else:
                return (None, False)

        quant_info = util.get_quant_info()

        remove_node_name = []
        # fuse quant nodes to previous innerproduct or matmul output dtype to enhance perf
        for node in model.nodes:
            if node.op_type == "Quantize":
                dtype = node.attr['output_dtype'] 
                quant_node, can_fuse = search_quant_fusion(node)
                if can_fuse:
                    if dtype == 'u8' or dtype == 's8':
                        if quant_node.op_type == "Softmax":
                            model.change_node_input_tensors(quant_node.name, 1, node.input_tensors[1],
                                                            'insert')
                            model.change_node_input_tensors(quant_node.name, 2, node.input_tensors[2],
                                                            'insert')
                            quant_node.attr['output_dtype'] = "u8"
                        else:
                            model.change_node_input_tensors(quant_node.name, -2, node.input_tensors[1],
                                                            'modify')
                            model.change_node_input_tensors(quant_node.name, -1, node.input_tensors[2],
                                                            'modify')
                            quant_node.attr['output_dtype'] = node.attr['output_dtype']
                    elif dtype == 'bf16':
                        quant_node.attr['output_dtype'] = dtype

                    for dst_node_name in node.output_tensors[0].dest_op:
                        dst_node = model.get_node_by_name(dst_node_name)
                        for idx, input_tensor in enumerate(dst_node.input_tensors):
                            if node.output_tensors[0].name == input_tensor.name:
                                model.change_node_input_tensors(dst_node_name, idx,
                                                                node.input_tensors[0], 'modify')

                    remove_node_name.append(node.name)


        model.remove_nodes(remove_node_name)

        return model
