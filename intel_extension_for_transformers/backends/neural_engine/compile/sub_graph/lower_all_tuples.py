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

"""The LowerAllTuples Pattern."""

from .pattern import Pattern, pattern_registry
from collections import namedtuple, OrderedDict
from .. import graph_utils as util
import copy


@pattern_registry(pattern_type='LowerAllTuples')
class LowerAllTuples(Pattern):
    """The LowerAllTuples pattern.

    LowerAllTuples
    """
    def __call__(self, model):
        """The __call__ function of this pattern class."""
        if model.framework_modeling_config['framework'] != 'torch':
            return model
        remove_list = []
        for node in model.nodes:
            if node.op_type in ['ListConstruct', 'TupleConstruct']:
                if node.output_tensors[0].dest_op == [] and node.output_tensors[0].name in model.output_tensors_name:
                    idx = model.output_tensors_name.index(node.output_tensors[0].name)
                    del model.output_tensors_name[idx]
                    for tensor in node.input_tensors:
                        for tensor in node.input_tensors:
                            model.output_tensors_name.insert(idx, tensor.name)
                            idx += 1

                for dest_op_name in node.output_tensors[0].dest_op:
                    dest_node = model.get_node_by_name(dest_op_name)
                    for i in range(len(dest_node.input_tensors)):
                        if dest_node.input_tensors[i].name == node.output_tensors[0].name:
                            del dest_node.input_tensors[i]
                            idx = i 
                            for tensor in node.input_tensors:
                                if node.name in tensor.dest_op:
                                    tensor.dest_op.remove(node.name)
                                tensor.dest_op.append(dest_node.name)
                                dest_node.input_tensors.insert(idx, copy.deepcopy(tensor))
                                idx += 1
                remove_list.append(node.name)
        node_idx = len(model.nodes) - 1
        dtype = "fp32"
        if util.get_autocast_info()['cast_type'] == "bf16":
            dtype = "bf16"
        while node_idx >= 0:
            node = model.nodes[node_idx]
            if node.op_type in ['ListUnpack', 'TupleUnpack']:
                for source_op_name in node.input_tensors[0].source_op:
                    source_op = model.get_node_by_name(source_op_name)
                    for i in range(len(source_op.output_tensors)):
                        if source_op.output_tensors[i].name == node.input_tensors[0].name:
                            del source_op.output_tensors[i]
                            idx = i
                            for tensor in node.output_tensors:
                                if node.name in tensor.source_op:
                                    tensor.source_op.remove(node.name)
                                tensor.source_op.append(source_op.name)
                                if source_op.op_type == 'Input':
                                    tensor.source_op = []
                                    tensor.dtype = dtype
                                    tensor.shape = [-1, -1, -1, -1]
                                source_op.output_tensors.insert(idx, copy.deepcopy(tensor))
                                idx += 1
                remove_list.append(node.name)
            node_idx -= 1

        model.remove_nodes(remove_list)
        return model
