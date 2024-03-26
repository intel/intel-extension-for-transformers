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

"""The RemoveZeros pattern."""

from .pattern import Pattern, pattern_registry
from collections import namedtuple, OrderedDict
from .. import graph_utils as util
from ..ops import Tensor
import copy

@pattern_registry(pattern_type='RemoveZeros')
class RemoveZeros(Pattern):
    """The RemoveZeros pattern.

    Fuse the original sub-graph into the custom acceleration 'RemoveZeros' graph.
    The search strategy is based on the following pattern mapping configs for different models.
    """
    def __call__(self, model):
        """The __call__ function of this pattern class."""
        if model.framework_modeling_config['framework'] != 'torch':
            return model

        remove_list = []
        node_idx = 0
        while node_idx < len(model.nodes):
            node = model.nodes[node_idx]
            if node.op_type == 'Zeros':
                for dest_op_name in node.output_tensors[0].dest_op:
                    dest_node = model.get_node_by_name(dest_op_name)
                    if dest_node.op_type == 'BinaryAdd' or  dest_node.op_type =='Add':
                        remove_list.append(node.name)
                        remove_list.append(dest_node.name)
                        # find origin tensor
                        ori_tensor = None
                        for in_tensor in dest_node.input_tensors:
                            if in_tensor.name != node.output_tensors[0].name:
                                ori_tensor = in_tensor
                        for add_dest_op_name in dest_node.output_tensors[0].dest_op:
                            add_dest_node = model.get_node_by_name(add_dest_op_name)
                            for in_tensor_idx in range(len(add_dest_node.input_tensors)):
                                if add_dest_node.input_tensors[in_tensor_idx].name == dest_node.output_tensors[0].name:
                                    add_dest_node.input_tensors[in_tensor_idx] = copy.deepcopy(ori_tensor)
                                    if dest_node.name in ori_tensor.dest_op:
                                        ori_tensor.dest_op.remove(dest_node.name)
                                    ori_tensor.dest_op.append(add_dest_node.name)


            node_idx += 1
        model.remove_nodes(remove_list)
        return model
