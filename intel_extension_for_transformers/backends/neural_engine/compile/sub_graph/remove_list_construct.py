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

"""The RemoveListConstruct Pattern."""

from .pattern import Pattern, pattern_registry
from collections import namedtuple, OrderedDict
from .. import graph_utils as util
import copy


@pattern_registry(pattern_type='RemoveListConstruct')
class RemoveListConstruct(Pattern):
    """The RemoveListConstruct pattern.

    RemoveListConstruct
    """
    def __call__(self, model):
        """The __call__ function of this pattern class."""
        remove_list = []
        flag = False 
        for node in model.nodes:
            if node.op_type == 'ListConstruct':
                for dest_op_name in node.output_tensors[0].dest_op:
                    dest_node = model.get_node_by_name(dest_op_name)
                    for i in range(len(dest_node.input_tensors)):
                        if dest_node.input_tensors[i].name == node.output_tensors[0].name:
                            del dest_node.input_tensors[i]
                            idx = i 
                            for tensor in node.input_tensors:
                                dest_node.input_tensors.insert(idx, copy.deepcopy(tensor))
                                idx += 1
                        # new_input_tensors.append(copy.deepcopy(tensor))
                    # dest_node.input_tensors = new_input_tensors
                remove_list.append(node.name)
            if node.op_type == 'TupleUnpack':
                if flag == False:
                    del model.get_node_by_name('input_data').output_tensors[1]
                    flag = True
                for output_tensor in node.output_tensors:
                    output_tensor.source_op = []
                    output_tensor.dtype = "fp32"
                    output_tensor.shape = [-1, -1, -1, -1]
                    input_node = model.get_node_by_name('input_data')
                    input_node.output_tensors.append(output_tensor)
                    for input_tensor in input_node.input_tensors:
                        if input_tensor.name == node.input_tensors[0].name:
                            input_node.input_tensors.remove(input_tensor)
                    for dest_op_name in output_tensor.dest_op:
                        dest_node = model.get_node_by_name(dest_op_name)
                        for i in range(len(dest_node.input_tensors)):
                            if dest_node.input_tensors[i].name == output_tensor.name:
                                dest_node.input_tensors[i].source_op = []
                remove_list.append(node.name)
                 
        model.remove_nodes(remove_list)
        return model
