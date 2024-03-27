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

"""The RemoveConstantOP Pattern."""

from .pattern import Pattern, pattern_registry
from collections import namedtuple, OrderedDict
from .. import graph_utils as util
import copy


@pattern_registry(pattern_type='RemoveConstantOP')
class RemoveConstantOP(Pattern):
    """The RemoveConstantOP pattern.

    Remove Constant OP
    """
    def __call__(self, model):
        """The __call__ function of this pattern class."""
        remove_list = []
        for node in model.nodes:
            if node.op_type == 'Constant':
                if node.output_tensors[0].name in model.input_tensors_name:
                    remove_list.append(node.name)
                    continue
                for dest_op_name in node.output_tensors[0].dest_op:
                    dest_node = model.get_node_by_name(dest_op_name)
                    new_input_tensors = []
                    for tensor in dest_node.input_tensors:
                        if tensor.name == node.output_tensors[0].name:
                            continue
                        new_input_tensors.append(copy.deepcopy(tensor))
                    dest_node.input_tensors = new_input_tensors
                remove_list.append(node.name)
        model.remove_nodes(remove_list)
        return model
