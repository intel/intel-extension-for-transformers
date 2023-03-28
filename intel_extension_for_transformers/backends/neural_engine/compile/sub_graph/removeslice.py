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

"""The RemoveSlice Pattern."""

from .pattern import Pattern, pattern_registry
from collections import namedtuple, OrderedDict
from .. import graph_utils as util
import copy


@pattern_registry(pattern_type='RemoveSlice')
class RemoveSlice(Pattern):
    """The RemoveSlice pattern.

    RemoveSlice
    """
    def __call__(self, model):
        """The __call__ function of this pattern class."""
        if model.framework_modeling_config['framework'] != 'torch':
            return model

        remove_list = []
        first_one = False 
        output_tensor = None
        for node in model.nodes:
            if node.op_type == 'SliceMask':
                if first_one == False:
                    output_tensor = copy.deepcopy(node.output_tensors[0])
                    first_one = True
                    continue
                next_node = model.get_node_by_name(node.output_tensors[0].dest_op[0])
                for i in range(len(next_node.input_tensors)):
                    if next_node.input_tensors[i].name ==  node.output_tensors[0].name:
                        next_node.input_tensors[i] = output_tensor
                        output_tensor.dest_op.append(model.get_node_by_name(node.output_tensors[0].dest_op[0]).name)
                      

                remove_list.append(node.name)
                 
        model.remove_nodes(remove_list)
        return model
