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

"""The RemoveLastView Pattern."""

from .pattern import Pattern, pattern_registry
from collections import namedtuple, OrderedDict
from .. import graph_utils as util
from .subgraph_matcher import EXECUTOR_TYPE
import numpy as np
import copy

@pattern_registry(pattern_type='RemoveLastView')
class RemoveLastView(Pattern):
    """The RemoveLastView pattern.

    Fuse the original sub-graph into the custom acceleration 'RemoveLastView' graph.
    The search strategy is based on the following pattern mapping configs for different models.
    """

    def __call__(self, model):
        """The __call__ function of this pattern class."""
        if model.framework_modeling_config['framework'] != 'torch':
            return model
        remove_list = []
        for node in model.nodes:
            if node.op_type == 'View':
                pre_node = model.get_node_by_name(node.input_tensors[0].source_op[0])
                dst_node = model.get_node_by_name(node.output_tensors[0].dest_op[0])
                node.output_tensors[0].source_op = [pre_node.name]
                pre_node.output_tensors[0] = node.output_tensors[0]
                dst_node.input_tensors[0].source_op = [pre_node.name]
                for in_tensor in node.input_tensors[1:]:
                    source_node = model.get_node_by_name(in_tensor.source_op[0])
                    if len(source_node.output_tensors) == 1 and len(source_node.output_tensors[0].dest_op) == 1 \
                        and source_node.output_tensors[0].dest_op[0] == node.name:
                        remove_list.append((source_node.name))
                remove_list.append(node.name)
        model.remove_nodes(remove_list)
        return model
