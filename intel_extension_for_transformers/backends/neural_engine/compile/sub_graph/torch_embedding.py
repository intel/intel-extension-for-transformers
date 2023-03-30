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

"""The TorchEmbedding pattern."""

from .pattern import Pattern, pattern_registry
from collections import namedtuple, OrderedDict
from .. import graph_utils as util


@pattern_registry(pattern_type='TorchEmbedding')
class TorchEmbedding(Pattern):
    """The TorchEmbedding pattern.

    Fuse the original sub-graph into the custom acceleration 'TorchEmbedding' graph.
    The search strategy is based on the following pattern mapping configs for different models.
    """
    def __call__(self, model):
        """The __call__ function of this pattern class."""
        if model.framework_modeling_config['framework'] != 'torch':
            return model

        node_idx = []
        for node in model.nodes:
            if node.op_type == 'Gather':
                if len(node.input_tensors) >=2 and len(node.input_tensors[1].source_op) == 1:
                    view_node = model.get_node_by_name(node.input_tensors[1].source_op[0])
                else:
                    continue
                if len(view_node.input_tensors) >=2 and len(view_node.input_tensors[1].source_op) == 1:
                    shape_node = model.get_node_by_name(view_node.input_tensors[1].source_op[0])
                else:
                    continue
                if shape_node.op_type != 'Shape' or view_node.op_type != 'View':
                    return model
                shape_node.op_type = 'Reshape'
                shape_node.attr.clear()
                shape_node.attr['dst_shape'] = '-1'

                node.input_tensors[1] = node.input_tensors[0]
                shape_node.output_tensors[0].dest_op = [node.name]
                node.input_tensors[0] = shape_node.output_tensors[0]
                node.attr['axis'] = 0
                node.attr['batch_dims'] = 0
                out_tensor = node.output_tensors[0]
                node.output_tensors[0] = view_node.output_tensors[0]
                node.output_tensors[0].source_op = [node.name]

                view_node.op_type = 'Reshape'
                node.output_tensors[0].dest_op = [view_node.name]
                view_node.input_tensors[0] = node.output_tensors[0]
                out_tensor.source_op = [view_node.name]
                view_node.output_tensors[0] = out_tensor
                view_node.input_tensors[1] = shape_node.input_tensors[0]
                view_node.attr.clear()
                view_node.attr['dst_shape'] = '-1,-1,-1'
                view_node.attr['dims'] = '0,1'
                view_node.attr['mul'] = '0,1'

                node_idx.append((model.get_node_id(view_node.name), model.get_node_id(node.name)))
        for idx1, idx2 in node_idx:
            tmp_node = model.nodes[idx1]
            model.nodes[idx1] = model.nodes[idx2]
            model.nodes[idx2] = tmp_node
        return model
