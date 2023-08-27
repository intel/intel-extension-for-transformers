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

"""The MatMulWithTransposeScaleAdd pattern."""

from .pattern import Pattern, pattern_registry
from collections import namedtuple, OrderedDict
from .. import graph_utils as util
import copy


@pattern_registry(pattern_type='MatMulWithTransposeScaleAdd')
class MatMulWithTranspose(Pattern):
    """The MatMulWithTransposeScaleAdd pattern.

    Fuse the original sub-graph into the custom acceleration 'MatMulWithTransposeScaleAdd' graph.
    The search strategy is based on the following pattern mapping configs for different models.
    Call the 'MatMulWithTranspose' pattern first.
    """
    def __call__(self, model):
        """The __call__ function of this pattern class."""
        pattern_mapping_config = {
            'MatMulWithTransposeScaleAdd': [
            {
                'patterns': {
                    'in': [[(0, 'MatmulwithTranspose'), (1, ['Div', 'Mul']), (2, 'Add')]],
                    'out': [[(0, 'TransposeBatchMatMul')]]
                },
                'search_mode': 'op_type',
                'node_names': {
                    0: 2
                },
                'input_tensors': {
                    0: [[{
                        0: [0]
                    }, {
                        0: [1]
                    }, {
                        2: [0, 1]
                    }], [[0, 1, 2], 3]]
                },
                'output_tensors': {
                    0: [[{
                        2: [0]
                    }], [[0], 1]]
                },
                'returns': [0, 1, 2]
            },
            ]
        }

        if model.framework_modeling_config['framework'] != 'torch':
            return model
        
        def _set_attr(node_names, old_nodes, scale, binary_add=True):
            mat_node_idx = model.get_node_id(node_names[0])
            attr = copy.deepcopy(old_nodes[0].attr)
            if scale:
                attr['output_scale'] = scale
            if binary_add:
                attr['append_op'] = 'binary_add'
            model.nodes[mat_node_idx].attr = attr

        for pattern_dict in pattern_mapping_config['MatMulWithTransposeScaleAdd']:
            model, new_node_names, ret_old_nodes = util.pattern_mapping(
                                        "MatMulWithTransposeScaleAdd", pattern_dict, model)
            if len(new_node_names) != 0:
                for i in range(len(new_node_names)):
                    scale_node = ret_old_nodes[i][1]
                    scale = None
                    if scale_node.op_type == 'Div':
                        scale = 1.0 / float(scale_node.input_tensors[-1].data.item())
                    if scale_node.op_type == 'Mul':
                        scale = float(scale_node.input_tensors[-1].data.item())
                    _set_attr(new_node_names[i], ret_old_nodes[i], scale)
                return model

        return model
