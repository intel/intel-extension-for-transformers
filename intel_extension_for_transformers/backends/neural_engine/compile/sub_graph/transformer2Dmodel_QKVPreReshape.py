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
"""The Transformer2Dmodel_QKVPreReshape Pattern."""

from .pattern import Pattern, pattern_registry
from collections import OrderedDict
from .. import graph_utils as util
from .. import logger


@pattern_registry(pattern_type='Transformer2Dmodel_QKVPreReshape')
class Transformer2Dmodel_QKVPreReshape(Pattern):
    """The Transformer2Dmodel_QKVPreReshape pattern.

    Fuse the original sub-graph into the custom acceleration 'Transformer2Dmodel_QKVPreReshape' graph.
    The search strategy is based on the following pattern mapping configs for the stable diffusionV1-5.
    """
    def __call__(self, model):
        """The __call__ function of this pattern class."""
        pattern_mapping_config = {
            'Transformer2Dmodel_QKVPreReshape': [
                # v2-1 only
                {
                    'patterns': {
                        'in': [[(0, 'Mul'), (1, 'Unsqueeze'), (6, 'Concat'), (7, 'Reshape')],
                               [(), (2, 'Unsqueeze'), (6, 'Concat')], [(), (3, 'Unsqueeze'), (6, 'Concat')],
                               [(), (4, 'GroupNorm'), (5, 'Transpose'), (7, 'Reshape')]],
                        'out': [[(0, 'GroupNorm'), (1, 'Transpose'), (2, 'Reshape')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 4,
                        1: 5,
                        2: 7,
                    },
                    'input_tensors': {
                        0: [[{
                            4: [0]
                        }, {
                            4: [1]
                        }, {
                            4: [2]
                        }], [[0, 1, 2], 3]],
                        1: [[], [[], 1]],
                        2: [[], [[], 1]],
                    },
                    'output_tensors': {
                        0: [[{
                            4: [0]
                        }], [[0], 1]],
                        1: [[{
                            5: [0]
                        }], [[0], 1]],
                        2: [[{
                            7: [0]
                        }], [[0], 1]]
                    },
                    'returns': [4, 5, 6, 7]
                },
                {
                    'patterns': {
                        'in': [[(0, 'Mul'), (1, 'Unsqueeze'), (6, 'Concat'), (7, 'Reshape')],
                               [(), (2, 'Unsqueeze'), (6, 'Concat')], [(), (3, 'Unsqueeze'), (6, 'Concat')],
                               [(), (4, 'Conv'), (5, 'Transpose'), (7, 'Reshape')]],
                        'out': [[(0, 'Conv'), (1, 'Transpose'), (2, 'Reshape')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 4,
                        1: 5,
                        2: 7,
                    },
                    'input_tensors': {
                        0: [[{
                            4: [0]
                        }, {
                            4: [1]
                        }, {
                            4: [2]
                        }], [[0, 1, 2], 3]],
                        1: [[], [[], 1]],
                        2: [[], [[], 1]],
                    },
                    'output_tensors': {
                        0: [[{
                            4: [0]
                        }], [[0], 1]],
                        1: [[{
                            5: [0]
                        }], [[0], 1]],
                        2: [[{
                            7: [0]
                        }], [[0], 1]]
                    },
                    'returns': [4, 5, 6, 7]
                },
            ]
        }

        for i in range(len(pattern_mapping_config['Transformer2Dmodel_QKVPreReshape'])):
            pattern_dict = pattern_mapping_config['Transformer2Dmodel_QKVPreReshape'][i]
            model, new_node_names, ret_old_nodes = util.pattern_mapping("Transformer2Dmodel_QKVPreReshape",
                                                                        pattern_dict, model)

            if len(new_node_names) != 0:
                if i == 0:
                    logger.info('Transformer2Dmodel_QKVPreReshape for V2-1 mathched...')
                    for node in model.nodes:
                        if node.op_type == 'Softmax':
                            attr = OrderedDict()
                            attr['version'] = 'V2'
                            node.attr = attr
                else:
                    logger.info('Transformer2Dmodel_QKVPreReshape mathched...')
                logger.debug('Transformer2Dmodel_QKVPreReshape = {}'.format(new_node_names))
                for j in range(len(new_node_names)):
                    # the first new node
                    assert ret_old_nodes[j][0].op_type == 'Conv' or ret_old_nodes[j][0].op_type == 'GroupNorm'
                    conv_node_idx = model.get_node_id(new_node_names[j][0])
                    model.nodes[conv_node_idx].attr = ret_old_nodes[j][0].attr

                    # the second new node
                    assert ret_old_nodes[j][1].op_type == 'Transpose'
                    transpose_node_idx = model.get_node_id(new_node_names[j][1])
                    model.nodes[transpose_node_idx].attr = ret_old_nodes[j][1].attr

                    # the third new node
                    weight = ret_old_nodes[j][0].input_tensors[1].shape[0]
                    attr = OrderedDict()
                    attr['dst_shape'] = '-1,' + str(weight)
                    reshape_node_idx = model.get_node_id(new_node_names[j][2])
                    model.nodes[reshape_node_idx].attr = attr

        return model
