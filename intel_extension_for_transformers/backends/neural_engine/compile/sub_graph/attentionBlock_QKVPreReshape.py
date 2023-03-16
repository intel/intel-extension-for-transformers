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

"""The AttentionBlock_QKVPreReshape Pattern."""

from .pattern import Pattern, pattern_registry
from collections import OrderedDict
from .. import graph_utils as util
from .. import logger


@pattern_registry(pattern_type='AttentionBlock_QKVPreReshape')
class AttentionBlock_QKVPreReshape(Pattern):
    """The AttentionBlock_QKVPreReshape pattern.

    Fuse the original sub-graph into the custom acceleration 'AttentionBlock_QKVPreReshape' graph.
    The search strategy is based on the following pattern mapping configs for the stable diffusionV1-5.
    """
    def __call__(self, model):
        """The __call__ function of this pattern class."""
        pattern_mapping_config = {
            'AttentionBlock_QKVPreReshape': [
                {
                    'patterns': {
                        'in': [
                            [(0, 'Unsqueeze'), (4, 'Concat'), (5, 'Reshape'), (6, 'Transpose')],
                            [(), (1, 'Unsqueeze'), (4, 'Concat')],
                            [(), (2, 'Mul'), (3, 'Unsqueeze'), (4, 'Concat')],
                        ],
                        'out': [[(0, 'Reshape'), (1, 'Transpose'), (2, 'Reshape')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 5,
                        1: 6,
                        2: 'AttentionBlock_QKVPreReshape/TransposeTo2D'
                    },
                    'input_tensors': {
                        0: [[{
                            5: [0]
                        }, {
                            'input_data': [0]
                        }], [[0, 1], 2]],
                        1: [[], [[], 1]],
                        2: [[], [[], 1]],
                    },
                    'output_tensors': {
                        0: [[{
                            5: [0]
                        }], [[0], 1]],
                        1: [[], [[], 1]],
                        2: [[{
                            6: [0]
                        }], [[0], 1]],
                    },
                    'returns': [6]
                },
            ]
        }

        for i in range(len(pattern_mapping_config['AttentionBlock_QKVPreReshape'])):
            pattern_dict = pattern_mapping_config['AttentionBlock_QKVPreReshape'][i]
            model, new_node_names, ret_old_nodes = util.pattern_mapping("AttentionBlock_QKVPreReshape",
                                                                        pattern_dict, model)

            logger.info('AttentionBlock_QKVPreReshape mathched...')
            logger.debug('AttentionBlock_QKVPreReshape = {}'.format(new_node_names))
            if len(new_node_names) != 0:
                for j in range(len(new_node_names)):
                    # the first new node
                    attr = OrderedDict()
                    attr['dst_shape'] = '-1,512,-1'
                    attr['dims'] = 0
                    reshape_node_idx = model.get_node_id(new_node_names[j][0])
                    model.nodes[reshape_node_idx].attr = attr

                    # the second new node
                    assert ret_old_nodes[j][0].op_type == 'Transpose'
                    transpose_node_idx = model.get_node_id(new_node_names[j][1])
                    model.nodes[transpose_node_idx].attr = ret_old_nodes[j][0].attr

                    # the third new node
                    attr = OrderedDict()
                    attr['dst_shape'] = '-1,512'
                    reshape_node_idx = model.get_node_id(new_node_names[j][2])
                    model.nodes[reshape_node_idx].attr = attr

        return model
