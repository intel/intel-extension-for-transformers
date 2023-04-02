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
"""The TextEncoder_AttentionReshape Pattern."""

from .pattern import Pattern, pattern_registry
from collections import OrderedDict
from .. import graph_utils as util
from .. import logger


@pattern_registry(pattern_type='TextEncoder_AttentionReshape')
class TextEncoder_AttentionReshape(Pattern):
    """The TextEncoder_AttentionReshape pattern.

    Fuse the original sub-graph into the custom acceleration 'TextEncoder_AttentionReshape' graph.
    The search strategy is based on the following pattern mapping configs for the stable diffusionV1-5.
    """
    def __call__(self, model):
        """The __call__ function of this pattern class."""
        pattern_mapping_config = {
            # for text encoder self_attn/out_proj/Add
            'TextEncoder_AttentionReshape': [
                {
                    'patterns': {
                        'in': [[(0, 'Shape'), (1, 'Gather'), (2, 'Unsqueeze'), (9, 'Concat'), (10, 'Reshape'),
                                (11, 'MatMulWithBias')],
                               [(), (3, 'Shape'), (4, 'Gather'), (5, 'Unsqueeze'), (9, 'Concat')],
                               [(), (6, 'Shape'), (7, 'Gather'), (8, 'Unsqueeze'), (9, 'Concat')]],
                        'out': [[(0, 'Reshape'), (1, 'Reshape'), (1, 'MatMulWithBias')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 10,
                        1: 'TextEncoder_AttentionReshape/reshape_to_2D',
                        2: 11
                    },
                    'input_tensors': {
                        0: [[{
                            10: [0]
                        }, {
                            'input_data': [0]
                        }], [[0, 1], 2]],
                        1: [[{
                            'input_data': [0]
                        }], [[1], 2]],
                        2: [[{
                            11: [1]
                        }, {
                            11: [2]
                        }], [[1, 2], 3]],
                    },
                    'output_tensors': {
                        0: [[{
                            10: [0]
                        }], [[0], 1]],
                        1: [[], [[], 1]],
                        2: [[{
                            11: [0]
                        }], [[0], 1]]
                    },
                    'returns': [9, 11]
                },
            ]
        }

        def _set_attr(node_names, model):
            attr = OrderedDict()
            # bsz, max_seq, hidden_size, 1, 77, 768
            attr['dst_shape'] = '-1,-1,-1'
            attr['dims'] = '0, 1'
            reshape_node_idx = model.get_node_id(node_names[0])
            model.nodes[reshape_node_idx].attr = attr

            attr_1 = OrderedDict()
            # bsz, max_seq, hidden_size, 1, 77, 768
            attr_1['dst_shape'] = '-1,-1'
            attr_1['dims'] = 1
            reshape_node_idx = model.get_node_id(node_names[1])
            model.nodes[reshape_node_idx].attr = attr_1

        for i in range(len(pattern_mapping_config['TextEncoder_AttentionReshape'])):
            pattern_dict = pattern_mapping_config['TextEncoder_AttentionReshape'][i]
            model, new_node_names, ret_old_nodes = util.pattern_mapping("TextEncoder_AttentionReshape",
                                                                        pattern_dict, model)

            if len(new_node_names) != 0:
                logger.info('TextEncoder_AttentionReshape mathched...')
                logger.debug('TextEncoder_AttentionReshape = {}'.format(new_node_names))
                for j in range(len(new_node_names)):
                    if len(ret_old_nodes[j]) == 2:
                        _set_attr(new_node_names[j], model)

                        assert ret_old_nodes[j][1].op_type == 'MatMulWithBias'
                        mat_node_idx = model.get_node_id(new_node_names[j][2])
                        model.nodes[mat_node_idx].attr = ret_old_nodes[j][1].attr

        return model
