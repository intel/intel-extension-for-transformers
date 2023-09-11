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
"""The TextEncoder_AttentionMaskAddReshape Pattern."""

from .pattern import Pattern, pattern_registry
from collections import OrderedDict
from .. import graph_utils as util
from .. import logger


@pattern_registry(pattern_type='TextEncoder_AttentionMaskAddReshape')
class TextEncoder_AttentionMaskAddReshape(Pattern):
    """The TextEncoder_AttentionMaskAddReshape pattern.

    Fuse the original sub-graph into the custom acceleration 'TextEncoder_AttentionMaskAddReshape' graph.
    The search strategy is based on the following pattern mapping configs for the stable diffusionV1-5.
    """
    def __call__(self, model):
        """The __call__ function of this pattern class."""
        pattern_mapping_config = {
            # for text encoder attention mask add (Reshape_6)
            'TextEncoder_AttentionMaskAddReshape': [
                {
                    'patterns': {
                        'in': [
                            [(0, 'Unsqueeze'), (3, 'Concat'), (4, 'Reshape'), (5, 'Add')],
                            [(), (1, 'Unsqueeze'), (3, 'Concat')],
                            [(), (2, 'Unsqueeze'), (3, 'Concat')],
                        ],
                        'out': [[(0, 'Reshape'), (1, 'Add')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 4,
                        1: 5
                    },
                    'input_tensors': {
                        0: [[{
                            4: [0]
                        }, {
                            'input_data': [0]
                        }], [[0, 1], 2]],
                        1: [[{
                            5: [1]
                        }], [[1], 2]],
                    },
                    'output_tensors': {
                        0: [[{
                            4: [0]
                        }], [[0], 1]],
                        1: [[{
                            5: [0]
                        }], [[0], 1]]
                    },
                    'returns': [3]
                },
            ]
        }

        def _set_attr(hidden_size, node_names, model, reshape_idx=0):
            attr = OrderedDict()
            # bsz, hidden_size, max_seq, max_seq
            attr['dst_shape'] = '-1,' + str(hidden_size) + ',-1,-1'
            attr['dims'] = '0,1,1'
            reshape_node_idx = model.get_node_id(node_names[reshape_idx])
            model.nodes[reshape_node_idx].attr = attr

        for i in range(len(pattern_mapping_config['TextEncoder_AttentionMaskAddReshape'])):
            pattern_dict = pattern_mapping_config['TextEncoder_AttentionMaskAddReshape'][i]
            model, new_node_names, ret_old_nodes = util.pattern_mapping("TextEncoder_AttentionMaskAddReshape",
                                                                        pattern_dict, model)

            if len(new_node_names) != 0:
                logger.info('TextEncoder_AttentionMaskAddReshape mathched...')
                logger.debug('TextEncoder_AttentionMaskAddReshape = {}'.format(new_node_names))
                for j in range(len(new_node_names)):
                    concat_node = ret_old_nodes[j][0]
                    hidden_size = int(concat_node.input_tensors[-3].data)
                    _set_attr(hidden_size, new_node_names[j], model)

        return model
