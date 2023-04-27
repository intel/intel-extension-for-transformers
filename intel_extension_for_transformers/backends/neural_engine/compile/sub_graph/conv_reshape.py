#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2022 Intel Corporation
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

"""The ConvReshape Pattern."""

from .pattern import Pattern, pattern_registry
from collections import namedtuple, OrderedDict
from .. import graph_utils as util
from .. import logger


@pattern_registry(pattern_type='ConvReshape')
class ConvReshape(Pattern):
    """The ConvReshape pattern.

    Fuse the original sub-graph into the custom acceleration 'ConvReshape' graph.
    The search strategy is based on the following pattern mapping configs for different models.
    """
    def __call__(self, model):
        """The __call__ function of this pattern class."""
        pattern_mapping_config = {
            'ConvReshape': [
                {
                    'patterns': {
                        'in': [[(0, 'Conv'), (1, 'Shape'), (2, 'Slice'), (3, 'Concat'), (4, 'Reshape')]],
                        'out': [[(0, 'Conv'), (1, 'Reshape')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 0,
                        1: 4
                    },
                    'input_tensors': {
                        # Conv has bias, so there are 3 inputs
                        0: [[{
                            0: [0]
                        }, {
                            0: [1]
                        }, {
                            0: [2]
                        }], [[0, 1, 2], 3]],
                        1: [[{
                            'input_data': [0]
                        }], [[1], 2]]
                    },
                    'output_tensors': {
                        0: [[{
                            0: [0]
                        }], [[0], 1]],
                        1: [[{
                            4: [0]
                        }], [[0], 1]]
                    },
                    'returns': [0]
                },
                {
                    'patterns': {
                        'in': [[(0, 'Transpose'), (1, 'Conv')]],
                        'out': [[(0, 'Conv')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 1
                    },
                    'input_tensors': {
                        0: [[{
                            0: [0]
                        }, {
                            1: [1]
                        }, {
                            1: [2]
                        }], [[0, 1, 2], 3]]
                    },
                    'output_tensors': {
                        0: [[{
                            1: [0]
                        }], [[0], 1]]
                    },
                    'returns': [0, 1]
                },
            ]
        }

        def _set_attr(channel, node_names, model):
            attr = OrderedDict()
            # Dynamic batch
            # We need to reshape Conv output to (batch, channel, -1)
            attr['dst_shape'] = '-1,' + str(channel) + ',-1'
            attr['dims'] = 0
            reshape_node_idx = model.get_node_id(node_names[1])
            model.nodes[reshape_node_idx].attr = attr

        pattern_dict = pattern_mapping_config['ConvReshape'][0]
        model, new_node_names, ret_old_nodes = util.pattern_mapping('ConvReshape',
                                                                    pattern_dict, model)
        if len(new_node_names) != 0:
            for j in range(len(new_node_names)):
                conv_node = ret_old_nodes[j][0]
                conv_node_idx = model.get_node_id(new_node_names[j][0])
                model.nodes[conv_node_idx].attr = conv_node.attr
                # Use Conv weight to get channel
                channel = conv_node.input_tensors[1].shape[0]
                _set_attr(channel, new_node_names[j], model)

        pattern_dict = pattern_mapping_config['ConvReshape'][1]
        model, new_node_names, ret_old_nodes = util.pattern_mapping("ConvReshape",
                                                                    pattern_dict, model)
        if len(new_node_names) != 0:
            logger.info('ConvReshape_1 mathched...')
            logger.debug('ConvReshape = {}'.format(new_node_names))
            for i in range(len(new_node_names)):
                # the first node
                conv_node_idx = model.get_node_id(new_node_names[i][0])
                model.nodes[conv_node_idx].attr = ret_old_nodes[i][1].attr
                model.nodes[conv_node_idx].attr['src_perm'] = ret_old_nodes[i][0].attr['dst_perm']
    
        return model
