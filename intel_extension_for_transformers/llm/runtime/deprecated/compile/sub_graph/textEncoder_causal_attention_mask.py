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
"""The TextEncoder_CasualAttentionMask Pattern."""

from .pattern import Pattern, pattern_registry
from collections import OrderedDict
from .. import graph_utils as util
from .. import logger


@pattern_registry(pattern_type='TextEncoder_CasualAttentionMask')
class TextEncoder_CasualAttentionMask(Pattern):
    """The TextEncoder_CasualAttentionMask pattern.

    Fuse the original sub-graph into the custom acceleration 'TextEncoder_CasualAttentionMask' graph.
    The search strategy is based on the following pattern mapping configs for the stable diffusionV1-5.
    """
    def __call__(self, model):
        """The __call__ function of this pattern class."""
        pattern_mapping_config = {
            # for text encoder self_attn/out_proj/Add
            'TextEncoder_CasualAttentionMask': [
                {
                    'patterns': {
                        'in': [[(0, 'Shape'), (1, 'Gather'), (2, 'Unsqueeze'), (7, 'Concat'),
                                (8, 'ConstantOfShape'), (9, 'Shape'), (10, 'ConstantOfShape'), (11, 'Trilu'),
                                (12, 'Unsqueeze'), (13, 'Cast')],
                               [(), (3, 'Shape'), (4, 'Gather'), (5, 'Unsqueeze'), (7, 'Concat')],
                               [(), (6, 'Unsqueeze'), (7, 'Concat')]],
                        'out': [[(0, 'Reshape'), (1, 'Reshape'), (2, 'ConstantOfShape'), (3, 'Unsqueeze')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 'diffusion_casualAttentionMask/input/reshape',
                        1: 'diffusion_casualAttentionMask/input/reshape3D',
                        2: 10,
                        3: 12
                    },
                    'input_tensors': {
                        0: [[{
                            'input_data': [0]
                        }], [[0], 1]],
                        1: [[{
                            'input_data': [0]
                        }], [[1], 2]],
                        2: [[], [[], 1]],
                        3: [[], [[], 1]],
                    },
                    'output_tensors': {
                        0: [[], [[], 1]],
                        1: [[{
                            0: [0]
                        }], [[0], 1]],
                        2: [[{
                            10: [0]
                        }], [[0], 1]],
                        3: [[{
                            13: [0]
                        }], [[0], 1]],
                    },
                    'returns': [9, 12]
                },
            ]
        }

        for i in range(len(pattern_mapping_config['TextEncoder_CasualAttentionMask'])):
            pattern_dict = pattern_mapping_config['TextEncoder_CasualAttentionMask'][i]
            model, new_node_names, ret_old_nodes = util.pattern_mapping("TextEncoder_CasualAttentionMask",
                                                                        pattern_dict, model)

            if len(new_node_names) != 0:
                logger.info('TextEncoder_CasualAttentionMask mathched...')
                logger.debug('TextEncoder_CasualAttentionMask = {}'.format(new_node_names))
                for j in range(len(new_node_names)):
                    # the first new node
                    attr = OrderedDict()
                    attr['dst_shape'] = '1,-1'
                    reshape_node_idx = model.get_node_id(new_node_names[j][0])
                    model.nodes[reshape_node_idx].attr = attr

                    # the second new node
                    attr_2 = OrderedDict()
                    attr_2['dst_shape'] = '-1,-1,-1'
                    attr_2['dims'] = '0, 1, 1'
                    second_reshape_node_idx = model.get_node_id(new_node_names[j][1])
                    model.nodes[second_reshape_node_idx].attr = attr_2

                    # the thrid new node
                    attr_3 = OrderedDict()
                    attr_3['trilu'] = 1
                    attr_3['value'] = -10000
                    attr_3['tensor'] = 1
                    constantOfShape_node_idx = model.get_node_id(new_node_names[j][2])
                    constantOfShape_node = model.nodes[constantOfShape_node_idx]
                    constantOfShape_node.attr = attr_3

                    # the forth new node
                    second_ret_old_node = ret_old_nodes[j][1]
                    assert second_ret_old_node.op_type == 'Unsqueeze'
                    unsqueeze_node_idx = model.get_node_id(new_node_names[j][3])
                    model.nodes[unsqueeze_node_idx].attr = second_ret_old_node.attr

                    # text encoder output modify
                    last_hidden_state_idx = len(model.nodes)
                    node_name_list = []
                    for node in model.nodes:
                        idx = model.get_node_id(node.name)
                        if 'final_layer_norm' in node.name:
                            last_hidden_state_idx = idx

                        if idx > last_hidden_state_idx:
                            node_name_list.append(node.name)

                    if node_name_list != []:
                        model.remove_nodes(node_name_list)

        return model
