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

"""The DecoderAttnReshape Pattern."""

from .pattern import Pattern, pattern_registry
from collections import namedtuple, OrderedDict
import copy
from .. import graph_utils as util


@pattern_registry(pattern_type='DecoderAttnReshape')
class DecoderAttnReshape(Pattern):
    """The DecoderAttnReshape pattern.

    Fuse the original sub-graph into the custom acceleration 'DecoderAttnReshape' graph.
    The search strategy is based on the following pattern mapping configs for different models.
    """
    def __call__(self, model):
        """The __call__ function of this pattern class."""
        pattern_mapping_config = {
            'DecoderAttnReshape': [
                {
                    'patterns': {
                        'in': [[(1, 'Shape'), (2, 'Gather'), (3, 'Unsqueeze'), (7, 'Concat'),
                                (8,'Reshape'), (9, 'Gather'), (10, 'Transpose')],
                               [(), (4, 'Shape'), (5, 'Gather'), (6, 'Unsqueeze'), (7, 'Concat')],
                               [(), (0, 'Unsqueeze'), (7, 'Concat')]],
                        'out': [[(0, 'Reshape'), (1, 'Gather'), (2, 'Transpose')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 8,
                        1: 9,
                        2: 10
                    },
                    'input_tensors': {
                        0: [[{
                            8: [0]
                        }, {
                            'input_data': [0]
                        }], [[0, 1], 2]],
                        1: [[{
                            9: [1]
                        }], [[0], 2]],
                        2: [[], [[], 1]],
                    },
                    'output_tensors': {
                        0 :[[], [[], 1]],
                        1: [[], [[], 1]],
                        2: [[{
                            10: [0]
                        }], [[0], 1]]
                    },
                    'returns': [7, 9, 10]
                },
            ]
        }

        def _set_attr(head_num, node_names, old_nodes, model, reshape_pos=0, keep_from=1):
            attr = OrderedDict()
            attr['dst_shape'] = '-1,' + str(head_num) + ',-1,-1'
            attr['dims'] = '1,0'

            reshape_node_idx = model.get_node_id(node_names[reshape_pos])
            model.nodes[reshape_node_idx].attr = attr
            keep_idx = keep_from
            while keep_idx < len(old_nodes):
                node_idx = model.get_node_id(node_names[keep_idx])
                model.nodes[node_idx].attr = copy.deepcopy(old_nodes[keep_idx].attr)
                keep_idx += 1

        for i in range(len(pattern_mapping_config['DecoderAttnReshape'])):
            pattern_dict = pattern_mapping_config['DecoderAttnReshape'][i]
            model, new_node_names, ret_old_nodes = util.pattern_mapping("ContentAttnReshape",
                                                                        pattern_dict, model)
            if len(new_node_names) != 0:
                for j in range(len(new_node_names)):
                    pack_node = ret_old_nodes[j][0]
                    head_num = int(pack_node.input_tensors[1].data)
                    _set_attr(head_num, new_node_names[j], ret_old_nodes[j], model)
                return model

        return model
