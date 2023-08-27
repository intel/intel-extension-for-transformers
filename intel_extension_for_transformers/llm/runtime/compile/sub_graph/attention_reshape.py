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

"""The AttentionReshape Pattern."""

from .pattern import Pattern, pattern_registry
from collections import namedtuple, OrderedDict
import copy
from .. import graph_utils as util


@pattern_registry(pattern_type='AttentionReshape')
class AttentionReshape(Pattern):
    """The AttentionReshape pattern.

    Fuse the original sub-graph into the custom acceleration 'AttentionReshape' graph.
    The search strategy is based on the following pattern mapping configs for different models.
    """
    def __call__(self, model):
        """The __call__ function of this pattern class."""
        pattern_mapping_config = {
            'AttentionReshape': [
                {
                    'patterns': {
                        'in': [[(0, 'Mul'), (1, 'Pack'), (2, 'Reshape')]],
                        'out': [[(0, 'Reshape')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 2
                    },
                    'input_tensors': {
                        0: [[{
                            2: [0]
                        }], [[0], 1]]
                    },
                    'output_tensors': {
                        0: [[{
                            2: [0]
                        }], [[0], 1]]
                    },
                    'returns': [1]
                },

                # bert_base_sparse
                {
                    'patterns': {
                        'in': [[(0, 'Shape'), (1, 'Gather'), (2, 'Unsqueeze'), (6, 'Concat'),
                                (7, 'Reshape'), (8, 'MatMulWithBias')],
                                [(), (3, 'Shape'), (4, 'Gather'), (5, 'Unsqueeze'),
                                (6, 'Concat')]],
                        'out': [[(0, 'Reshape'), (1, 'MatMulWithBias')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 7,
                        1: 8
                    },
                    'input_tensors': {
                        0: [[{
                            0: [0]
                        }], [[0], 1]],
                        1: [[
                            {8: [1]}, {8: [2]}
                        ], [[1, 2], 3]],
                    },
                    'output_tensors': {
                        0: [[{
                            7: [0]
                        }], [[0], 1]],
                        1: [[{
                            8: [0]
                        }], [[0], 1]]
                    },
                    'returns': [6, 8]
                },

                # Lat_int8
                {
                    'patterns': {
                        'in': [ [(0, 'Shape'), (1, 'Gather'), (2, 'Gather'), (3, 'Unsqueeze'), 
                                (4, 'Concat'), (6, 'Reshape'),(7, 'MatMulWithBias')],
                                [(),(5, 'Transpose'), (6, 'Reshape')]],
                        'out': [[(0, 'Transpose'), (1, 'Reshape'), (2, 'MatMulWithBias')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 5,
                        1: 6,
                        2: 7,
                    },
                    'input_tensors': {
                        0: [[{
                            5: [0]
                        }], [[0], 1]],
                        1: [[], [[],1]],
                        2: [[
                        {7: [1]},{7: [2]}
                        ],[[1, 2],3]],
                    },
                    'output_tensors': {
                        0: [[], [[],1]],

                        1: [[{
                        6:[0]
                        }],[[0],1]],
                        2: [[{
                            7: [0]
                        }], [[0], 1]]
                    },
                    'returns': [5,4,7]
                },

                # Shira new model Reshape_128, 373, 618
                {
                    'patterns': {
                        'in': [[(0, ['MatMul', 'BatchMatMul']), (1, 'Transpose'), (2, 'Shape'),
                                (3, 'Gather'), (4, 'Unsqueeze'), (5, 'Concat'), (6, 'Reshape'),
                                (7, 'MatMulWithBias')]
                                ],
                        'out': [[(0, 'BatchMatMul'), (1, 'Transpose'), (2, 'Reshape'),
                                 (3, 'MatMulWithBias')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 0,
                        1: 1,
                        2: 6,
                        3: 7,
                    },
                    'input_tensors': {
                        0: [[{
                            0: [0]
                        }, {
                            0: [1]
                        }], [[0, 1], 2]],
                        1: [[], [[], 1]],
                        2: [[], [[], 1]],
                        3: [[{
                            7: [1]
                        },{
                            7: [2]
                        }],[[1, 2], 3]],
                    },
                    'output_tensors': {
                        0: [[], [[], 1]],
                        1: [[], [[], 1]],
                        2: [[{
                            6: [0]
                        }],[[0], 1]],
                        3: [[{
                            7: [0]
                        }], [[0], 1]]
                    },
                    'returns': [1, 5, 7, 0]
                },

                # bert_mini_int8
                {
                    'patterns': {
                        'in': [[(0, 'Shape'), (1, 'Gather'), (2, 'Gather'), (3, 'Unsqueeze'), (6, 'Concat'),
                                (7, 'Reshape'), (8, 'MatMulWithBias')],
                                [(1, 'Gather'), (4, 'Gather'), (5, 'Unsqueeze'),
                                (6, 'Concat')]],
                        'out': [[(0, 'Reshape'), (1, 'MatMulWithBias')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 7,
                        1: 8
                    },
                    'input_tensors': {
                        0: [[{
                            7: [0]
                        }], [[0], 1]],
                        1: [[
                            {8: [1]}, {8: [2]}
                        ], [[1, 2], 3]],
                    },
                    'output_tensors': {
                        0: [[{
                            7: [0]
                        }], [[0], 1]],
                        1: [[{
                            8: [0]
                        }], [[0], 1]]
                    },
                    'returns': [6, 8]
                },

                # distil_bert_base
                {
                    'patterns': {
                        'in': [[(0, 'Unsqueeze'), (1, 'Concat'), (2, 'Reshape'),
                                (3, 'MatMulWithBias')]],
                        'out': [[(0, 'Reshape'), (1, 'MatMulWithBias')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 2,
                        1: 3
                    },
                    'input_tensors': {
                        0: [[{
                            2: [0]
                        }], [[0], 1]],
                        1: [[
                            {3: [1]}, {3: [2]}
                        ], [[1, 2], 3]],
                    },
                    'output_tensors': {
                        0: [[{
                            2: [0]
                        }], [[0], 1]],
                        1: [[{
                            3: [0]
                        }], [[0], 1]]
                    },
                    'returns': [1, 3]
                },

                # geminet
                {
                    'patterns': {
                        'in': [[(0, 'Reshape'), (1, 'MatMulWithBias')]],
                        'out': [[(0, 'Reshape'), (1, 'MatMulWithBias')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 0,
                        1: 1
                    },
                    'input_tensors': {
                        0: [[{
                            0: [0]
                        }], [[0], 1]],
                        1: [[
                            {1: [1]}, {1: [2]}
                        ], [[1, 2], 3]],
                    },
                    'output_tensors': {
                        0: [[{
                            0: [0]
                        }], [[0], 1]],
                        1: [[{
                            1: [0]
                        }], [[0], 1]]
                    },
                    'returns': [0, 1]
                },
            ]
        }

        def _set_attr(hidden_size, node_names, model, reshape_idx=0):
            attr = OrderedDict()
            attr['dst_shape'] = '-1,' + str(hidden_size)
            reshape_node_idx = model.get_node_id(node_names[reshape_idx])
            model.nodes[reshape_node_idx].attr = attr

        for i in range(len(pattern_mapping_config['AttentionReshape'])-1):
            pattern_dict = pattern_mapping_config['AttentionReshape'][i]
            model, new_node_names, ret_old_nodes = util.pattern_mapping("AttentionReshape",
                                                                        pattern_dict, model)

            if len(new_node_names) != 0:
                for j in range(len(new_node_names)):
                    if len(ret_old_nodes[j]) == 3 or len(ret_old_nodes[j]) == 4:
                        pack_node = ret_old_nodes[j][1]
                        hidden_size = int(pack_node.input_tensors[-1].data)
                        reshape_idx = 1 if len(ret_old_nodes[j]) == 3 else 2
                        _set_attr(hidden_size, new_node_names[j], model,reshape_idx=reshape_idx)

                        assert ret_old_nodes[j][0].op_type == 'Transpose'
                        idx_get = 0 if len(ret_old_nodes[j]) == 3 else 1
                        trans_node_idx = model.get_node_id(new_node_names[j][idx_get])
                        model.nodes[trans_node_idx].attr = ret_old_nodes[j][0].attr

                        assert ret_old_nodes[j][2].op_type == 'MatMulWithBias'
                        idx_get = 2 if len(ret_old_nodes[j]) == 3 else 3
                        mat_node_idx = model.get_node_id(new_node_names[j][idx_get])
                        model.nodes[mat_node_idx].attr = ret_old_nodes[j][2].attr

                        if ret_old_nodes[j][-1].op_type == 'BatchMatMul':
                            bmat_node_idx = model.get_node_id(new_node_names[j][0])
                            model.nodes[bmat_node_idx].attr = ret_old_nodes[j][-1].attr

                    elif len(ret_old_nodes[j]) == 2:
                        pack_node = ret_old_nodes[j][0]
                        hidden_size = int(pack_node.input_tensors[-1].data)
                        _set_attr(hidden_size, new_node_names[j], model)
                        assert ret_old_nodes[j][1].op_type == 'MatMulWithBias'
                        mat_node_idx = model.get_node_id(new_node_names[j][1])
                        model.nodes[mat_node_idx].attr = ret_old_nodes[j][1].attr
                return model
        # special reshape node, like has '0,0,768' or '-1,369,384' dst_shape attr
        pattern_dict = pattern_mapping_config['AttentionReshape'][-1]
        model, new_node_names, ret_old_nodes = util.pattern_mapping("AttentionReshape",
                                                                    pattern_dict, model)

        if len(new_node_names) != 0:
            for j in range(len(new_node_names)):
                reshape_node = ret_old_nodes[j][0]
                dst_shape = reshape_node.attr.get('dst_shape', None)
                if dst_shape != None and dst_shape.split(',')[0] == '0' or '-1':
                    hidden_size = int(dst_shape.split(',')[-1])
                    _set_attr(hidden_size, new_node_names[j], model)
                mat_node_idx = model.get_node_id(new_node_names[j][1])
                model.nodes[mat_node_idx].attr = ret_old_nodes[j][1].attr

        return model
