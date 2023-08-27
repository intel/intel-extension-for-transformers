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

"""The InnerproductReshapeFusion Pattern."""

from .pattern import Pattern, pattern_registry
from collections import namedtuple, OrderedDict
from .. import graph_utils as util
import copy


@pattern_registry(pattern_type='InnerproductReshapeFusion')
class InnerproductReshapeFusion(Pattern):
    """The InnerproductReshapeFusion pattern.

    Fuse the original sub-graph into the custom acceleration 'InnerproductReshapeFusion' graph.
    The search strategy is based on the following pattern mapping configs for different models.
    """
    def __call__(self, model):
        """The __call__ function of this pattern class."""
        pattern_mapping_config = {
            'InnerproductReshapeFusion': [
                {
                    'patterns': {
                        'in': [[(0, 'InnerProduct'), (1, 'Shape'), (3, 'View')],
                                [(0, 'InnerProduct'), (2, 'Shape'), (3, 'View')],
                                [(0, 'InnerProduct'), (3, 'View')]
                                ],
                        'out': [[(0, 'InnerProduct')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 0
                    },
                    'input_tensors': {
                        0: [[{
                            0: [0]
                        }, {
                            0: [1]
                        }, {
                            'input_data': [0]
                        }
                        ], [[0, 1, 2], 3]]
                    },
                    'output_tensors': {
                        0: [[{
                            3: [0]
                        }], [[0], 1]]
                    },
                    'returns': [3, 0]
                },
                
                {
                    'patterns': {
                        'in': [[(0, 'InnerProduct'), (1, 'View'), (2, 'Reorder')],
                                ],
                        'out': [[(0, 'InnerProduct'), (1, 'Reorder')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 0,
                        1: 2
                    },
                    'input_tensors': {
                        0: [[{
                            0: [0]
                        }, {
                            0: [1]
                        }, {
                            'input_data': [0]
                        }
                        ], [[0, 1, 2], 3]],
                        
                        1: [[], [[], 1]]
                    },
                    'output_tensors': {
                        0:[[], [[], 1]],
                        1: [[{
                            2: [0]
                        }], [[0], 1]]
                    },
                    'returns': [1, 0]
                },
            ]
        }


        def _set_attr1(new_node_names, ret_old_nodes, model):
            for i in range(len(new_node_names)):
                mat_node_idx = model.get_node_id(new_node_names[i][0])
                ret_mat_node = ret_old_nodes[i][1]
                if len(ret_mat_node.input_tensors) == 3:
                    model.nodes[mat_node_idx].input_tensors.insert(-1, copy.deepcopy(
                        ret_mat_node.input_tensors[-1]))
                attr = OrderedDict()
                if 'shape' in ret_old_nodes[i][0].attr.keys():
                    attr['reshape'] = ret_old_nodes[i][0].attr['shape']
                    attr['reshape_dims'] = '0'
                model.nodes[mat_node_idx].attr = attr
                reorder_node_idx = model.get_node_by_name(new_node_names[i][1])
                attr1 = OrderedDict()
                attr1['src_perm'] = "0,1,2,3"
                attr1['dst_perm'] = "0,2,1,3"
                reorder_node_idx.attr = attr1
                
        def _set_attr(new_node_names, ret_old_nodes, model):
            for i in range(len(new_node_names)):
                mat_node_idx = model.get_node_id(new_node_names[i][0])
                ret_mat_node = ret_old_nodes[i][1]
                if len(ret_mat_node.input_tensors) == 3:
                    model.nodes[mat_node_idx].input_tensors.insert(-1, copy.deepcopy(
                        ret_mat_node.input_tensors[-1]))
                attr = OrderedDict()
                if 'shape' in ret_old_nodes[i][0].attr.keys():
                    attr['reshape'] = ret_old_nodes[i][0].attr['shape']
                    attr['reshape_dims'] = '0'
                model.nodes[mat_node_idx].attr = attr
        
        if model.framework_modeling_config['framework'] == 'torch':
            pattern_dict = pattern_mapping_config['InnerproductReshapeFusion'][0]
            model, new_node_names, ret_old_nodes = util.pattern_mapping("InnerproductReshapeFusion",
                                                                        pattern_dict, model)
            if len(new_node_names) != 0:
                _set_attr(new_node_names, ret_old_nodes, model)


            pattern_dict = pattern_mapping_config['InnerproductReshapeFusion'][1]
            model, new_node_names, ret_old_nodes = util.pattern_mapping("InnerproductReshapeFusion", 
                                                                        pattern_dict, model)
            if len(new_node_names) != 0:
                _set_attr1(new_node_names, ret_old_nodes, model)
                return model

        return model
