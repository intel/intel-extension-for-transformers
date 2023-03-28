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

"""The EinsumwithArange Pattern."""

from .pattern import Pattern, pattern_registry
from collections import namedtuple, OrderedDict
from .. import graph_utils as util
from ..ops import Tensor
import copy


@pattern_registry(pattern_type='EinsumwithArange')
class EinsumwithArange(Pattern):
    """The InnerproductReshapeFusion pattern.

    Fuse the original sub-graph into the custom acceleration 'EinsumwithArange' graph.
    The search strategy is based on the following pattern mapping configs for different models.
    """
    def __call__(self, model):
        """The __call__ function of this pattern class."""
        pattern_mapping_config = {
            'EinsumwithArange': [
                {
                    'patterns': {
                        'in': [[(0, 'Shape'), (1, 'Arange'), (2, 'Einsum')]
                                ],
                        'out': [[(0, 'Range'), (1, 'Reshape'), (2, 'Matmul')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 1,
                        1: 0,
                        2: 2
                    },
                    'input_tensors': {
                        0: [[{
                            'input_data': [0]
                        }
                        ], [[0], 1]],
                        1: [[], [[], 1]],
                        2: [[{
                            2: [0]
                        }], [[1], 2]]
                    },
                    'output_tensors': {
                        0: [[], [[], 1]],
                        1: [[], [[], 1]],
                        2: [[{
                            2: [0]
                        }], [[0], 1]],
                    },
                    'returns': [0, 1, 2]
                },
                
                {
                    'patterns': {
                        'in': [[(0, 'Shape'), (2, 'Add'), (3, 'Arange'), (4, 'Einsum')],
                               [(), (1, 'Shape'), (2, 'Add')]
                                ],
                        'out': [[(0, 'Range'), (1, 'Reshape'), (2, 'Matmul')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 3,
                        1: 0,
                        2: 4
                    },
                    'input_tensors': {
                        0: [[{
                            'input_data': [0]
                        },{
                            'input_data': [2]
                        }], [[0, 1], 2]],
                        1: [[], [[], 1]],
                        2: [[{
                            4: [1]
                        }], [[1], 2]]
                        
                        
                    },
                    'output_tensors': {
                        0: [[], [[], 1]],
                        1: [[], [[], 1]],
                        2: [[{
                            4: [0]
                        }], [[0], 1]],
                    },
                    'returns': [0]
                }
                
            ]
        }
        if model.framework_modeling_config['framework'] != 'torch':
            return model
        def _set_attr(new_node_names, ret_old_nodes, model):
            for i in range(len(new_node_names)):
                range_node_idx = model.get_node_id(new_node_names[i][0])
                attr = OrderedDict()
                if 'end' in ret_old_nodes[i][0].attr.keys():
                    attr['end_with_shape'] = ret_old_nodes[i][0].attr['end']
                model.nodes[range_node_idx].attr = attr
                
                matmul_node = model.get_node_by_name(new_node_names[i][2])
                reshape_node = model.get_node_by_name(new_node_names[i][1])
                reshape_node.attr = OrderedDict({'dst_shape': '-1, 1'})
                reshape_output = Tensor(name=reshape_node.input_tensors[0].name + "_reshape",
                                        source_op=[matmul_node.name + "_reshape"],
                                        dest_op=[matmul_node.name],
                                        dtype=matmul_node.input_tensors[0].dtype)
                matmul_node.input_tensors[1].dest_op = [matmul_node.name + "_reshape"]
                range_2 = model.get_node_by_name(matmul_node.input_tensors[1].source_op[0])
                range_2.output_tensors[0].dest_op = [matmul_node.name + "_reshape"]
                reshape_op = util.construct_node(
                    node_name=matmul_node.name + "_reshape",
                    op_type='Reshape',
                    input_tensors=[matmul_node.input_tensors[1]],
                    output_tensors=[reshape_output],
                    attr=OrderedDict({'dst_shape': '1, -1'}))
                
                matmul_node.input_tensors[1] = reshape_output
                insert_idx = model.get_node_id(new_node_names[i][2])
                model.insert_nodes(insert_idx, [reshape_op])
                
        pattern_dict = pattern_mapping_config['EinsumwithArange'][0]
        model, new_node_names, ret_old_nodes = util.pattern_mapping("EinsumwithArange", 
                                                                    pattern_dict, model)
        if len(new_node_names) != 0:
            _set_attr(new_node_names, ret_old_nodes, model)

        def _set_attr1(new_node_names, ret_old_nodes, model):
            for i in range(len(new_node_names)):
                range_node_idx = model.get_node_id(new_node_names[i][0])
                attr = OrderedDict()
                attr['algorithm'] = "add"
                attr['end_with_shape'] = 1
                model.nodes[range_node_idx].attr = attr            
                
                matmul_node = model.get_node_by_name(new_node_names[i][2])
                reshape_node = model.get_node_by_name(new_node_names[i][1])
                reshape_node.attr = OrderedDict({'dst_shape': '-1, 1'})
                reshape_output = Tensor(name=reshape_node.input_tensors[0].name + "_reshape",
                                        source_op=[matmul_node.name + "_reshape"],
                                        dest_op=[matmul_node.name],
                                        dtype=matmul_node.input_tensors[0].dtype)
                matmul_node.input_tensors[1].dest_op = [matmul_node.name + "_reshape"]
                range_2 = model.get_node_by_name(matmul_node.input_tensors[1].source_op[0])
                range_2.output_tensors[0].dest_op = [matmul_node.name + "_reshape"]
                reshape_op = util.construct_node(
                    node_name=matmul_node.name + "_reshape",
                    op_type='Reshape',
                    input_tensors=[matmul_node.input_tensors[1]],
                    output_tensors=[reshape_output],
                    attr=OrderedDict({'dst_shape': '1, -1'}))
                matmul_node.input_tensors[1] = reshape_output
                insert_idx = model.get_node_id(new_node_names[i][2])
                model.insert_nodes(insert_idx, [reshape_op])
                
        pattern_dict = pattern_mapping_config['EinsumwithArange'][1]
        model, new_node_names, ret_old_nodes = util.pattern_mapping("EinsumwithArange", 
                                                                    pattern_dict, model)
        if len(new_node_names) != 0:
            _set_attr1(new_node_names, ret_old_nodes, model)


            return model

        return model
