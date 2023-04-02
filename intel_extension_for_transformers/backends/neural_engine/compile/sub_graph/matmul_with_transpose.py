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

"""The MatMulWithTranspose pattern."""

from .pattern import Pattern, pattern_registry
from collections import namedtuple, OrderedDict
from .. import graph_utils as util
from ..onnx_utils import bias_to_int32
import copy


@pattern_registry(pattern_type='MatMulWithTranspose')
class MatMulWithTranspose(Pattern):
    """The MatMulWithTranspose pattern.

    Fuse the original sub-graph into the custom acceleration 'MatMulWithTranspose' graph.
    The search strategy is based on the following pattern mapping configs for different models.
    """
    def __call__(self, model):
        """The __call__ function of this pattern class."""
        pattern_mapping_config = {
            'MatMulWithTranspose': [
                {
                    'patterns': {
                        'in': [[(0, 'Reorder'), (1, 'Reorder'), (3, 'Matmul')],
                               [(), (2, 'Reorder'), (3, 'Matmul')]
                               ],
                        'out': [[(0, 'MatmulwithTranspose')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 3
                    },
                    'input_tensors': {
                        0: [[{
                            2: [0]
                        }, {
                            0: [0]
                        }], [[0, 1], 2]]
                    },
                    'output_tensors': {
                        0: [[{
                            3: [0]
                        }], [[0], 1]]
                    },
                    'returns': [0, 1, 2]
                },
                
                {
                    'patterns': {
                        'in': [[(0, 'Reorder'), (1, 'Matmul'), (2, 'Reorder'), (3, 'Shape'), (5, 'View')],
                               [(2, 'Reorder'), (4, 'Shape'), (5, 'View')]
                               ],
                        'out': [[(0, 'Matmul')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 1
                    },
                    'input_tensors': {
                        0: [[{
                            1: [0]
                        }, {
                            0: [0]
                        }], [[0, 1], 2]]
                    },
                    'output_tensors': {
                        0: [[{
                            5: [0]
                        }], [[0], 1]]
                    },
                    'returns': [0, 2, 5]
                },
                
                {
                    'patterns': {
                        'in': [[(0, 'Reorder'), (4, 'Matmul')],
                               [(), (1, 'Reorder'), (2, 'Concat'), (3, 'Reorder'), (4, 'Matmul')]
                               ],
                        'out': [[(0, 'Concat'), (1, 'MatmulwithTranspose')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 2,
                        1: 4,
                    },
                    'input_tensors': {
                        0: [[{
                            2: [0]
                        }, {
                            1: [0]
                        }], [[0, 1], 2]],
                        1: [[{
                            0: [0]
                        }], [[0], 2]]
                    },
                    'output_tensors': {
                        0: [[{
                            2: [0]
                        }], [[0], 1]],
                        1: [[{
                            4: [0]
                        }], [[0], 1]]
                    },
                    'returns': [0, 3, 1]
                },

                {
                    'patterns': {
                        'in': [[(0, 'Reorder'), (1, 'Concat'), (2, 'Matmul'), (3, 'Reorder'),
                                (4, 'Shape'), (6, 'View')],
                               [(3, 'Reorder'), (5, 'Shape'), (6, 'View')]
                               ],
                        'out': [[(0, 'Concat'), (1, 'MatmulwithTranspose')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 1,
                        1: 2,
                    },
                    'input_tensors': {
                        0: [[{
                            1: [0]
                        }, {
                            0: [0]
                        }], [[0, 1], 2]],
                        1: [[{
                            2: [0]
                        }], [[0], 2]]
                    },
                    'output_tensors': {
                         0: [[{
                            1: [0]
                        }], [[0], 1]],
                        1: [[{
                            6: [0]
                        }], [[0], 1]]
                    },
                    'returns': [0, 2, 3, 6]
                },
            ]
        }
        if model.framework_modeling_config['framework'] != 'torch':
            return model
        def _set_attr(new_node_names, ret_old_nodes, model):
            for i in range(len(new_node_names)):
                transpose_a = ret_old_nodes[i][0].attr['dst_perm']
                transpose_b = ret_old_nodes[i][2].attr['dst_perm']
                a_trans = ret_old_nodes[i][1].attr['transpose_dims']
                a_trans = util.str2list(a_trans)
                mat_node_idx = model.get_node_id(new_node_names[i][0])
                attr = OrderedDict()
                if transpose_a:
                    attr['src1_perm'] = transpose_a
                    if a_trans:
                        transpose_a = util.str2list(transpose_a)
                        tmp = transpose_a[a_trans[0]]
                        transpose_a[a_trans[0]] = transpose_a[a_trans[1]]
                        transpose_a[a_trans[1]] = tmp
                        attr['src1_perm'] = util.list2str(transpose_a)
                if transpose_b:
                    attr['src0_perm'] = transpose_b
                model.nodes[mat_node_idx].attr = attr
                
                concat_node =  model.get_node_by_name(model.nodes[mat_node_idx].input_tensors[0].source_op[0])
                concat1_node = model.get_node_by_name(model.nodes[mat_node_idx].input_tensors[1].source_op[0])
                if concat_node.op_type == "Concat":
                    concat_node.attr = OrderedDict({'axis': '3'})
                if concat1_node.op_type == "Concat":
                    concat1_node.attr = OrderedDict({'axis': '3'})    

        pattern_dict = pattern_mapping_config['MatMulWithTranspose'][0]
        model, new_node_names, ret_old_nodes = util.pattern_mapping("MatMulWithTranspose", 
                                                                    pattern_dict, model)
        if len(new_node_names) != 0:
            _set_attr(new_node_names, ret_old_nodes, model)


        pattern_dict = pattern_mapping_config['MatMulWithTranspose'][1]
        model, new_node_names, ret_old_nodes = util.pattern_mapping("MatMulWithTranspose", 
                                                                    pattern_dict, model)
        if len(new_node_names) != 0:
            for i in range(len(new_node_names)):
                transpose_a = ret_old_nodes[i][0].attr['dst_perm']
                transpose_b = ret_old_nodes[i][1].attr['dst_perm']
                mat_node_idx = model.get_node_id(new_node_names[i][0])
                attr = OrderedDict()
                if transpose_a:
                    attr['src1_perm'] = transpose_a
                if transpose_b:
                    attr['dst_perm'] = transpose_b
                reshape_attr = util.str2list(ret_old_nodes[i][2].attr['shape'])
                if reshape_attr:
                    attr['reshape'] = '-1, ' + str(reshape_attr[-1])
                model.nodes[mat_node_idx].attr = attr
                
                
                
        def _set_attr1(new_node_names, ret_old_nodes, model):
            for i in range(len(new_node_names)):
                transpose_a = ret_old_nodes[i][0].attr['dst_perm']
                transpose_dims = util.str2list(ret_old_nodes[i][1].attr.get('transpose_dims',
                                                                            '-1,-2'))
                assert len(transpose_dims) == 2
                b_dim_ori = util.str2list(ret_old_nodes[i][2].attr['dst_perm'])
                b_dim_ori[transpose_dims[0]], b_dim_ori[transpose_dims[1]] = \
                    b_dim_ori[transpose_dims[1]], b_dim_ori[transpose_dims[0]]
                transpose_b = util.list2str(b_dim_ori) #"0,1,3,2"
                # a_trans = ret_old_nodes[i][1].attr['transpose_dims']
                # a_trans = util.str2list(a_trans)
                mat_node_idx = model.get_node_id(new_node_names[i][1])
                attr = OrderedDict()
                if transpose_a:
                    attr['src0_perm'] = transpose_a
                    # if a_trans:
                    #     transpose_b = util.str2list(transpose_a)
                    #     tmp = transpose_a[a_trans[0]]
                    #     transpose_a[a_trans[0]] = transpose_a[a_trans[1]]
                    #     transpose_a[a_trans[1]] = tmp
                    #     attr['src1_perm'] = util.list2str(transpose_a)
                if transpose_b:
                    attr['src1_perm'] = transpose_b
                model.nodes[mat_node_idx].attr = attr
                
                concat_node =  model.get_node_by_name(model.nodes[mat_node_idx].input_tensors[0].source_op[0])
                concat1_node = model.get_node_by_name(model.nodes[mat_node_idx].input_tensors[1].source_op[0])
                if concat_node.op_type == "Concat":
                    concat_node.attr = OrderedDict({'axis': '3'})
                if concat1_node.op_type == "Concat":
                    concat1_node.attr = OrderedDict({'axis': '1'})
                    concat2 = model.get_node_by_name(concat1_node.input_tensors[1].source_op[0])
                    # concat2 = model.get_node_by_name(reorder_node1.input_tensors[0].source_op[0])
                    concat2.attr = OrderedDict({'axis': '3'})
                    
        pattern_dict = pattern_mapping_config['MatMulWithTranspose'][2]
        model, new_node_names, ret_old_nodes = util.pattern_mapping("MatMulWithTranspose", 
                                                                    pattern_dict, model)
        if len(new_node_names) != 0:
            _set_attr1(new_node_names, ret_old_nodes, model)


        pattern_dict = pattern_mapping_config['MatMulWithTranspose'][3]
        model, new_node_names, ret_old_nodes = util.pattern_mapping("MatMulWithTranspose", 
                                                                    pattern_dict, model)
        if len(new_node_names) != 0:
            for i in range(len(new_node_names)):

                mat_node_idx = model.get_node_id(new_node_names[i][1])
                attr = OrderedDict()
                transpose_b = ret_old_nodes[i][0].attr['dst_perm']
                transpose_dst = ret_old_nodes[i][2].attr['dst_perm']
                if transpose_b:
                    attr['src1_perm'] = transpose_b
                if transpose_dst:
                    attr['dst_perm'] = transpose_dst
                reshape_attr = util.str2list(ret_old_nodes[i][3].attr['shape'])
                if reshape_attr:
                    attr['reshape'] = '-1, ' + str(reshape_attr[-1])
                model.nodes[mat_node_idx].attr = attr
                concat_node = model.get_node_by_name(model.nodes[mat_node_idx].input_tensors[1].source_op[0])
                concat_node.attr = OrderedDict({'axis': '1'})
                
            return model
        return model
