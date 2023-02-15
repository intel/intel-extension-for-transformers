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

from .pattern import Pattern, pattern_registry
from collections import namedtuple, OrderedDict
from .. import graph_utils as util


@pattern_registry(pattern_type='MultiHeadAttention')
class MultiHeadAttention(Pattern):
    def __call__(self, model):

        pattern_mapping_config = {
            'MultiHeadAttention': [
                # for multi head attention
                {
                    'patterns': {
                        'in': [[(0, 'TransposeBatchMatMul'), (1, 'Softmax'), (2, 'TransposeBatchMatMul')]],
                        'out': [[(0, 'MultiHeadAttenion')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 2,
                    },
                    'input_tensors': {
                        0: [[
                            {0: [0]}, 
                            {0: [1]},
                            {2: [1]},
                            {0: [2]},
                            {0: [3]},
                            {0: [4]},
                            {0: [5]},
                            {0: [6]},
                            {2: [4]},
                            {2: [5]},
                            {2: [2]},
                            {2: [3]},
                            {2: [6]},
                            {2: [7]},
                            ], 
                            [[0,1,2,3,4,5,6,7,8,9,10,11,12,13], 14]]
                    },
                    'output_tensors': {
                        0 : [[{2: [0]}], [[0], 1]]
                    },
                    'returns': [0, 1, 2]
                },
            ]
        }

        def _set_attr(new_node_names, ret_old_nodes, model):
            for i in range(len(new_node_names)):
                new_node = model.get_node_by_name(new_node_names[i][0])
                attr = OrderedDict()
                if 'src0_perm' in ret_old_nodes[i][0].attr.keys():
                    attr['Q_perm'] = ret_old_nodes[i][0].attr['src0_perm']
                if 'src1_perm' in ret_old_nodes[i][0].attr.keys():
                    attr['K_perm'] = ret_old_nodes[i][0].attr['src1_perm']
                if 'output_scale' in ret_old_nodes[i][0].attr.keys():
                    attr['output_scale'] = ret_old_nodes[i][0].attr['output_scale']
                if 'src1_perm' in ret_old_nodes[i][2].attr.keys():
                    attr['V_perm'] = ret_old_nodes[i][2].attr['src1_perm']
                if 'dst_perm' in ret_old_nodes[i][2].attr.keys():
                    attr['dst_perm'] = ret_old_nodes[i][2].attr['dst_perm']
                if 'reshape' in ret_old_nodes[i][2].attr.keys():
                    attr['reshape'] = ret_old_nodes[i][2].attr['reshape']
                if 'output_dtype' in ret_old_nodes[i][2].attr.keys():
                    attr['output_dtype'] = ret_old_nodes[i][2].attr['output_dtype']
                new_node.attr = attr
            
        for i in range(len(pattern_mapping_config['MultiHeadAttention'])):
            pattern_dict = pattern_mapping_config['MultiHeadAttention'][i]
            model, new_node_names, ret_old_nodes = util.pattern_mapping("MultiHeadAttention", pattern_dict, model)
            if len(new_node_names) != 0:
                _set_attr(new_node_names, ret_old_nodes, model)
                for node in model.nodes:
                    if node.op_type == 'PaddingSequence':
                        node.op_type = 'SequenceLength'
                        
        return model
