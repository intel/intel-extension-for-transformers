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
from ..ops import Tensor
import numpy as np


@pattern_registry(pattern_type='MultiHeadAttention')
class MultiHeadAttention(Pattern):
    def __call__(self, model):
        from cpuinfo import get_cpu_info
        if 'amx_tile' not in get_cpu_info()['flags']:
            return model
        quant_info = util.get_quant_info()
        if not (quant_info or util.get_autocast_info()['cast_type'] == "bf16"):
            return model

        pattern_mapping_config = {
            'MultiHeadAttention': [
                # for multi head attention
                # Bert based models
                {
                    'patterns': {
                        'in': [[(0, 'TransposeBatchMatMul'), (1, 'Softmax'),
                                (2, 'TransposeBatchMatMul')]],
                        'out': [[(0, 'MultiHeadAttention')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 2,
                    },
                    'input_tensors': {
                        0: [[
                            {0: [0]},  # Q
                            {0: [1]},  # K
                            {2: [1]},  # V
                            {0: [2]},  # mask
                            {0: [3]},  # Q_min
                            {0: [4]},  # Q_max
                            {0: [5]},  # K_min
                            {0: [6]},  # K_max
                            {2: [4]},  # V_min
                            {2: [5]},  # V_max
                            {2: [2]},  # QK_min
                            {2: [3]},  # QK_max
                            {2: [6]},  # dst_min
                            {2: [7]},  # dst_max
                            ],
                            [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], 14]]
                    },
                    'output_tensors': {
                        0 : [[{2: [0]}], [[0], 1]]
                    },
                    'returns': [0, 1, 2]
                },

                # GPT-J torch model
                {
                    'patterns': {
                        'in': [[(0, ['Matmul', 'MatmulwithTranspose', 'BatchMatMul',
                                'TransposeBatchMatMul']), (1, ['Add', 'AddV2', 'BinaryAdd']),
                                (2, ['Div', 'BinaryOp']), (3, ['Add', 'AddV2', 'BinaryAdd']),
                                (4, 'Softmax'), (5, ['Matmul', 'MatmulwithTranspose',
                                'BatchMatMul', 'TransposeBatchMatMul'])]],
                        'out': [[(0, 'MultiHeadAttention')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 5,
                    },
                    'input_tensors': {
                        0: [[
                            {0: [0]},  # Q
                            {0: [1]},  # K
                            {5: [1]},  # V
                            {1: [1]},  # mask_0
                            {3: [1]},  # mask_1
                            {0: [2]},  # Q_min
                            {0: [3]},  # Q_max
                            {0: [4]},  # K_min
                            {0: [5]},  # K_max
                            {5: [4]},  # V_min
                            {5: [5]},  # V_max
                            {5: [2]},  # QK_min
                            {5: [3]},  # QK_max
                            {5: [6]},  # dst_min
                            {5: [7]},  # dst_max
                            ],
                            [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], 15]]
                    },
                    'output_tensors': {
                        0 : [[{5: [0]}], [[0], 1]]
                    },
                    'returns': [0, 1, 2, 3, 4, 5]
                },

                # GPT-J bf16 torch model
                {
                    'patterns': {
                        'in': [[(0, 'MatmulwithTranspose'), (1, 'BinaryAdd'),
                                (2, 'Div'), (3, 'Add'),
                                (4, 'Softmax'), (5, 'Quantize'), (6, 'MatmulwithTranspose')]],
                        'out': [[(0, 'MultiHeadAttention')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 6,
                    },
                    'input_tensors': {
                        0: [[
                            {0: [0]},  # Q
                            {0: [1]},  # K
                            {6: [1]},  # V
                            {3: [1]},  # mask_0
                            {1: [1]},  # mask_1
                            ],
                            [[0, 1, 2, 3, 4], 5]]
                    },
                    'output_tensors': {
                        0 : [[{6: [0]}], [[0], 1]]
                    },
                    'returns': [0, 1, 2, 3, 4, 6]
                },

                # Bert based models bf16
                {
                    'patterns': {
                        'in': [[(0, 'TransposeBatchMatMul'), (1, 'Softmax'),
                                (2, 'TransposeBatchMatMul')]],
                        'out': [[(0, 'MultiHeadAttention')]]
                        },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 2,
                    },
                    'input_tensors': {
                        0: [[
                            {0: [0]},  # Q
                            {0: [1]},  # K
                            {2: [1]},  # V
                            {0: [2]},  # mask
                            ],
                            [[0, 1, 2, 3], 4]]
                    },
                    'output_tensors': {
                        0 : [[{2: [0]}], [[0], 1]]
                    },
                    'returns': [0, 1, 2]
                },

                # GPT-NEOX torch model
                {
                    'patterns': {
                        'in': [[(0, ['Matmul', 'MatmulwithTranspose', 'BatchMatMul', 'TransposeBatchMatMul']),
                                (1, ['Div', 'BinaryOp']), (2, ['Add', 'AddV2', 'BinaryAdd']),
                                (3, ['Add', 'AddV2', 'BinaryAdd']),(4, 'Softmax'), (5, 'Quantize'),
                                (6, ['Matmul', 'MatmulwithTranspose', 'BatchMatMul','TransposeBatchMatMul'])]],
                        'out': [[(0, 'MultiHeadAttention')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 6,
                    },
                    'input_tensors': {
                        0: [[
                            {0: [0]},  # Q
                            {0: [1]},  # K
                            {6: [1]},  # V
                            {3: [1]},  # mask_0
                            {2: [1]},  # mask_1
                            ],
                            [[0, 1, 2, 3, 4], 5]]
                    },
                    'output_tensors': {
                        0 : [[{6: [0]}], [[0], 1]]
                    },
                    'returns': [0, 2, 1, 3, 4, 5, 6]
                },
            ]
        }

        def _set_attr(new_node_names, ret_old_nodes, model):
            for i in range(len(new_node_names)):
                new_node = model.get_node_by_name(new_node_names[i][0])
                attr = OrderedDict()
                if len(ret_old_nodes[i]) == 3:
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
                elif len(ret_old_nodes[i]) == 6:
                    if 'src0_perm' in ret_old_nodes[i][0].attr.keys():
                        attr['Q_perm'] = ret_old_nodes[i][0].attr['src0_perm']
                    if 'src1_perm' in ret_old_nodes[i][0].attr.keys():
                        attr['K_perm'] = ret_old_nodes[i][0].attr['src1_perm']
                    if ret_old_nodes[i][2].attr.get('algorithm', None) == 'div':
                        assert isinstance(ret_old_nodes[i][2].input_tensors[1].data, np.ndarray)
                        attr['output_scale'] = 1 / ret_old_nodes[i][2].input_tensors[1].data.item()
                    if 'src1_perm' in ret_old_nodes[i][5].attr.keys():
                        attr['V_perm'] = ret_old_nodes[i][5].attr['src1_perm']
                    if 'dst_perm' in ret_old_nodes[i][5].attr.keys():
                        attr['dst_perm'] = ret_old_nodes[i][5].attr['dst_perm']
                    if 'reshape' in ret_old_nodes[i][5].attr.keys():
                        attr['reshape'] = ret_old_nodes[i][5].attr['reshape']
                    if 'output_dtype' in ret_old_nodes[i][5].attr.keys():
                        attr['output_dtype'] = ret_old_nodes[i][5].attr['output_dtype']
                elif len(ret_old_nodes[i]) == 7:
                    if 'src0_perm' in ret_old_nodes[i][0].attr.keys():
                        attr['Q_perm'] = ret_old_nodes[i][0].attr['src0_perm']
                    if 'src1_perm' in ret_old_nodes[i][0].attr.keys():
                        attr['K_perm'] = ret_old_nodes[i][0].attr['src1_perm']
                    if ret_old_nodes[i][2].attr.get('algorithm', None) == 'div':
                        assert isinstance(ret_old_nodes[i][2].input_tensors[1].data, np.ndarray)
                        attr['output_scale'] = 1 / ret_old_nodes[i][2].input_tensors[1].data.item()
                    if 'src1_perm' in ret_old_nodes[i][6].attr.keys():
                        attr['V_perm'] = ret_old_nodes[i][6].attr['src1_perm']
                    if 'dst_perm' in ret_old_nodes[i][6].attr.keys():
                        attr['dst_perm'] = ret_old_nodes[i][6].attr['dst_perm']
                    if 'reshape' in ret_old_nodes[i][6].attr.keys():
                        attr['reshape'] = ret_old_nodes[i][6].attr['reshape']
                    if 'output_dtype' in ret_old_nodes[i][6].attr.keys():
                        attr['output_dtype'] = ret_old_nodes[i][6].attr['output_dtype']
                    attr['stable_softmax'] = True

                new_node.attr = attr
                if len(new_node.input_tensors) == 15:
                    mask_1 = new_node.input_tensors[4]
                    if model.get_node_by_name(mask_1.source_op[0]).op_type == "PaddingSequence":
                        new_node.input_tensors[3], new_node.input_tensors[4] = \
                            new_node.input_tensors[4], new_node.input_tensors[3]
        all_fused = True
        for i in range(len(pattern_mapping_config['MultiHeadAttention'])):
            if util.get_autocast_info()['cast_type'] == "bf16" and i in [0,1]:
                continue
            search_pattern = pattern_mapping_config['MultiHeadAttention'][i]['patterns']['in']
            patterns_search_nodes_name = util.search_pattern(search_pattern, model)
            pattern_dict = pattern_mapping_config['MultiHeadAttention'][i]
            model, new_node_names, ret_old_nodes = util.pattern_mapping("MultiHeadAttention",
                                                                        pattern_dict, model)
            if len(new_node_names) != 0:
                if i in [0, 1]:  # entry int8
                    if len(new_node_names) != len(patterns_search_nodes_name):
                        all_fused = False
                    _set_attr(new_node_names, ret_old_nodes, model)

                    for node in model.nodes:
                        if node.op_type == 'PaddingSequence':
                            if all_fused:
                                node.op_type = 'SequenceLength'
                            else:
                                dest_op_list = []
                                for i in range(len(new_node_names)):
                                    dest_op_list.append(new_node_names[i][0])

                                seq_len_output = Tensor(name="mha_seq_len_mask",
                                                        source_op=[node.name + "_seq_len"],
                                                        dest_op=dest_op_list,
                                                        dtype='fp32')

                                seq_len_op = util.construct_node(
                                    node_name=node.name + "_seq_len",
                                    op_type='SequenceLength',
                                    input_tensors=node.input_tensors,
                                    output_tensors=[seq_len_output])

                                insert_idx = model.get_node_id(node.name)
                                model.insert_nodes(insert_idx, [seq_len_op])

                                for i in range(len(new_node_names)):
                                    mha_node = model.get_node_by_name(new_node_names[i][0])
                                    mha_node.input_tensors[3] = seq_len_output
                            return model
        return model
