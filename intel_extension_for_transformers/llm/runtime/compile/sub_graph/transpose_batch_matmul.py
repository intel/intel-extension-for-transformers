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

"""The TransposeBatchMatMul pattern."""

from .pattern import Pattern, pattern_registry
from collections import namedtuple, OrderedDict
import copy
from .. import graph_utils as util
import copy


@pattern_registry(pattern_type='TransposeBatchMatMul')
class TransposeBatchMatMul(Pattern):
    """The TransposeBatchMatMul pattern.

    Fuse the original sub-graph into the custom acceleration 'TransposeBatchMatMul' graph.
    The search strategy is based on the following pattern mapping configs for different models.
    """
    def __call__(self, model):
        """The __call__ function of this pattern class."""
        pattern_mapping_config = {
            'TransposeBatchMatMul': [
                {
                'patterns': {
                    'in': [[(0, 'Transpose'), (2, ['BatchMatMul', 'BatchMatMulV2']), (3, 'Mul'),
                            (4, ['AddV2', 'Add'])],
                           [(), (1, 'Transpose'), (2, ['BatchMatMul', 'BatchMatMulV2'])]],
                    'out': [[(0, 'TransposeBatchMatMul')]]
                },
                'search_mode': 'op_type',
                'node_names': {
                    0: 4
                },
                'input_tensors': {
                    0: [[{
                        0: [0]
                    }, {
                        1: [0]
                    }, {
                        4: [0, 1]
                    }], [[0, 1, 2], 3]]
                },
                'output_tensors': {
                    0: [[{
                        4: [0]
                    }], [[0], 1]]
                },
                'returns': [0, 1, 2, 3]
            },

            # distil_bert_base
            {
                'patterns': {
                    'in': [[(0, 'Transpose'), (1, 'Div'), (3, ['MatMul', 'BatchMatMul']),
                            (4, ['AddV2', 'Add'])],
                           [(), (2, 'Transpose'), (3, ['MatMul', 'BatchMatMul'])]],
                    'out': [[(0, 'TransposeBatchMatMul')]]
                },
                'search_mode': 'op_type',
                'node_names': {
                    0: 4
                },
                'input_tensors': {
                    0: [[{
                        0: [0]
                    }, {
                        2: [0]
                    }, {
                        4: [0, 1]
                    }], [[0, 1, 2], 3]]
                },
                'output_tensors': {
                    0: [[{
                        4: [0]
                    }], [[0], 1]]
                },
                'returns': [0, 2, 3, 1]
            },

            # opennmt encoder
            {
                'patterns': {
                    'in': [[(0, 'Transpose'), (1, 'Div'), (3, ['MatMul', 'BatchMatMul']),
                            (4, 'Cast'), (5, ['AddV2', 'Add'])],
                           [(), (2, 'Transpose'), (3, ['MatMul', 'BatchMatMul'])]],
                    'out': [[(0, 'TransposeBatchMatMul')]]
                },
                'search_mode': 'op_type',
                'node_names': {
                    0: 5
                },
                'input_tensors': {
                    0: [[{
                        0: [0]
                    }, {
                        2: [0]
                    }, {
                        5: [0, 1]
                    }], [[0, 1, 2], 3]]
                },
                'output_tensors': {
                    0: [[{
                        5: [0]
                    }], [[0], 1]]
                },
                'returns': [0, 2, 3, 1]
            },

            # opennmt decoder
            {
                'patterns': {
                    'in': [[(0, 'Transpose'), (1, 'Div'), (3, ['MatMul', 'BatchMatMul']),
                            (4, 'Cast')],
                           [(), (2, 'Transpose'), (3, ['MatMul', 'BatchMatMul'])]],
                    'out': [[(0, 'TransposeBatchMatMul')]]
                },
                'search_mode': 'op_type',
                'node_names': {
                    0: 4
                },
                'input_tensors': {
                    0: [[{
                        0: [0]
                    }, {
                        2: [0]
                    }], [[0, 1], 2]]
                },
                'output_tensors': {
                    0: [[{
                        4: [0]
                    }], [[0], 1]]
                },
                'returns': [0, 2, 3, 1]
            },

            # bert_base_sparse
            {
                'patterns': {
                    'in': [[(0, 'Transpose'), (2, ['MatMul', 'BatchMatMul']), (3, 'Div'),
                            (4, ['AddV2', 'Add'])],
                           [(), (1, 'Transpose'), (2, ['MatMul', 'BatchMatMul'])]],
                    'out': [[(0, 'TransposeBatchMatMul')]]
                },
                'search_mode': 'op_type',
                'node_names': {
                    0: 4
                },
                'input_tensors': {
                    0: [[{
                        0: [0]
                    }, {
                        1: [0]
                    }, {
                        4: [0, 1]
                    }], [[0, 1, 2], 3]]
                },
                'output_tensors': {
                    0: [[{
                        4: [0]
                    }], [[0], 1]]
                },
                'returns': [0, 1, 2, 3]
            },

            # vit
            {
                'patterns': {
                    'in': [[(0, 'Transpose'), (2, ['MatMul', 'BatchMatMul']), (3, 'Div'),
                            (4, 'Softmax')],
                           [(), (1, 'Transpose'), (2, ['MatMul', 'BatchMatMul'])]],
                    'out': [[(0, 'TransposeBatchMatMul'), (1, 'Softmax')]]
                },
                'search_mode': 'op_type',
                'node_names': {
                    0: 2,
                    1: 4
                },
                'input_tensors': {
                    0: [[{
                        0: [0]
                    }, {
                        1: [0]
                    }], [[0, 1], 2]],
                    1: [[], [[], 1]]
                },
                'output_tensors': {
                    0: [[{
                        3: [0]
                    }], [[0], 1]],
                    1: [[{
                        4: [0]
                    }], [[0], 1]]
                },
                'returns': [0, 1, 2, 3, 4]
            },

            # geminet
            {
                'patterns': {
                    'in': [[(0, 'Transpose'), (2, 'FusedMatMul'), (3, ['AddV2', 'Add'])],
                           [(), (1, 'Transpose'), (2, 'FusedMatMul')]],
                    'out': [[(0, 'TransposeBatchMatMul')]]
                },
                'search_mode': 'op_type',
                'node_names': {
                    0: 3
                },
                'input_tensors': {
                    0: [[{
                        0: [0]
                    }, {
                        1: [0]
                    }, {
                        3: [0, 1]
                    }], [[0, 1, 2], 3]]
                },
                'output_tensors': {
                    0: [[{
                        3: [0]
                    }], [[0], 1]]
                },
                'returns': [0, 1, 2]
            },

            # transpose, transpose - BatchMatMul-transpose
            {
                'patterns': {
                    'in': [
                        [(0, 'Transpose'), (2, ['BatchMatMul', 'BatchMatMulV2', 'MatMul']),
                         (3, 'Transpose')],
                        [(), (1, 'Transpose'), (2, ['BatchMatMul', 'BatchMatMulV2', 'MatMul'])]
                        ],
                    'out': [[(0, 'TransposeBatchMatMul')]]
                },
                'search_mode': 'op_type',
                'node_names': {
                    0: 3
                },
                'input_tensors': {
                    0: [[{
                        0: [0]
                    }, {
                        1: [0]
                    }], [[0, 1], 2]]
                },
                'output_tensors': {
                    0: [[{
                        3: [0]
                    }], [[0], 1]]
                },
                'returns': [0, 1, 2, 3]
            },
            # transpose-BatchMatMul-transpose
            # remove one input_tensor after fusion
            {
                'patterns': {
                    'in': [[(0, 'Transpose'), (1, ['BatchMatMul', 'BatchMatMulV2', 'MatMul']),
                            (2, 'Transpose')]],
                    'out': [[(0, 'TransposeBatchMatMul')]]
                },
                'search_mode': 'op_type',
                'node_names': {
                    0: 2
                },
                'input_tensors': {
                    0: [[{
                        0: [0]
                    }, {
                        1: [0]
                    }, {
                        1: [1]
                    }], [[0, 1, 2], 3]]
                },
                'output_tensors': {
                    0: [[{
                        2: [0]
                    }], [[0], 1]]
                },
                'returns': [0, 1, 2]
            },
            # transpose, transpose - BatchMatMul
            {
                'patterns': {
                    'in': [
                        [(0, 'Transpose'), (2, ['BatchMatMul', 'BatchMatMulV2', 'MatMul'])],
                        [(), (1, 'Transpose'), (2, ['BatchMatMul', 'BatchMatMulV2', 'MatMul'])]
                        ],
                    'out': [[(0, 'TransposeBatchMatMul')]]
                },
                'search_mode': 'op_type',
                'node_names': {
                    0: 2
                },
                'input_tensors': {
                    0: [[{
                        0: [0]
                    }, {
                        1: [0]
                    }], [[0, 1], 2]]
                },
                'output_tensors': {
                    0: [[{
                        2: [0]
                    }], [[0], 1]]
                },
                'returns': [0, 1, 2]
            },
            # transpose-BatchMatMul
            # remove one input_tensor after fusion
            {
                'patterns': {
                    'in': [[(0, 'Transpose'), (1, ['BatchMatMul', 'BatchMatMulV2', 'MatMul'])]],
                    'out': [[(0, 'TransposeBatchMatMul')]]
                },
                'search_mode': 'op_type',
                'node_names': {
                    0: 1
                },
                'input_tensors': {
                    0: [[{
                        0: [0]
                    }, {
                        1: [0]
                    }, {
                        1: [1]
                    }], [[0, 1, 2], 3]]
                },
                'output_tensors': {
                    0: [[{
                        1: [0]
                    }], [[0], 1]]
                },
                'returns': [0, 1]
            },
            # BatchMatMul-transpose
            {
                'patterns': {
                    'in': [[(0, ['BatchMatMul', 'BatchMatMulV2', 'MatMul']), (1, 'Transpose')]],
                    'out': [[(0, 'TransposeBatchMatMul')]]
                },
                'search_mode': 'op_type',
                'node_names': {
                    0: 1
                },
                'input_tensors': {
                    0: [[{
                        0: [0]
                    }, {
                        0: [1]
                    }], [[0, 1], 2]]
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

        def _adj_perm(src_perm):
            perm = copy.deepcopy(list(src_perm))
            _x = perm[-2:]
            ret = perm[:-2] + _x[::-1]
            return ret

        for i in range(0, len(pattern_mapping_config['TransposeBatchMatMul'])-5):
            pattern_dict = pattern_mapping_config['TransposeBatchMatMul'][i]
            model, new_node_names, ret_old_nodes = util.pattern_mapping("TransposeBatchMatMul", 
                                                                        pattern_dict, model)
            if len(new_node_names) != 0:
                for j in range(len(new_node_names)):
                    transpose_list = []
                    transpose_list.append(util.str2list(ret_old_nodes[j][0].attr['dst_perm']))
                    transpose_list.append(util.str2list(ret_old_nodes[j][1].attr['dst_perm']))
                    transpose_a = ret_old_nodes[j][2].attr['transpose_a']
                    transpose_b = ret_old_nodes[j][2].attr['transpose_b']
                    if transpose_a:
                        transpose_list[0] = _adj_perm(transpose_list[0])
                    if transpose_b:
                        transpose_list[1] = _adj_perm(transpose_list[1])

                    attr = OrderedDict()
                    attr['src0_perm'] = util.list2str(transpose_list[0])
                    attr['src1_perm'] = util.list2str(transpose_list[1])
                    if len(ret_old_nodes[j]) == 3 \
                        and ret_old_nodes[j][2].op_type in ['FusedMatMul']:
                        output_scale = ret_old_nodes[j][2].attr['alpha']
                    else:
                        if ret_old_nodes[j][3].op_type == 'Div':
                            output_scale = 1.0 / ret_old_nodes[j][3].input_tensors[1].data
                        else:
                            output_scale = ret_old_nodes[j][3].input_tensors[1].data
                    attr['output_scale'] = float(output_scale)
                    attr['format_any'] = False
                    if len(ret_old_nodes[j]) == 5 \
                        and ret_old_nodes[j][4].op_type == 'Softmax':
                        softmax_node_idx = model.get_node_id(new_node_names[j][1])
                        model.nodes[softmax_node_idx].attr = ret_old_nodes[j][4].attr
                    else:
                        if len(model.get_node_by_name(new_node_names[j][0]).input_tensors) == 3:
                            attr['append_op'] = 'binary_add'
                    tb_node_idx = model.get_node_id(new_node_names[j][0])
                    model.nodes[tb_node_idx].attr = attr

        for pattern_dict in pattern_mapping_config['TransposeBatchMatMul'][-5:]:
            model, new_node_names, ret_old_nodes = util.pattern_mapping("TransposeBatchMatMul",
                                                                        pattern_dict, model)
            if len(new_node_names) != 0:
                for i in range(len(new_node_names)):
                    tb_node_idx = model.get_node_id(new_node_names[i][0])
                    perm_dict = OrderedDict()
                    # eliminate one input tensor
                    tb_input_tensors = model.nodes[tb_node_idx].input_tensors
                    if len(tb_input_tensors) == 3:
                        src_t = ret_old_nodes[i][0].output_tensors[0]
                        if src_t.name == tb_input_tensors[1].name:
                            model.nodes[tb_node_idx].input_tensors = [tb_input_tensors[0],
                                                                      tb_input_tensors[2]]
                            perm_dict['src0_perm'] = ret_old_nodes[i][0].attr['dst_perm']
                        else:
                            model.nodes[tb_node_idx].input_tensors = [tb_input_tensors[1],
                                                                      tb_input_tensors[0]]
                            perm_dict['src1_perm'] = ret_old_nodes[i][0].attr['dst_perm']
                    else:
                        if ret_old_nodes[i][0].op_type == 'Transpose':
                            perm_dict['src0_perm'] = ret_old_nodes[i][0].attr['dst_perm']
                            perm_dict['src1_perm'] = ret_old_nodes[i][1].attr['dst_perm']
                    if ret_old_nodes[i][-1].op_type == 'Transpose':
                        perm_dict['dst_perm'] = ret_old_nodes[i][-1].attr['dst_perm']

                    for n in ret_old_nodes[i]:
                        if n.op_type in ['BatchMatMul', 'BatchMatMulV2', 'MatMul']:
                            transpose_a = n.attr.get('transpose_a', False)
                            transpose_b = n.attr.get('transpose_b', False)
                            m_x = n.input_tensors[0]
                            if transpose_a:
                                src0_perm = [j for j in range(len(m_x.shape))]
                                src0_perm = _adj_perm(src0_perm)
                                perm_dict['src0_perm'] = src0_perm
                            if transpose_b:
                                perm_dict['src1_perm'] = _adj_perm(perm_dict['src1_perm'])
                            break
                    model.nodes[tb_node_idx].attr = perm_dict

        return model
