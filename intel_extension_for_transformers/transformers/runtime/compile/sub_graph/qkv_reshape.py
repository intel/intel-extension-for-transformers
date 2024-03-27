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

"""The QKVReshape Pattern."""

from .pattern import Pattern, pattern_registry
from collections import namedtuple, OrderedDict
import copy
from .. import graph_utils as util


@pattern_registry(pattern_type='QKVReshape')
class QKVReshape(Pattern):
    """The QKVReshape pattern.

    Fuse the original sub-graph into the custom acceleration 'QKVReshape' graph.
    The search strategy is based on the following pattern mapping configs for different models.
    """

    def __call__(self, model):
        """The __call__ function of this pattern class."""
        pattern_mapping_config = {
            'QKVReshape': [
                {
                    'patterns': {
                        'in': [[(0, 'Shape'), (1, 'StridedSlice'), (3, 'Pack'), (4, 'Reshape')],
                               [(0, 'Shape'), (2, 'StridedSlice'), (3, 'Pack')]],
                        'out': [[(0, 'Reshape')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 4
                    },
                    'input_tensors': {
                        0: [[{
                            4: [0]
                        }, {
                            'input_data': [0]
                        }], [[0, 1], 2]]
                    },
                    'output_tensors': {
                        0: [[{
                            4: [0]
                        }], [[0], 1]]
                    },
                    'returns': [3]
                },

                # bert_base_mrpc
                {
                    'patterns': {
                        'in': [[(0, 'Shape'), (1, 'StridedSlice'), (2, 'Pack'), (3, 'Reshape')]],
                        'out': [[(0, 'Reshape')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 3
                    },
                    'input_tensors': {
                        0: [[{
                            3: [0]
                        }, {
                            'input_data': [0]
                        }], [[0, 1], 2]]
                    },
                    'output_tensors': {
                        0: [[{
                            3: [0]
                        }], [[0], 1]]
                    },
                    'returns': [2]
                },

                # bert_base_sparse
                {
                    'patterns': {
                        'in': [[(0, 'MatMulWithBias'), (1, 'Shape'), (2, 'Gather'),
                                (3, 'Unsqueeze'), (7, 'Concat'), (8, 'Reshape')],
                               [(), (4, 'Shape'), (5, 'Gather'), (6, 'Unsqueeze'), (7, 'Concat')]],
                        'out': [[(0, 'MatMulWithBias'), (1, 'Reshape')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 0,
                        1: 8
                    },
                    'input_tensors': {
                        0: [[{
                            0: [0]
                        }, {
                            0: [1]
                        }, {
                            0: [2]
                        }], [[0, 1, 2], 3]],
                        1: [[{
                            'input_data': [0]
                        }], [[1], 2]],
                    },
                    'output_tensors': {
                        0: [[], [[], 1]],
                        1: [[{
                            8: [0]
                        }], [[0], 1]]
                    },
                    'returns': [7, 0]
                },

                # shira new model
                {
                    'patterns': {
                        'in': [[(0, 'MatMulWithBias'), (1, 'Shape'), (2, 'Gather'),
                                (3, 'Unsqueeze'), (4, 'Concat'), (5, 'Reshape')]],
                        'out': [[(0, 'MatMulWithBias'), (1, 'Reshape')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 0,
                        1: 5
                    },
                    'input_tensors': {
                        0: [[{
                            0: [0]
                        }, {
                            0: [1]
                        }, {
                            0: [2]
                        }], [[0, 1, 2], 3]],
                        1: [[{
                            'input_data': [0]
                            #0:[0]
                        }], [[1], 2]],
                    },
                    'output_tensors': {
                        0: [[],[[],1]],

                        1: [[{
                            5: [0]
                        }], [[0], 1]]
                    },
                    'returns': [4, 0]
                },



                # distil_bert_base
               {
                    'patterns': {
                        'in': [[(0, 'Shape'), (1, 'Gather'), (2, 'Unsqueeze'), (3, 'Concat'),
                                (5, 'Reshape')],
                               [(), (4, 'MatMulWithBias'), (5, 'Reshape')]],
                        'out': [[(0, 'MatMulWithBias'), (1, 'Reshape')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 4,
                        1: 5,
                    },
                    'input_tensors': {
                        0: [[{
                            4: [0]
                        }, {
                            4: [1]
                        }, {
                            4: [2]
                        }], [[0, 1, 2], 3]],

                        1: [[
                            {'input_data': [0]
                        }], [[1], 2]],
                    },
                    'output_tensors': {
                        0: [[{
                            4: [0]
                        }], [[0], 1]],
                        1: [[{
                            5: [0]
                        }], [[0], 1]],
                    },
                    'returns': [3, 4, 0]
                },


                # geminet
                {
                    'patterns': {
                        'in': [[(0, 'MatMulWithBias'), (1, 'Reshape')]],
                        'out': [[(0, 'MatMulWithBias'), (1, 'Reshape')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 0,
                        1: 1
                    },
                    'input_tensors': {
                        0: [[{
                            0: [0]
                        }, {
                            0: [1]
                        }, {
                            0: [2]
                        }], [[0, 1, 2], 3]],
                        1: [[{
                            'input_data': [0]
                        }], [[1], 2]],
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

        def _set_attr(head_num, head_size, node_names, model, reshape_pos=0):
            attr = OrderedDict()
            attr['dst_shape'] = '-1,-1,' + str(head_num) + ',' + str(head_size)
            seq_len_first_dim = model.framework_modeling_config.get('seq_len_first_dim', False)
            attr['dims'] = '1' if seq_len_first_dim else '0'

            reshape_node_idx = model.get_node_id(node_names[reshape_pos])
            model.nodes[reshape_node_idx].attr = attr

        for i in range(len(pattern_mapping_config['QKVReshape']) - 1):
            pattern_dict = pattern_mapping_config['QKVReshape'][i]
            model, new_node_names, ret_old_nodes = util.pattern_mapping(
                "QKVReshape", pattern_dict, model)
            if len(new_node_names) != 0:
                for j in range(len(new_node_names)):
                    pack_node = ret_old_nodes[j][0]
                    head_size = int(pack_node.input_tensors[-1].data)
                    head_num = int(pack_node.input_tensors[-2].data)
                    if len(ret_old_nodes[j]) == 2 or len(ret_old_nodes[j]) == 3:
                        assert ret_old_nodes[j][1].op_type == 'MatMulWithBias'
                        mat_node_idx = model.get_node_id(new_node_names[j][0])
                        model.nodes[mat_node_idx].attr = ret_old_nodes[j][1].attr
                        _set_attr(head_num, head_size, new_node_names[j], model, reshape_pos=1)
                        # get QKV reshape dst shape from model input tensors
                        # modify model nodes if original framework graph use other nodes to get
                        # the dst shape
                        if ret_old_nodes[j][-1].op_type == "Shape":
                            try:
                                pre_node = model.get_node_by_name(
                                    ret_old_nodes[j][-1].input_tensors[0].source_op[0])
                            except:
                                pre_node = None
                            if pre_node and pre_node.op_type in ['Transpose']:
                                dest_op = pre_node.output_tensors[0].dest_op
                                if len(dest_op) == 0:
                                    model.remove_nodes([pre_node.name])
                    else:
                        _set_attr(head_num, head_size, new_node_names[j], model)
                return model

        # special reshape node, like has '0,0,12,64' dst_shape attr
        pattern_dict = pattern_mapping_config['QKVReshape'][-1]
        model, new_node_names, ret_old_nodes = util.pattern_mapping("QKVReshape", pattern_dict,
                                                                    model)
        if len(new_node_names) != 0:
            for j in range(len(new_node_names)):
                reshape_node = ret_old_nodes[j][1]
                dst_shape = reshape_node.attr.get('dst_shape', None)
                if dst_shape != None and dst_shape.split(',')[0] == '0':
                    head_num = int(dst_shape.split(',')[-2])
                    reshape_node_idx = model.get_node_id(new_node_names[j][1])
                    model.nodes[reshape_node_idx].attr = OrderedDict({
                        'dst_shape': '-1,-1,' + str(head_num) + ',64',
                        'dims': '0'
                    })
                mat_node_idx = model.get_node_id(new_node_names[j][0])
                model.nodes[mat_node_idx].attr = ret_old_nodes[j][0].attr


        return model
