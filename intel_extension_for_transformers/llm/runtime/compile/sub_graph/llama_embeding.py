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

"""The LlamaEmbeddings pattern."""

from .pattern import Pattern, pattern_registry
from collections import namedtuple, OrderedDict
from .. import graph_utils as util
import numpy as np


@pattern_registry(pattern_type='LlamaEmbeddings')
class LlamaEmbeddings(Pattern):
    """The LlamaEmbeddings pattern.

    Fuse the original sub-graph into the custom acceleration 'LlamaEmbeddings' graph.
    The search strategy is based on the following pattern mapping configs for different models.
    """
    def __call__(self, model):
        """The __call__ function of this pattern class."""
        pattern_mapping_config = {
            'LlamaEmbeddings': [
                # llama embedding
                {
                    'patterns': {
                        'in': [[(0, 'Gather'), (1, 'RmsNorm')]
                              ],
                        'out': [[(0, 'Reshape'), (1, 'Gather'),(2, 'Reshape'),
                                 (3, 'RmsNorm')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 'llama_embeddings_before/reshape',
                        1: 0,
                        2: 'llama_embeddings_after/reshape',
                        3: 1
                    },
                    'input_tensors': {
                        0: [[{
                            0: [1]
                        }], [[0], 1]],
                        1: [[{
                            0: [0]
                        }], [[1], 2]],
                        2: [[{
                            0: [1]
                        }], [[1], 2]],
                        3: [[{
                            1: [1]
                        }], [[1], 2]]
                    },
                    'output_tensors': {
                        0: [[], [[], 1]],
                        1: [[], [[], 1]],
                        2: [[], [[], 1]],
                        3: [[{
                            1: [0]
                        }], [[0], 1]],
                    },
                    'returns': [1, 0]
                },
                {
                    'patterns': {
                        'in': [[(0, 'Slice'),(1, 'Unsqueeze'), (2, 'Unsqueeze'), (3, 'Slice'), (7, 'Expand'), 
                                (8,'Rsub'), (9,'ConstantOfShape')],
                               [(), (4, 'Shape'), (7, 'Expand')],
                               [(), (5, 'Shape'), (7, 'Expand')],
                               [(), (6, 'Shape'), (7, 'Expand')]
                              ],
                        'out': [[(0, 'PaddingSequence'), (1, 'ConstantOfShape'),(2, 'BinaryAdd')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 0,
                        1: 9,
                        2: 8,
                    },
                    'input_tensors': {
                        0: [[{
                            0: [0]
                        }], [[0], 1]],
                        1: [[{
                            'input_data': [0]
                        }], [[1], 2]],
                        2: [[], [[], 1]]
                    },
                    'output_tensors': {
                        0: [[], [[], 1]],
                        1: [[], [[], 1]],
                        2: [[{
                            9: [0]
                        }], [[0], 1]],
                    },
                    'returns': [1]
                },
                
                {
                    'patterns': {
                        'in': [[(0, 'Arange'), (1, 'Less'), (3, 'ConstantOfShape'),
                                (5,'Concat'), (6,'Unsqueeze'), (7,'Unsqueeze'), (8,'Slice'), (9, 'Slice'), 
                                (11, 'Expand'), (13, 'Add')],
                               [(), (2, 'Full'), (3, 'ConstantOfShape')],
                               [(), (4, 'Zeros'), (5, 'Concat')],
                               [(), (10, 'Add'), (11, 'Expand')],
                               [(), (12, 'BinaryAdd'), (13, 'Add')]
                              ],
                        'out': [[(0, 'BinaryAdd')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 13
                    },
                    'input_tensors': {
                        0: [[{
                            12: [0],
                        }], [[0], 1]]
                    },
                    'output_tensors': {
                        0: [[{
                            13: [0]
                        }], [[0], 1]],
                    },
                    'returns': [0, 12]
                },
                
            ]
        }

        pattern_dict = pattern_mapping_config['LlamaEmbeddings'][0]
        model, new_node_names, ret_old_nodes = \
            util.pattern_mapping('LlamaEmbeddings', pattern_dict, model)
        if len(new_node_names) != 0:
            for i in range(len(new_node_names)):
                reshape_node1 = model.get_node_by_name(new_node_names[i][0])
                reshape_node1.attr = OrderedDict({'dst_shape': '-1'})
                reshape_node2 = model.get_node_by_name(new_node_names[i][2])
                reshape_node2.attr = OrderedDict({'dst_shape': '-1, -1, -1', 'dims': '0, 1',
                                                  'mul': '0, 1'})
                gather_node = model.get_node_by_name(new_node_names[i][1])
                gather_node.attr = OrderedDict({'axis': '0', 'batch_dims': '0'})
                binary_node = model.get_node_by_name(ret_old_nodes[i][1].output_tensors[0].dest_op[0])
                binary_node.input_tensors[0] = reshape_node2.output_tensors[0]
                rmsnorm_node = model.get_node_by_name(new_node_names[i][3])
                rmsnorm_node.attr = ret_old_nodes[i][0].attr

        pattern_dict = pattern_mapping_config['LlamaEmbeddings'][1]
        model, new_node_names, ret_old_nodes = \
            util.pattern_mapping('LlamaEmbeddings', pattern_dict, model)
        if len(new_node_names) != 0:
            for i in range(len(new_node_names)):
                padding_node = model.get_node_by_name(new_node_names[i][0])
                padding_node.attr = OrderedDict({'dst_shape': '-1,1,1,-1',
                                                 'dims': '1',
                                                 'mode': 'llama'})
                ConstantOfShape = model.get_node_by_name(new_node_names[i][1])
                ConstantOfShape.attr = OrderedDict({'tensor': '1', 'mode': 'llama', 'trilu': '2',
                                                    'value': -10000})
                BinaryAdd = model.get_node_by_name(new_node_names[i][2])
                BinaryAdd.input_tensors.append(padding_node.output_tensors[0])

        pattern_dict = pattern_mapping_config['LlamaEmbeddings'][2]
        model, new_node_names, ret_old_nodes = \
            util.pattern_mapping('LlamaEmbeddings', pattern_dict, model)
        if len(new_node_names) != 0:
            for i in range(len(new_node_names)):
                add_node = model.get_node_by_name(new_node_names[i][0])
                add_node.input_tensors.append(ret_old_nodes[i][1].input_tensors[1])
                remove_add = model.get_node_by_name(
                    ret_old_nodes[i][0].output_tensors[0].dest_op[0])
                model.remove_nodes(remove_add.output_tensors[0].dest_op)
                model.remove_nodes([ret_old_nodes[i][0].output_tensors[0].dest_op[0]])
        model._framework_modeling_config['architecture'] = 'decoder_only'
        return model