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

"""The TokenTypeEmbeddingsV1 pattern."""

from .pattern import Pattern, pattern_registry
from collections import namedtuple, OrderedDict
from .. import graph_utils as util
import numpy as np


@pattern_registry(pattern_type='TokenTypeEmbeddingsV1')
class TokenTypeEmbeddingsV1(Pattern):
    """The TokenTypeEmbeddingsV1 pattern.

    Fuse the original sub-graph into the custom acceleration 'TokenTypeEmbeddingsV1' graph.
    The search strategy is based on the following pattern mapping configs for different models.
    """
    def __call__(self, model):
        """The __call__ function of this pattern class."""
        pattern_mapping_config = {
            'TokenTypeEmbeddingsV1': [
                # roberta_base
                {
                    'patterns': {
                        'in': [[(2, 'Shape'), (3, 'Gather'), (7, 'Unsqueeze'), (8, 'Concat'),
                                (9, 'Reshape'), (10, 'Shape'), (11, 'ConstantOfShape'), 
                                (14, 'Where'), (15, 'Expand'), (16, 'Gather')],
                                [(), (0, 'Shape'), (1, 'Gather'), (6, 'Unsqueeze'), (8, 'Concat')],
                                [(3, 'Gather'), (4, 'Unsqueeze'), (5, 'Slice'), (15, 'Expand')],
                                [(11, 'ConstantOfShape'), (12, 'Mul'), (13, 'Equal'), 
                                (14, 'Where')]],
                        'out': [[(0, 'TokenTypeIds'), (1, 'Reshape'),
                                 (2, 'Gather'), (3, 'Reshape'), (4, 'Reshape')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 'token_type_embeddings/Expand_21',
                        1: 'token_type_embeddings/reshape',
                        2: 16,
                        3: 'token_type_embeddings/after/reshape',
                        4: 'token_type_embeddings/add_reshape',
                    },
                    'input_tensors': {
                        0: [[{
                            'input_data': [0]
                        }, {5: [0]}], [[0, 1], 2]],
                        1: [[], [[], 1]],
                        2: [[{
                            16: [0]
                        }], [[1], 2]],
                        3: [[{
                            'input_data': [0]
                        }], [[1], 2]],
                        4: [[{
                            'input_data': [0]
                        }], [[1], 2]],
                    },
                    'output_tensors': {
                        0: [[], [[], 1]],
                        1: [[], [[], 1]],
                        2: [[], [[], 1]],
                        3: [[], [[], 1]],
                        4: [[{
                            16: [0]
                        }], [[0], 1]],
                    },
                    'returns': [16]
                },
                # roberta_base int8
                {
                    'patterns': {
                        'in': [[(0, 'Shape'), (2, 'Gather'), (3, 'Unsqueeze'), (7, 'Concat'),
                                (8, 'Reshape'), (10, 'Where'), (11, 'Expand'), (12, 'Gather')],
                                [(0, 'Shape'), (1, 'Gather'), (6, 'Unsqueeze'), (7, 'Concat')],
                                [(2, 'Gather'), (4, 'Unsqueeze'), (5, 'Slice'), (11, 'Expand')],
                                [(8, 'Reshape'), (9, 'Equal'), 
                                (10, 'Where')]],
                        'out': [[(0, 'TokenTypeIds'), (1, 'Reshape'),
                                 (2, 'Gather'), (3, 'Reshape'), (4, 'Reshape')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 'token_type_embeddings/Expand_21',
                        1: 'token_type_embeddings/reshape',
                        2: 12,
                        3: 'token_type_embeddings/after/reshape',
                        4: 'token_type_embeddings/add_reshape',
                    },
                    'input_tensors': {
                        0: [[{
                            'input_data': [0]
                        }, {5: [0]}], [[0, 1], 2]],
                        1: [[], [[], 1]],
                        2: [[{
                            12: [0]
                        }], [[1], 2]],
                        3: [[{
                            'input_data': [0]
                        }], [[1], 2]],
                        4: [[{
                            'input_data': [0]
                        }], [[1], 2]],
                    },
                    'output_tensors': {
                        0: [[], [[], 1]],
                        1: [[], [[], 1]],
                        2: [[], [[], 1]],
                        3: [[], [[], 1]],
                        4: [[{
                            12: [0]
                        }], [[0], 1]],
                    },
                    'returns': [12]
                },
            ]
        }

        def _set_attr(hidden_size, axis, batch_dims, node_names, model):
            attr0 = OrderedDict()
            attr0['mode'] = "roberta"
            attr1 = OrderedDict()
            attr1['dst_shape'] = -1
            attr2 = OrderedDict()
            attr2['axis'] = axis
            attr2['batch_dims'] = batch_dims
            attr3 = OrderedDict()
            attr3['dst_shape'] = '-1,-1,' + str(hidden_size)
            attr3['dims'] = '0,1'
            attr4 = OrderedDict()
            attr4['dst_shape'] = '-1,-1,' + str(hidden_size)
            attr4['dims'] = '0,1'
            attr4['mul'] = '1,2'
            
            tokentype_node_idx = model.get_node_id(node_names[0])
            slice_data = model.nodes[tokentype_node_idx].input_tensors[1].data
            model.nodes[tokentype_node_idx].input_tensors[1].data = slice_data.astype(np.int32)
            model.nodes[tokentype_node_idx].input_tensors[1].dtype = 's32'
            model.nodes[tokentype_node_idx].attr = attr0

            reshape_0_node_idx = model.get_node_id(node_names[1])
            model.nodes[reshape_0_node_idx].attr = attr1

            gather_node_idx = model.get_node_id(node_names[2])
            model.nodes[gather_node_idx].attr = attr2

            reshape_1_node_idx = model.get_node_id(node_names[3])
            model.nodes[reshape_1_node_idx].attr = attr3

            reshape_2_node_idx = model.get_node_id(node_names[4])
            model.nodes[reshape_2_node_idx].attr = attr4
        
        # roberta_base
        for i in range(len(pattern_mapping_config['TokenTypeEmbeddingsV1'])):
            pattern_dict = pattern_mapping_config['TokenTypeEmbeddingsV1'][i]
            model, new_node_names, ret_old_nodes = \
                util.pattern_mapping('TokenTypeEmbeddingsV1', pattern_dict, model)
            if len(new_node_names) != 0:
                for i in range(len(new_node_names)):
                    gatherv2_node = ret_old_nodes[i][0]
                    hidden_size = int(gatherv2_node.input_tensors[0].shape[-1])
                    axis = gatherv2_node.attr['axis']
                    batch_dims = gatherv2_node.attr['batch_dims']
                    _set_attr(hidden_size, axis, batch_dims, new_node_names[i], model)

        
        return model