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
"""The TextEncoder_WordEmbedding pattern."""

from .pattern import Pattern, pattern_registry
from collections import OrderedDict
from .. import graph_utils as util
from .. import logger


@pattern_registry(pattern_type='TextEncoder_WordEmbedding')
class TextEncoder_WordEmbedding(Pattern):
    """The TextEncoder_WordEmbedding pattern.

    Fuse the original sub-graph into the custom acceleration 'TextEncoder_WordEmbedding' graph.
    The search strategy is based on the following pattern mapping configs for the stable textEncoderV1-5.
    """
    def __call__(self, model):
        """The __call__ function of this pattern class."""
        pattern_mapping_config = {
            'TextEncoder_WordEmbedding': [
                {
                    'patterns': {
                        'in': [[(0, 'Unsqueeze'), (1, 'Concat'), (2, 'Reshape'), (3, 'Gather'), (4, 'Add')]],
                        'out': [[(0, 'Reshape'), (1, 'Reshape'), (2, 'Gather'), (3, 'Reshape'),
                                 (4, 'Reshape'), (5, 'Add')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 2,
                        1: 'textEncoder_word_embeddings/reshape',
                        2: 3,
                        3: 'textEncoder_word_embeddings/after/reshape',
                        4: 'textEncoder_word_embeddings/add_reshape',
                        5: 4
                    },
                    'input_tensors': {
                        0: [[{
                            2: [0]
                        }], [[0], 1]],
                        1: [[], [[], 1]],
                        2: [[{
                            3: [0]
                        }], [[1], 2]],
                        3: [[{
                            2: [0]
                        }], [[1], 2]],
                        4: [[{
                            2: [0]
                        }], [[1], 2]],
                        5: [[{
                            4: [1]
                        }], [[1], 2]]
                    },
                    'output_tensors': {
                        0: [[{
                            2: [0]
                        }], [[], 1]],
                        1: [[], [[], 1]],
                        2: [[], [[], 1]],
                        3: [[], [[], 1]],
                        4: [[], [[], 1]],
                        5: [[{
                            4: [0]
                        }], [[0], 1]],
                    },
                    'returns': [3, 2]
                },
            ]
        }

        def _set_attr(hidden_size, axis, batch_dims, node_names, model):
            attr0 = OrderedDict()
            attr0['dst_shape'] = '1,-1'
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

            reshape_node_idx = model.get_node_id(node_names[0])
            model.nodes[reshape_node_idx].attr = attr0

            reshape_0_node_idx = model.get_node_id(node_names[1])
            model.nodes[reshape_0_node_idx].attr = attr1

            gather_node_idx = model.get_node_id(node_names[2])
            model.nodes[gather_node_idx].attr = attr2

            reshape_1_node_idx = model.get_node_id(node_names[3])
            model.nodes[reshape_1_node_idx].attr = attr3

            reshape_2_node_idx = model.get_node_id(node_names[4])
            model.nodes[reshape_2_node_idx].attr = attr4

        for i in range(len(pattern_mapping_config['TextEncoder_WordEmbedding'])):
            pattern_dict = pattern_mapping_config['TextEncoder_WordEmbedding'][i]
            model, new_node_names, ret_old_nodes = util.pattern_mapping("TextEncoder_WordEmbedding",
                                                                        pattern_dict, model)

            if len(new_node_names) != 0:
                logger.info('TextEncoder_WordEmbedding mathched...')
                logger.debug('TextEncoder_WordEmbedding = {}'.format(new_node_names))
                for j in range(len(new_node_names)):
                    gatherv2_node = ret_old_nodes[j][0]
                    hidden_size = int(gatherv2_node.input_tensors[0].shape[-1])
                    axis = gatherv2_node.attr['axis']
                    batch_dims = gatherv2_node.attr['batch_dims']
                    _set_attr(hidden_size, axis, batch_dims, new_node_names[j], model)

                return model

        return model
