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
"""The AttentionBlock_ConstantOfShapeWithMul Pattern."""

from .pattern import Pattern, pattern_registry
from .. import graph_utils as util
from .. import logger


@pattern_registry(pattern_type='AttentionBlock_ConstantOfShapeWithMul')
class AttentionBlock_ConstantOfShapeWithMul(Pattern):
    """The AttentionBlock_ConstantOfShapeWithMul pattern.

    Fuse the original sub-graph into the custom acceleration 'AttentionBlock_ConstantOfShapeWithMul' graph.
    The search strategy is based on the following pattern mapping configs for the stable diffusionV1-5.
    """
    def __call__(self, model):
        """The __call__ function of this pattern class."""
        pattern_mapping_config = {
            'AttentionBlock_ConstantOfShapeWithMul': [
                {
                    'patterns': {
                        'in': [[(0, 'Shape'), (1, 'Gather'), (2, 'Unsqueeze'), (9, 'Concat'),
                                (10, 'ConstantOfShape'), (11, 'Mul'), (12, 'Add'), (13, 'Cast'),
                                (14, 'Softmax'), (15, 'Cast')],
                               [(), (3, 'Shape'), (4, 'Gather'), (5, 'Unsqueeze'), (9, 'Concat')],
                               [(), (6, 'Shape'), (7, 'Gather'), (8, 'Unsqueeze'), (9, 'Concat')]],
                    },
                },
                {
                    'patterns': {
                        'in': [[(0, 'Shape'), (1, 'Gather'), (2, 'Unsqueeze'), (9, 'Concat'),
                                (10, 'ConstantOfShape'), (11, 'Mul'), (12, 'Add'), (13, 'Cast'),
                                (14, 'Softmax'), (15, 'Cast')],
                               [(), (3, 'Shape'), (4, 'Gather'), (5, 'Unsqueeze'), (9, 'Concat')],
                               [(), (6, 'Shape'), (7, 'Gather'), (8, 'Unsqueeze'), (9, 'Concat')]],
                        'out': [[(0, 'Softmax')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 14,
                    },
                    'input_tensors': {
                        0: [[{
                            12: [0]
                        }], [[0], 1]],
                    },
                    'output_tensors': {
                        0: [[{
                            15: [0]
                        }], [[0], 1]],
                    },
                    'returns': [9, 11, 14]
                },
                {
                    'patterns': {
                        'in': [[(0, 'Shape'), (1, 'Gather'), (2, 'Unsqueeze'), (9, 'Concat'),
                                (10, 'ConstantOfShape'), (11, 'Mul'), (12, 'Add'), (13, 'Cast'),
                                (14, 'Softmax'), (15, 'Cast')],
                               [(), (3, 'Shape'), (4, 'Gather'), (5, 'Unsqueeze'), (9, 'Concat')],
                               [(), (6, 'Shape'), (7, 'Gather'), (8, 'Unsqueeze'), (9, 'Concat')]],
                        'out': [[(0, 'Reshape'), (1, 'ConstantOfShape'), (2, 'Mul'), (3, 'Add'),
                                 (4, 'Softmax')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 'attentionBlock/constantOfShapeReshape',
                        1: 10,
                        2: 11,
                        3: 12,
                        4: 14
                    },
                    'input_tensors': {
                        0: [[{
                            0: [0]
                        }], [[0], 1]],
                        1: [[], [[], 1]],
                        2: [[{
                            11: [1]
                        }], [[1], 2]],
                        3: [[{
                            12: [0]
                        }], [[0], 2]],
                        4: [[], [[], 1]],
                    },
                    'output_tensors': {
                        0: [[], [[], 1]],
                        1: [[{
                            10: [0]
                        }], [[0], 1]],
                        2: [[{
                            11: [0]
                        }], [[0], 1]],
                        3: [[{
                            12: [0]
                        }], [[0], 1]],
                        4: [[{
                            15: [0]
                        }], [[0], 1]],
                    },
                    'returns': [9, 11, 14]
                },
            ]
        }

        pattern = pattern_mapping_config['AttentionBlock_ConstantOfShapeWithMul'][0]['patterns']['in']
        patterns_nodes_name = util.search_pattern(pattern, model)
        mul = -1
        if len(patterns_nodes_name) != 0:
            logger.info('AttentionBlock_ConstantOfShapeWithMul mathched...')
            logger.debug('AttentionBlock_ConstantOfShapeWithMul = {}'.format(patterns_nodes_name))
            for j in range(len(patterns_nodes_name)):
                mul_idx = model.get_node_id(patterns_nodes_name[j][11])
                if int(model.nodes[mul_idx].input_tensors[1].data) == 0:
                    mul = 0

        if mul == 0:
            pattern_dict = pattern_mapping_config['AttentionBlock_ConstantOfShapeWithMul'][1]
            model, new_node_names, ret_old_nodes = util.pattern_mapping(
                "AttentionBlock_ConstantOfShapeWithMul", pattern_dict, model)
            if len(new_node_names) != 0:
                logger.info('AttentionBlock_ConstantOfShapeWithMul mathched...')
                logger.debug('AttentionBlock_ConstantOfShapeWithMul = {}'.format(patterns_nodes_name))

        return model
