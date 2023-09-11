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

"""The AttentionOutputLayerNormLengthAdaptiveExpandIndices Pattern."""

from .pattern import Pattern, pattern_registry
from collections import namedtuple, OrderedDict
from .. import graph_utils as util
import numpy as np


@pattern_registry(pattern_type='AttentionOutputLayerNormLengthAdaptiveExpandIndices')
class AttentionOutputLayerNormLengthAdaptiveExpandIndices(Pattern):
    """The AttentionOutputLayerNormLengthAdaptiveExpandIndices pattern.

    Fuse the original sub-graph into the custom acceleration graph.
    The search strategy is based on the following pattern mapping configs for different models.
    """

    def __call__(self, model):
        """The __call__ function of this pattern class."""
        pattern_mapping_config = {
            'AttentionOutputLayerNormLengthAdaptiveExpandIndices': [
                # minilmv2-lat-roberta-2-seq
                {
                    'patterns': {
                        'in': [[(0, 'Shape'), (1, 'Gather'), (2, 'Unsqueeze'), (6, 'Concat'),
                                (7, 'Reshape'), (8, 'Shape'), (9, 'ConstantOfShape'), (10, 'Mul'),
                                (11, 'Equal'), (12, 'Where'), (14, 'Expand')],
                               [(), (3, 'Shape'), (4, 'Gather'), (5, 'Unsqueeze'), (6, 'Concat')],
                               [(), (13, 'Unsqueeze'), (14, 'Expand')]],
                        'out': [[(0, 'ExpandIndices')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 14,
                    },
                    'input_tensors': {
                        0: [[{
                            13: [0]
                        }, {
                            0: [0]
                        }], [[0, 1], 2]],
                    },
                    'output_tensors': {
                        0: [[{
                            14: [0]
                        }], [[0], 1]],
                    },
                    'returns': [13]
                },
                # minilmv2-lat-roberta-1-seq
                # concat 155
                {
                    'patterns': {
                        'in': [[(0, 'Shape'), (1, 'Gather'), (2, 'Unsqueeze'), (3, 'Concat'),
                                (4, 'Reshape'), (5, 'Shape'), (6, 'ConstantOfShape'), (7, 'Mul'),
                                (8, 'Equal'), (9, 'Where'), (10, 'Expand')],
                               [(), (11, 'Unsqueeze'), (10, 'Expand')]],
                        'out': [[(0, 'ExpandIndices')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 10,
                    },
                    'input_tensors': {
                        0: [[{
                            11: [0]
                        }, {
                            0: [0]
                        }], [[0, 1], 2]],
                    },
                    'output_tensors': {
                        0: [[{
                            10: [0]
                        }], [[0], 1]],
                    },
                    'returns': [11]
                },
                # ConstantOfShape_573,323
                {
                    'patterns': {
                        'in': [[(0, 'ConstantOfShape'), (1, 'Mul'),
                                (2, 'Equal'), (3, 'Where'), (4, 'Expand'), (5, 'GatherElements')],
                               [(), (6, 'Unsqueeze'), (7, 'Unsqueeze'), (4, 'Expand')]],
                        'out': [[(0, 'ExpandIndices'), (1, 'GatherElements')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 4,
                        1: 5,
                    },
                    'input_tensors': {
                        0: [[{
                            6: [0]
                        }, {
                            5: [0]
                        }], [[0, 1], 2]],

                        1: [[{
                            5: [0]
                        },

                        ], [[0], 2]],
                    },
                    'output_tensors': {
                        0: [[], [[], 1]],
                        1: [[{
                            5: [0]
                        }], [[0], 1]],
                    },
                    'returns': [5,6,7]
                },
                # ConstantOfShape_272, 517
                {
                    'patterns': {
                        'in': [[(0, 'ConstantOfShape'), (1, 'Mul'),
                                (2, 'Equal'), (3, 'Where'), (4, 'Expand'), (5, 'GatherElements')],
                               [(), (6, 'Unsqueeze'), (4, 'Expand')]],
                        'out': [[(0, 'ExpandIndices'), (1, 'GatherElements')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 4,
                        1: 5,
                    },
                    'input_tensors': {
                        0: [[{
                            6: [0]
                        }, {
                            5: [0]
                        }], [[0, 1], 2]],

                        1: [[{
                            5: [0]
                        },

                        ], [[0], 2]],
                    },
                    'output_tensors': {
                        0: [[], [[], 1]],
                        1: [[{
                            5: [0]
                        }], [[0], 1]],
                    },
                    'returns': [5,6]
                },
                
                # int8 lat
                {
                    'patterns': {
                        'in': [[(0, 'Shape'), (1, 'Gather'), (2, 'Unsqueeze'), (3, 'Concat'),
                                (4, 'Reshape'), (5, 'Equal'), (6, 'Where'), (8, 'Expand')],
                               [(), (7, 'Unsqueeze'), (8, 'Expand')]],
                        'out': [[(0, 'ExpandIndices')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 7,
                    },
                    'input_tensors': {
                        0: [[{
                            7: [0]
                        }, {
                            0: [0]
                        }], [[0, 1], 2]],
                    },
                    'output_tensors': {
                        0: [[{
                            8: [0]
                        }], [[0], 1]],
                    },
                    'returns': [7]
                },
            ]
        }

        # minilmv2-lat-roberta
        #for _, pattern_dict in pattern_mapping_config.items():
        for idx in range(len(pattern_mapping_config[
            'AttentionOutputLayerNormLengthAdaptiveExpandIndices'])):

            pattern_dict = pattern_mapping_config[
                'AttentionOutputLayerNormLengthAdaptiveExpandIndices'][idx]
            model, new_node_names, ret_old_nodes = util.pattern_mapping(
                 'AttentionOutputLayerNormLengthAdaptiveExpandIndices', pattern_dict, model)
            if len(new_node_names) != 0:
                for i in range(len(new_node_names)):
                    attr = OrderedDict()
                    attr_gather = OrderedDict()
                    input_indices = []
                    axis_gather = []
                    for ret_old_node in ret_old_nodes[i]:
                        if ret_old_node.op_type == 'Unsqueeze':
                           input_indices.append(int(ret_old_node.attr['axes']))
                        elif ret_old_node.op_type == 'GatherElements':
                            axis_gather.append(int(ret_old_node.attr['axis']))


                    attr['position'] = util.list2str(input_indices)
                    keep_indices_node_idx = model.get_node_id(new_node_names[i][0])
                    model.nodes[keep_indices_node_idx].attr = attr

                    # set attr for gather_elements
                    if 1 < len(new_node_names[i]):
                       attr_gather['axis'] = util.list2str(axis_gather)
                       gather_element_node_idx = model.get_node_id(new_node_names[i][1])
                       model.nodes[gather_element_node_idx].attr = attr_gather



                #return model

        return model
