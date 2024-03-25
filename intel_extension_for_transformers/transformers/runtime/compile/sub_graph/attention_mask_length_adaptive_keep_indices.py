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

"""The AttentionMaskLengthAdaptiveExpandIndices Pattern."""

from .pattern import Pattern, pattern_registry
from collections import namedtuple, OrderedDict
from .. import graph_utils as util
import numpy as np


@pattern_registry(pattern_type='AttentionMaskLengthAdaptiveExpandIndices')
class AttentionMaskLengthAdaptiveExpandIndices(Pattern):
    """The AttentionMaskLengthAdaptiveExpandIndices pattern.

    Fuse the original sub-graph into the custom acceleration 'AttentionMaskLengthAdaptiveExpandIndices' graph.
    The fusion strategy is based on 'AddClsToken' pattern map configurations and different kinds of models.
    """

    def __call__(self, model):
        """The __call__ function of this pattern class."""
        pattern_mapping_config = {
            'AttentionMaskLengthAdaptiveExpandIndices': [
                # minilmv2-lat-roberta
                {
                    'patterns': {
                        'in': [[(0, 'Shape'), (1, 'Gather'), (2, 'Unsqueeze'), (9, 'Concat'),
                                (10, 'Reshape'), (11, 'Shape'), (12, 'ConstantOfShape'),
                                (13, 'Mul'), (14, 'Equal'), (15, 'Where'), (18, 'Expand')],
                               [(), (3, 'Shape'), (4, 'Gather'), (5, 'Unsqueeze'), (9, 'Concat')],
                               [(), (6, 'Shape'), (7, 'Gather'), (8, 'Unsqueeze'), (9, 'Concat')],
                               [(), (16, 'Unsqueeze'), (17, 'Unsqueeze'), (18, 'Expand')]],
                        'out': [[(0, 'ExpandIndices')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 18,
                    },
                    'input_tensors': {
                        0: [[{
                            16: [0]
                        }, {
                            0: [0]
                        }], [[0, 1], 2]],
                    },
                    'output_tensors': {
                        0: [[{
                            18: [0]
                        }], [[0], 1]],
                    },
                    'returns': [16, 17]
                },
                # Expand 229
                {
                    'patterns': {
                        'in': [[(0, 'Shape'), (1, 'Gather'), (2, 'Unsqueeze'), (6, 'Concat'),
                                (7, 'Reshape'), (8, 'Shape'), (9, 'ConstantOfShape'),
                                (10, 'Mul'), (11, 'Equal'), (12, 'Where'), (15, 'Expand')],
                               [(), (3, 'Shape'), (4, 'Gather'), (5, 'Unsqueeze'), (6, 'Concat')],
                               [(), (13, 'Unsqueeze'), (14, 'Unsqueeze'), (15, 'Expand')]],
                        'out': [[(0, 'ExpandIndices')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 15,
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
                            15: [0]
                        }], [[0], 1]],
                    },
                    'returns': [13, 14]
                },
                # minilmv2-lat-roberta int8
                {
                    'patterns': {
                        'in': [[(0, 'Shape'), (1, 'Gather'), (2, 'Unsqueeze'), (5, 'Concat'),
                                (6, 'Reshape'), (7, 'Equal'), (8, 'Where'),  (11, 'Expand')],
                               [(0, 'Shape'), (3, 'Gather'), (4, 'Unsqueeze'), (5, 'Concat')],
                               [(), (9, 'Unsqueeze'), (10, 'Unsqueeze'), (11, 'Expand')]],
                        'out': [[(0, 'ExpandIndices')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 11,
                    },
                    'input_tensors': {
                        0: [[{
                            9: [0]
                        }, {
                            0: [0]
                        }], [[0, 1], 2]],
                    },
                    'output_tensors': {
                        0: [[{
                            11: [0]
                        }], [[0], 1]],
                    },
                    'returns': [9, 10]
                },
            ]
        }

        # minilmv2-lat-roberta
        for idx, pattern_dict in enumerate(pattern_mapping_config['AttentionMaskLengthAdaptiveExpandIndices']):
            model, new_node_names, ret_old_nodes = \
             util.pattern_mapping('AttentionMaskLengthAdaptiveExpandIndices', pattern_dict, model)
            if len(new_node_names) != 0:
                for i in range(len(new_node_names)):
                    attr = OrderedDict()
                    input_indices = []
                    for unsqueeze_node in ret_old_nodes[i]:
                        input_indices.append(int(unsqueeze_node.attr['axes']))
                        attr['position'] = util.list2str(input_indices)
                        keep_indices_node_idx = model.get_node_id(new_node_names[i][0])
                        model.nodes[keep_indices_node_idx].attr = attr

        return model
