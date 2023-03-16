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
"""The Transformer2Dmodel_GetSampleBatch pattern."""

from .pattern import Pattern, pattern_registry
from collections import OrderedDict
from .. import graph_utils as util
from .. import logger


@pattern_registry(pattern_type='Transformer2Dmodel_GetSampleBatch')
class Transformer2Dmodel_GetSampleBatch(Pattern):
    """The Transformer2Dmodel_GetSampleBatch pattern.

    Fuse the original sub-graph into the custom acceleration 'Transformer2Dmodel_GetSampleBatch' graph.
    The search strategy is based on the following pattern mapping configs for the stable textEncoderV1-5.
    """
    def __call__(self, model):
        """The __call__ function of this pattern class."""
        pattern_mapping_config = {
            'Transformer2Dmodel_GetSampleBatch': [
                {
                    'patterns': {
                        'in': [[(0, 'Shape'), (1, 'Gather'), (2, 'Unsqueeze'), (3, 'Concat'), (4, 'Reshape'),
                                (5, 'Shape'), (6, 'ConstantOfShape'), (7, 'Mul'), (8, 'Equal'), (9, 'Where'),
                                (10, 'Expand'), (11, 'Unsqueeze'), (12, 'Cast')]],
                        'out': [[(0, 'Reshape'), (1, 'Concat'), (2, 'Unsqueeze')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 'timestep/reshape',
                        1: 10,
                        2: 11
                    },
                    'input_tensors': {
                        0: [[{
                            'input_data': [1]
                        }], [[0], 1]],
                        1: [[{
                            10: [0]
                        }], [[0], 2]],
                        2: [[], [[], 1]],
                    },
                    'output_tensors': {
                        0: [[], [[], 1]],
                        1: [[{
                            10: [0]
                        }], [[0], 1]],
                        2: [[{
                            12: [0]
                        }], [[0], 1]],
                    },
                    'returns': [10, 11]
                },
            ]
        }

        for i in range(len(pattern_mapping_config['Transformer2Dmodel_GetSampleBatch'])):
            pattern_dict = pattern_mapping_config['Transformer2Dmodel_GetSampleBatch'][i]
            model, new_node_names, ret_old_nodes = util.pattern_mapping("Transformer2Dmodel_GetSampleBatch",
                                                                        pattern_dict, model)

            if len(new_node_names) != 0:
                logger.info('Transformer2Dmodel_GetSampleBatch mathched...')
                logger.debug('Transformer2Dmodel_GetSampleBatch = {}'.format(new_node_names))
                for j in range(len(new_node_names)):

                    # the first new node
                    attr = OrderedDict()
                    attr['dst_shape'] = -1
                    reshape_node_idx = model.get_node_id(new_node_names[j][0])
                    reshape_node = model.nodes[reshape_node_idx]
                    reshape_node.attr = attr

                    #the second new node
                    attr = OrderedDict()
                    attr['axis'] = 0
                    concat_node_idx = model.get_node_id(new_node_names[j][1])
                    concat_node = model.nodes[concat_node_idx]
                    concat_node.attr = attr

                    # the third new node
                    assert ret_old_nodes[j][1].op_type == 'Unsqueeze'
                    unsqueeze_node_idx = model.get_node_id(new_node_names[j][2])
                    model.nodes[unsqueeze_node_idx].attr = ret_old_nodes[j][1].attr

                return model

        return model
