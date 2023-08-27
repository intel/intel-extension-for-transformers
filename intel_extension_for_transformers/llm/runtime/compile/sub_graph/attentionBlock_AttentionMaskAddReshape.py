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

"""The AttentionBlock_AttentionMaskAddReshape Pattern."""

from .pattern import Pattern, pattern_registry
from collections import OrderedDict
from .. import graph_utils as util
from .. import logger


@pattern_registry(pattern_type='AttentionBlock_AttentionMaskAddReshape')
class AttentionBlock_AttentionMaskAddReshape(Pattern):
    """The AttentionBlock_AttentionMaskAddReshape pattern.

    Fuse the original sub-graph into the custom acceleration 'AttentionBlock_AttentionMaskAddReshape' graph.
    The search strategy is based on the following pattern mapping configs for the stable diffusionV1-5.
    """
    def __call__(self, model):
        """The __call__ function of this pattern class."""
        pattern_mapping_config = {
            # must be placed behind the QKVPreReshape pattern.
            'AttentionBlock_AttentionMaskAddReshape': [
                {
                    'patterns': {
                        'in': [[(0, 'Shape'), (1, 'Gather'), (2, 'Unsqueeze'), (12, 'Concat'),
                                (13, 'Reshape'), (14, 'Add')],
                               [(), (3, 'Shape'), (4, 'Gather'), (5, 'Unsqueeze'), (12, 'Concat')],
                               [(), (6, 'Shape'), (7, 'Gather'), (8, 'Unsqueeze'), (12, 'Concat')],
                               [(), (9, 'Shape'), (10, 'Gather'), (11, 'Unsqueeze'), (12, 'Concat')]],
                        'out': [[(0, 'Reshape'), (1, 'Add')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 13,
                        1: 14
                    },
                    'input_tensors': {
                        0: [[{
                            13: [0]
                        }, {
                            'input_data': [0]
                        }], [[0, 1], 2]],
                        1: [[{
                            14: [1]
                        }], [[1], 2]],
                    },
                    'output_tensors': {
                        0: [[{
                            13: [0]
                        }], [[0], 1]],
                        1: [[{
                            14: [0]
                        }], [[0], 1]]
                    },
                    'returns': [12]
                },
            ]
        }

        for i in range(len(pattern_mapping_config['AttentionBlock_AttentionMaskAddReshape'])):
            pattern_dict = pattern_mapping_config['AttentionBlock_AttentionMaskAddReshape'][i]
            model, new_node_names, ret_old_nodes = util.pattern_mapping(
                "AttentionBlock_AttentionMaskAddReshape", pattern_dict, model)

            if len(new_node_names) != 0:
                logger.info('AttentionBlock_AttentionMaskAddReshape mathched...')
                logger.debug('AttentionBlock_AttentionMaskAddReshape = {}'.format(new_node_names))
                for j in range(len(new_node_names)):
                    # the first new node
                    attr = OrderedDict()
                    attr['dst_shape'] = '-1,' + '512,' + '-1,-1'
                    attr['dims'] = '0, 2, 3'
                    reshape_node_idx = model.get_node_id(new_node_names[j][0])
                    model.nodes[reshape_node_idx].attr = attr

        return model
