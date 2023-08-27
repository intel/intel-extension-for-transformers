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

from .pattern import Pattern, pattern_registry
from collections import OrderedDict
from .. import graph_utils as util
from .. import logger


@pattern_registry(pattern_type='GroupNorm')
class GroupNorm(Pattern):
    """
    The input channels are separated into num_groups groups, each containing num_channels / 
    num_groups channels. Each group is calculated like:
    y = (x - E(X)) / (Var(x) + epsilon) * gamma + beta
    More info can see: https://pytorch.org/docs/stable/generated/torch.nn.GroupNorm.html 
    """
    def __call__(self, model):

        pattern_mapping_config = {
            'GroupNorm': [
                # wav2vec2-base / wav2vec2-large
                {
                    'patterns': {
                        'in': [[(0, 'Reshape'), (1, 'InstanceNormalization'), (3, 'Reshape'), (4, 'Mul'),
                                (5, 'Add')], [(), (2, 'Shape'), (3, 'Reshape')]],
                        'out': [[(0, 'GroupNorm')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 5
                    },
                    'input_tensors': {
                        0: [[{
                            0: [0]
                        }, {
                            4: [1]
                        }, {
                            5: [1]
                        }], [[0, 1, 2], 3]]
                    },
                    'output_tensors': {
                        0: [[{
                            5: [0]
                        }], [[0], 1]]
                    },
                    'returns': [1, 4]
                }
            ]
        }

        def _set_attr(group, channels, epsilon, node_names, model):
            attr = OrderedDict()
            attr['group'] = int(group)
            attr['channels'] = int(channels)
            attr['epsilon'] = float(epsilon)
            gn_node_idx = model.get_node_id(node_names[0])
            model.nodes[gn_node_idx].attr = attr

        for i in range(len(pattern_mapping_config['GroupNorm'])):
            pattern_dict = pattern_mapping_config['GroupNorm'][i]
            model, new_node_names, ret_old_nodes = util.pattern_mapping("GroupNorm", pattern_dict, model)
            if len(new_node_names) != 0:
                logger.info('GroupNorm mathched...')
                logger.debug('GroupNorm = {}'.format(new_node_names))
                for j in range(len(new_node_names)):
                    group = ret_old_nodes[j][0].input_tensors[1].shape[0]
                    channels = ret_old_nodes[j][1].input_tensors[1].shape[0]
                    epsilon = 9.999999960041972e-13
                    #epsilon = ret_old_nodes[j][0].attr['epsilon']
                    _set_attr(group, channels, epsilon, new_node_names[j], model)

                return model

        return model