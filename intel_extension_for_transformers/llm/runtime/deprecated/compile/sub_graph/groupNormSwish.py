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


@pattern_registry(pattern_type='GroupNormSwish')
class GroupNormSwish(Pattern):
    """
    The input channels are separated into num_groups groups, each containing num_channels / 
    num_groups channels. Each group is calculated like:
    y = (x - E(X)) / (Var(x) + epsilon) * gamma + beta
    More info can see: https://pytorch.org/docs/stable/generated/torch.nn.GroupNorm.html 
    """

    def __call__(self, model):

        pattern_mapping_config = {
            'GroupNormSwish': [
                # wav2vec2-base / wav2vec2-large
                {
                    'patterns': {
                        'in': [[(0, 'GroupNorm'), (1, 'Sigmoid'), (2, 'Mul')]],
                        'out': [[(0, 'GroupNorm')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 0
                    },
                    'input_tensors': {
                        0: [[{
                            0: [0]
                        }, {
                            0: [1]
                        }, {
                            0: [2]
                        }], [[0, 1, 2], 3]]
                    },
                    'output_tensors': {
                        0: [[{
                            2: [0]
                        }], [[0], 1]]
                    },
                    'returns': [0]
                },
            ]
        }

        pattern_dict = pattern_mapping_config['GroupNormSwish'][0]
        model, new_node_names, ret_old_nodes = util.pattern_mapping("GroupNormSwish", pattern_dict, model)
        if len(new_node_names) != 0:
            logger.info('GroupNormSwish mathched...')
            logger.debug('GroupNormSwish = {}'.format(new_node_names))
            for j in range(len(new_node_names)):
                attr = OrderedDict()
                attr = ret_old_nodes[j][0].attr
                attr['append_op'] = 'swish'
                attr['swish_beta'] = 1

                groupnorm_node = model.get_node_by_name(new_node_names[j][0])
                groupnorm_node.attr = attr

        return model
