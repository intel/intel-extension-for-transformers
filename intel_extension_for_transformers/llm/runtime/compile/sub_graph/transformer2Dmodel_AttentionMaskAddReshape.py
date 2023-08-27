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
"""The Transformer2Dmodel_AttentionMaskAddReshape Pattern."""

from .pattern import Pattern, pattern_registry
from collections import OrderedDict
from .. import graph_utils as util
from .. import logger


@pattern_registry(pattern_type='Transformer2Dmodel_AttentionMaskAddReshape')
class Transformer2Dmodel_AttentionMaskAddReshape(Pattern):
    """The Transformer2Dmodel_AttentionMaskAddReshape pattern.

    Fuse the original sub-graph into the custom acceleration 'Transformer2Dmodel_AttentionMaskAddReshape' graph.
    The search strategy is based on the following pattern mapping configs for the stable diffusionV1-5.
    """
    def __call__(self, model):
        """The __call__ function of this pattern class."""
        pattern_mapping_config = {
            # this pattern must be placed behind the QKVPreReshape pattern
            'Transformer2Dmodel_AttentionMaskAddReshape': [
                {
                    'patterns': {
                        'in': [[(0, 'Shape'), (1, 'Gather'), (2, 'Unsqueeze'), (12, 'Concat'),
                                (13, 'Reshape'), (14, 'Transpose')],
                               [(), (3, 'Shape'), (4, 'Gather'), (5, 'Unsqueeze'), (12, 'Concat')],
                               [(), (6, 'Shape'), (7, 'Gather'), (8, 'Unsqueeze'), (12, 'Concat')],
                               [(), (9, 'Shape'), (10, 'Gather'), (11, 'Unsqueeze'), (12, 'Concat')]],
                        'out': [[(0, 'Reshape'), (1, 'Transpose')]]
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
                        1: [[], [[], 1]],
                    },
                    'output_tensors': {
                        0: [[{
                            13: [0]
                        }], [[0], 1]],
                        1: [[{
                            14: [0]
                        }], [[0], 1]]
                    },
                    'returns': [12, 13, 14, 0]
                },
            ]
        }

        for i in range(len(pattern_mapping_config['Transformer2Dmodel_AttentionMaskAddReshape'])):
            pattern_dict = pattern_mapping_config['Transformer2Dmodel_AttentionMaskAddReshape'][i]
            model, new_node_names, ret_old_nodes = util.pattern_mapping(
                "Transformer2Dmodel_AttentionMaskAddReshape", pattern_dict, model)

            if len(new_node_names) != 0:
                logger.info('Transformer2Dmodel_AttentionMaskAddReshape mathched...')
                logger.debug('Transformer2Dmodel_AttentionMaskAddReshape = {}'.format(new_node_names))
                for j in range(len(new_node_names)):
                    assert ret_old_nodes[j][2].op_type == 'Transpose'
                    mat_node_idx = model.get_node_id(new_node_names[j][1])
                    model.nodes[mat_node_idx].attr = ret_old_nodes[j][2].attr
                    # the first new node
                    new_reshape_node = model.nodes[model.get_node_id(new_node_names[j][0])]
                    add_node = model.get_node_by_name(new_reshape_node.input_tensors[0].source_op[0])

                    matmulwithbias_node = model.get_node_by_name(add_node.input_tensors[0].source_op[0])
                    # stable diffusion v2.1 is Add here.
                    if matmulwithbias_node.op_type == 'MatmulWithBias':
                        output_channel = matmulwithbias_node.input_tensors[1].data.shape[1]

                    # get H * W
                    div_node = model.get_node_by_name(ret_old_nodes[j][3].input_tensors[0].source_op[0])

                    new_reshape_node.input_tensors.pop(1)
                    new_reshape_node.input_tensors.append(div_node.output_tensors[0])
                    attr = OrderedDict()
                    attr['dst_shape'] = '-1,-1,-1,-1'
                    attr['dims'] = '0, 2, 3'
                    new_reshape_node.attr = attr

        return model
