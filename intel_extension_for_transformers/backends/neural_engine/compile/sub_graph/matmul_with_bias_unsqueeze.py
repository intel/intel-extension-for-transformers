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
"""The MatMulWithBiasUnsqueeze pattern."""

from .pattern import Pattern, pattern_registry
from .. import graph_utils as util
from .. import logger


@pattern_registry(pattern_type='MatMulWithBiasUnsqueeze')
class MatMulWithBiasUnsqueeze(Pattern):
    """The MatMulWithBiasUnsqueeze pattern.

    Fuse the original sub-graph into the custom acceleration 'MatMulWithBiasUnsqueeze' graph.
    The search strategy is based on the following pattern mapping configs for different models.
    """

    def __call__(self, model):
        """The __call__ function of this pattern class."""
        pattern_mapping_config = {
            'MatMulWithBiasUnsqueeze': [
                # unet
                {
                    'patterns': {
                        'in': [[(0, 'MatMulWithBias'), (1, 'Unsqueeze'), (2, 'Unsqueeze')]],
                        'out': [[(0, 'MatMulWithBias')]]
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
                        }, {
                            'input_data': [2]
                        }], [[0, 1, 2, 3], 4]],
                    },
                    'output_tensors': {
                        0: [[{
                            2: [0]
                        }], [[0], 1]],
                    },
                    'returns': [0]
                },
            ]
        }

        pattern_dict = pattern_mapping_config['MatMulWithBiasUnsqueeze'][0]
        model, new_node_names, ret_old_nodes = util.pattern_mapping("MatMulWithBiasUnsqueeze", pattern_dict,
                                                                    model)

        if len(new_node_names) != 0:
            logger.info('MatMulWithBiasUnsqueeze matched...')
            for j in range(len(new_node_names)):
                # the first new node
                assert ret_old_nodes[j][0].op_type == 'MatMulWithBias'
                innerproduct_node_idx = model.get_node_id(new_node_names[j][0])
                innerproduct_node = model.nodes[innerproduct_node_idx]
                innerproduct_node.attr = ret_old_nodes[j][0].attr

                x, y = innerproduct_node.input_tensors[1].shape
                if ret_old_nodes[j][0].attr['src1_perm'] == '0,1':
                    innerproduct_node.attr['reshape'] = '-1,' + str(x) + ',1,1'
                    innerproduct_node.attr['reshape_dims'] = 0
                if ret_old_nodes[j][0].attr['src1_perm'] == '1,0':
                    innerproduct_node.attr['reshape'] = '-1,' + str(y) + ',1,1'
                    innerproduct_node.attr['reshape_dims'] = 0

        return model
