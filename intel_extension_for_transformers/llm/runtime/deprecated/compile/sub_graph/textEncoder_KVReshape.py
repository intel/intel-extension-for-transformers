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
"""The TextEncoder_KVReshape Pattern."""

from .pattern import Pattern, pattern_registry
from collections import OrderedDict
from .. import graph_utils as util
from .. import logger


@pattern_registry(pattern_type='TextEncoder_KVReshape')
class TextEncoder_KVReshape(Pattern):
    """The TextEncoder_KVReshape pattern.

    Fuse the original sub-graph into the custom acceleration 'TextEncoder_KVReshape' graph.
    """
    def __call__(self, model):
        """The __call__ function of this pattern class."""
        pattern_mapping_config = {
            'TextEncoder_KVReshape': [
                # for text encoder k_proj and v_proj (reshape 0 & 1)
                {
                    'patterns': {
                        'in': [[(0, 'Unsqueeze'), (2, 'Concat'), (3, 'Reshape'), (4, 'Transpose')],
                               [(), (1, 'MatMulWithBias'), (3, 'Reshape')]],
                        'out': [[(0, 'MatMulWithBias'), (1, 'Reshape'), (2, 'Transpose')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 1,
                        1: 3,
                        2: 4
                    },
                    'input_tensors': {
                        0: [[{
                            1: [0]
                        }, {
                            1: [1]
                        }, {
                            1: [2]
                        }], [[0, 1, 2], 3]],
                        1: [[{
                            'input_data': [0]
                        }], [[1], 2]],
                        2: [[], [[], 1]],
                    },
                    'output_tensors': {
                        0: [[{
                            1: [0]
                        }], [[0], 1]],
                        1: [[{
                            3: [0]
                        }], [[0], 1]],
                        2: [[{
                            4: [0]
                        }], [[0], 1]],
                    },
                    'returns': [2, 1, 4]
                },
            ]
        }

        def _set_attr(head_num, head_size, node_names, model):
            attr = OrderedDict()
            attr['dst_shape'] = '-1,-1,' + str(head_num) + ',' + str(head_size)
            attr['dims'] = '0'

            reshape_node_idx = model.get_node_id(node_names[1])
            assert model.nodes[reshape_node_idx].op_type == 'Reshape'
            model.nodes[reshape_node_idx].attr = attr

        def _keep_attr(ret_old_nodes, node_names, model):
            output_node_names_idx = 0
            ret_old_names_idx = 1
            MatMulWithBias_node_idx = model.get_node_id(node_names[output_node_names_idx])
            assert ret_old_nodes[ret_old_names_idx].op_type == model.nodes[MatMulWithBias_node_idx].op_type
            model.nodes[MatMulWithBias_node_idx].attr = ret_old_nodes[ret_old_names_idx].attr

            output_node_names_idx = 2
            ret_old_names_idx = 2
            transpose_node_idx = model.get_node_id(node_names[output_node_names_idx])
            assert ret_old_nodes[ret_old_names_idx].op_type == model.nodes[transpose_node_idx].op_type
            model.nodes[transpose_node_idx].attr = ret_old_nodes[ret_old_names_idx].attr

        for i in range(len(pattern_mapping_config['TextEncoder_KVReshape'])):
            pattern_dict = pattern_mapping_config['TextEncoder_KVReshape'][i]
            model, new_node_names, ret_old_nodes = util.pattern_mapping("TextEncoder_KVReshape", pattern_dict,
                                                                        model)

            logger.info('TextEncoder_KVReshape mathched...')
            logger.debug('TextEncoder_KVReshape = {}'.format(new_node_names))
            if len(new_node_names) != 0:
                for j in range(len(new_node_names)):
                    pack_node = ret_old_nodes[j][0]
                    head_size = int(pack_node.input_tensors[-1].data)  # 32
                    head_num = int(pack_node.input_tensors[-2].data)  # 12
                    _set_attr(head_num, head_size, new_node_names[j], model)
                    _keep_attr(ret_old_nodes[j], new_node_names[j], model)
                return model

        return model
