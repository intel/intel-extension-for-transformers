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
"""The StableDiffusion_MHAReshape Pattern."""

from .pattern import Pattern, pattern_registry
from collections import OrderedDict
from .. import graph_utils as util
from .subgraph_matcher import EXECUTOR_TYPE
from .. import logger


@pattern_registry(pattern_type='StableDiffusion_MHAReshape')
class StableDiffusion_MHAReshape(Pattern):
    """The StableDiffusion_MHAReshape pattern.

    Fuse the original sub-graph into the custom acceleration 'StableDiffusion_MHAReshape' graph.
    The search strategy is based on the following pattern mapping configs for different models.
    """
    def __call__(self, model):
        """The __call__ function of this pattern class."""
        pattern_mapping_config = {
            'StableDiffusion_MHAReshape': [
                {
                    # unet MHA
                    'patterns': {
                        'in': [[(0, 'Transpose'), (1, 'Reshape'), (4, 'TransposeBatchMatMul'), (5, 'Mul'),
                                (6, 'Softmax'), (9, 'BatchMatMul'), (10, 'Transpose'), (11, 'Reshape')],
                               [(), (2, 'Transpose'), (3, 'Reshape'), (4, 'TransposeBatchMatMul')],
                               [(), (7, 'Transpose'), (8, 'Reshape'), (9, 'BatchMatMul')]],
                        'out': [[(0, 'TransposeBatchMatMul'), (1, 'Softmax'), (2, 'TransposeBatchMatMul')]],
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 4,
                        1: 6,
                        2: 9,
                    },
                    'input_tensors': {
                        0: [[{
                            0: [0]
                        }, {
                            2: [0]
                        }], [[0, 1], 2]],
                        1: [[], [[], 1]],
                        2: [[{
                            7: [0]
                        }], [[1], 2]],
                    },
                    'output_tensors': {
                        0: [[{
                            4: [0]
                        }], [[0], 1]],
                        1: [[{
                            6: [0]
                        }], [[0], 1]],
                        2: [[{
                            11: [0]
                        }], [[0], 1]],
                    },
                    'returns': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
                },
            ]
        }

        pattern_dict = pattern_mapping_config['StableDiffusion_MHAReshape'][0]
        model, new_node_names, ret_old_nodes = util.pattern_mapping("StableDiffusion_MHAReshape",
                                                                    pattern_dict, model)

        if len(new_node_names) != 0:
            logger.info('StableDiffusion_MHAReshape mathched...')
            logger.debug('StableDiffusion_MHAReshape = {}'.format(new_node_names))
            for j in range(len(new_node_names)):
                # the first matmul node
                attr = OrderedDict()
                assert ret_old_nodes[j][0].op_type == 'Transpose'
                matmul_node = model.get_node_by_name(new_node_names[j][0])
                attr['src0_perm'] = ret_old_nodes[j][0].attr['dst_perm']

                assert ret_old_nodes[j][2].op_type == 'Transpose'
                attr['src1_perm'] = '0,2,3,1'

                # output_scales
                output_scale_value = ret_old_nodes[j][5].input_tensors[1].data
                attr['output_scale'] = float(output_scale_value)
                matmul_node.attr = attr

                # the softmax node
                softmax_node = model.get_node_by_name(new_node_names[j][1])
                old_softmax_node = ret_old_nodes[j][6]
                assert softmax_node.op_type == 'Softmax'
                assert old_softmax_node.op_type == 'Softmax'
                softmax_node.attr = old_softmax_node.attr

                # # the second matmul node
                attr = OrderedDict()
                assert ret_old_nodes[j][10].op_type == 'Transpose'
                second_matmul_node = model.get_node_by_name(new_node_names[j][2])
                attr['src1_perm'] = '0,2,1,3'
                attr['dst_perm'] = ret_old_nodes[j][10].attr['dst_perm']

                assert ret_old_nodes[j][11].op_type == 'Reshape'
                attr['reshape'] = ret_old_nodes[j][11].attr['dst_shape']
                second_matmul_node.attr = attr

        return model
