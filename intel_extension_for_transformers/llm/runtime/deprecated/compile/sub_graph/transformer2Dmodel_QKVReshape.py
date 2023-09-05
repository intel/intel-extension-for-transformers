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
"""The Transformer2Dmodel_QKVReshape Pattern."""

from .pattern import Pattern, pattern_registry
from collections import OrderedDict
from .. import graph_utils as util
from .. import logger


@pattern_registry(pattern_type='Transformer2Dmodel_QKVReshape')
class Transformer2Dmodel_QKVReshape(Pattern):
    """The Transformer2Dmodel_QKVReshape pattern.

    Fuse the original sub-graph into the custom acceleration 'Transformer2Dmodel_QKVReshape' graph.
    The search strategy is based on the following pattern mapping configs for the stable diffusionV1-5.
    """
    def __call__(self, model):
        """The __call__ function of this pattern class."""
        pattern_mapping_config = {
            # this pattern is espciall for nodes that the length of concat equals 3.
            'Transformer2Dmodel_QKVReshape': [
                # v2-1 only
                {
                    'patterns': {
                        'in': [[(0, 'Mul'), (1, 'Unsqueeze'), (4, 'Concat'), (5, 'Reshape'), (6, 'Cast')],
                               [(), (2, 'Unsqueeze'), (4, 'Concat')], [(), (3, 'Unsqueeze'), (4, 'Concat')]],
                        'out': [[(0, 'Reshape')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 5,
                    },
                    'input_tensors': {
                        0: [[{
                            5: [0]
                        }], [[0], 1]],
                    },
                    'output_tensors': {
                        0: [[{
                            6: [0]
                        }], [[0], 1]]
                    },
                    'returns': [5, 0, 6]
                },
                # v1-4 & v1-5
                {
                    'patterns': {
                        'in': [[(0, 'Mul'), (1, 'Unsqueeze'), (4, 'Concat'), (5, 'Reshape')],
                               [(), (2, 'Unsqueeze'), (4, 'Concat')], [(), (3, 'Unsqueeze'), (4, 'Concat')]],
                        'out': [[(0, 'Reshape')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 5,
                    },
                    'input_tensors': {
                        0: [[{
                            5: [0]
                        }], [[0], 1]],
                    },
                    'output_tensors': {
                        0: [[{
                            5: [0]
                        }], [[0], 1]]
                    },
                    'returns': [5, 0]
                },
            ]
        }

        for i in range(len(pattern_mapping_config['Transformer2Dmodel_QKVReshape'])):
            pattern_dict = pattern_mapping_config['Transformer2Dmodel_QKVReshape'][i]
            model, new_node_names, ret_old_nodes = util.pattern_mapping("Transformer2Dmodel_QKVReshape",
                                                                        pattern_dict, model)
            weight = False
            if len(new_node_names) != 0:
                logger.info('Transformer2Dmodel_QKVReshape mathched...')
                logger.debug('Transformer2Dmodel_QKVReshape = {}'.format(new_node_names))
                for j in range(len(new_node_names)):
                    # the first new node
                    reshape_node_idx = model.get_node_id(new_node_names[j][0])

                    mul_node = ret_old_nodes[j][1]
                    assert mul_node.op_type == 'Mul'
                    mul_value = int(mul_node.input_tensors[1].data)

                    matmul_node_name = model.nodes[reshape_node_idx].output_tensors[0].dest_op[0]
                    matmul_node = model.get_node_by_name(matmul_node_name)
                    # especially for reshape_7 followed by MatMulWithBias
                    if matmul_node.op_type == 'MatMulWithBias':
                        weight = matmul_node.input_tensors[1].data.shape[1]
                        attr = OrderedDict()
                        attr['dst_shape'] = '-1,' + str(weight)
                        model.nodes[reshape_node_idx].attr = attr
                    else:
                        gather_node = model.get_node_by_name(
                            ret_old_nodes[j][1].input_tensors[0].source_op[0])
                        shape_node = model.get_node_by_name(gather_node.input_tensors[0].source_op[0])
                        matmul_node = model.get_node_by_name(shape_node.input_tensors[0].source_op[0])
                        weight = matmul_node.input_tensors[1].data.shape[1]
                        attr = OrderedDict()
                        attr['dst_shape'] = '16,-1,' + str(int(weight / mul_value))
                        model.nodes[reshape_node_idx].attr = attr

        return model
