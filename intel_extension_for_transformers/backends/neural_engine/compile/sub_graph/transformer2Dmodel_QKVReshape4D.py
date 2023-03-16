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
"""The Transformer2Dmodel_QKVReshapeTo4D Pattern."""

from .pattern import Pattern, pattern_registry
from collections import OrderedDict
from .. import graph_utils as util
from .. import logger


@pattern_registry(pattern_type='Transformer2Dmodel_QKVReshapeTo4D')
class Transformer2Dmodel_QKVReshapeTo4D(Pattern):
    """The Transformer2Dmodel_QKVReshapeTo4D pattern.

    Fuse the original sub-graph into the custom acceleration 'Transformer2Dmodel_QKVReshapeTo4D' graph.
    The search strategy is based on the following pattern mapping configs for the stable diffusionV1-5.
    """
    def __call__(self, model):
        """The __call__ function of this pattern class."""
        pattern_mapping_config = {
            # for Reshape 1 - 8, Reshape 1/3/5/7, Reshape2/4/6/8
            'Transformer2Dmodel_QKVReshapeTo4D': [
                {
                    'patterns': {
                        'in': [[(0, 'Shape'), (1, 'Gather'), (2, 'Div'), (3, 'Cast'), (4, 'Cast'),
                                (5, 'Unsqueeze'), (12, 'Concat'), (13, 'Reshape')],
                               [(), (6, 'Shape'), (7, 'Gather'), (8, 'Unsqueeze'), (12, 'Concat')],
                               [(), (9, 'Shape'), (10, 'Gather'), (11, 'Unsqueeze'), (12, 'Concat')]],
                        'out': [[(0, 'Reshape')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 13,
                    },
                    'input_tensors': {
                        0: [[{
                            13: [0]
                        }, {
                            'input_data': [2]
                        }], [[0, 1], 2]],
                    },
                    'output_tensors': {
                        0: [[{
                            13: [0]
                        }], [[0], 1]]
                    },
                    'returns': [12, 13, 0, 2]
                },
            ]
        }

        for i in range(len(pattern_mapping_config['Transformer2Dmodel_QKVReshapeTo4D'])):
            pattern_dict = pattern_mapping_config['Transformer2Dmodel_QKVReshapeTo4D'][i]
            model, new_node_names, ret_old_nodes = util.pattern_mapping("Transformer2Dmodel_QKVReshapeTo4D",
                                                                        pattern_dict, model)

            if len(new_node_names) != 0:
                logger.info('Transformer2Dmodel_QKVReshapeTo4D mathched...')
                logger.debug('Transformer2Dmodel_QKVReshapeTo4D = {}'.format(new_node_names))
                for j in range(len(new_node_names)):
                    concat_node = ret_old_nodes[j][0]
                    concat_node_input_size = len(concat_node.input_tensors)

                    if concat_node_input_size == 4:
                        # get weight from MatMul Node
                        matmul_node_name = ret_old_nodes[j][1].input_tensors[0].source_op[0]
                        matmul_node = model.get_node_by_name(matmul_node_name)
                        # sometimes matmul_node.op_type could be BatchMatmul, /decoder/mid_block/attentions.0/Reshape_7
                        if matmul_node.op_type == 'MatMulWithBias':
                            weight = matmul_node.input_tensors[1].data.shape[1]
                        elif matmul_node.op_type == 'MatMul':
                            weight = matmul_node.input_tensors[1].data.shape[1]

                        div_node = ret_old_nodes[j][3]
                        assert div_node.op_type == 'Div'
                        div_value = int(div_node.input_tensors[1].data)

                        constant_weight_idx = []
                        for idx in range(len(concat_node.input_tensors)):
                            if concat_node.input_tensors[idx].data != None:
                                constant_weight_idx.append(idx)
                                break
                        constant_weight_idx = constant_weight_idx[0]

                        attr = OrderedDict()
                        attr['dst_shape'] = []
                        for i in range(concat_node_input_size):
                            if i == constant_weight_idx:
                                attr['dst_shape'].append(str(div_value))
                            elif i == 3:
                                attr['dst_shape'].append(str(int(weight / div_value)))
                            else:
                                attr['dst_shape'].append('-1')

                        attr['dst_shape'] = ','.join(attr['dst_shape'])
                        attr['dims'] = 0

                        reshape_node_idx = model.get_node_id(new_node_names[j][0])
                        model.nodes[reshape_node_idx].attr = attr

        return model
