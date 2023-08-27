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
"""The AttentionBlock_QKVReshape Pattern."""

from .pattern import Pattern, pattern_registry
from collections import OrderedDict
from .. import graph_utils as util
from ..ops import Tensor
from .. import logger


@pattern_registry(pattern_type='AttentionBlock_QKVReshape')
class AttentionBlock_QKVReshape(Pattern):
    """The AttentionBlock_QKVReshape pattern.

    Fuse the original sub-graph into the custom acceleration 'AttentionBlock_QKVReshape' graph.
    The search strategy is based on the following pattern mapping configs for the stable diffusionV1-5.
    """
    def __call__(self, model):
        """The __call__ function of this pattern class."""
        pattern_mapping_config = {
            # for Reshape 1 - 8, Reshape 1/3/5/7, Reshape2/4/6/8
            'AttentionBlock_QKVReshape': [
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
                            'input_data': [0]
                        }], [[0, 1], 2]],
                    },
                    'output_tensors': {
                        0: [[{
                            13: [0]
                        }], [[0], 1]]
                    },
                    'returns': [12, 13, 0]
                },
            ]
        }

        for i in range(len(pattern_mapping_config['AttentionBlock_QKVReshape'])):
            pattern_dict = pattern_mapping_config['AttentionBlock_QKVReshape'][i]
            model, new_node_names, ret_old_nodes = util.pattern_mapping("AttentionBlock_QKVReshape",
                                                                        pattern_dict, model)

            if len(new_node_names) != 0:
                logger.info('AttentionBlock_QKVReshape mathched...')
                logger.debug('AttentionBlock_QKVReshape = {}'.format(new_node_names))
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

                        constant_weight_idx = False
                        for idx in range(len(concat_node.input_tensors)):
                            if concat_node.input_tensors[idx].data != None:
                                constant_weight_idx = idx
                        if constant_weight_idx != False:
                            dim = int(concat_node.input_tensors[constant_weight_idx].data)

                        attr = OrderedDict()
                        attr['dst_shape'] = []
                        for i in range(concat_node_input_size):
                            if i == constant_weight_idx:
                                attr['dst_shape'].append(str(dim))
                            elif i == 3:
                                attr['dst_shape'].append(str(int(weight / dim)))
                            else:
                                attr['dst_shape'].append('-1')

                        attr['dst_shape'] = ','.join(attr['dst_shape'])
                        attr['dims'] = 0
                        reshape_node_idx = model.get_node_id(new_node_names[j][0])
                        model.nodes[reshape_node_idx].attr = attr

                    elif concat_node_input_size == 3:
                        # the first new node
                        reshape_node_idx = model.get_node_id(new_node_names[j][0])
                        reshape_node = model.nodes[reshape_node_idx]

                        matmul_node_name = model.nodes[reshape_node_idx].output_tensors[0].dest_op[0]
                        matmul_node = model.get_node_by_name(matmul_node_name)
                        if matmul_node.op_type == 'MatMulWithBias':
                            # the 3D transpose_node: reshape2D -> Innerprodcut -> 3D transpose -> reshape
                            # reshape output to 2D for Innerprodcut
                            weight = matmul_node.input_tensors[1].data.shape[1]
                            attr = OrderedDict()
                            attr['dst_shape'] = '-1,' + str(weight)
                            reshape_node.input_tensors.pop(1)
                            model.nodes[reshape_node_idx].attr = attr

                            # restore to 3d after Innerprodcut
                            transpose_node = model.get_node_by_name(matmul_node.output_tensors[0].dest_op[0])
                            assert transpose_node.op_type == 'Transpose'

                            new_node_name = reshape_node.name + '_3D'
                            new_node_output_tensor_name = reshape_node.output_tensors[0].name + '_3D'

                            output_tensor = matmul_node.output_tensors[0].name + '_3D'
                            input_tensor = [matmul_node.output_tensors[0], model.nodes[0].output_tensors[0]]
                            output_tensor = [
                                Tensor(name=new_node_output_tensor_name,
                                       source_op=[new_node_name],
                                       dest_op=[transpose_node.name],
                                       dtype=reshape_node.output_tensors[0].dtype)
                            ]
                            new_node = util.construct_node(node_name=new_node_name,
                                                           op_type='Reshape',
                                                           input_tensors=input_tensor,
                                                           output_tensors=output_tensor)
                            attr = OrderedDict()
                            attr['dst_shape'] = '-1,-1,' + str(weight)
                            attr['dims'] = 0
                            new_node.attr = attr

                            # the place of insert is the next of the matmul node
                            matmul_node_idx = model.get_node_id(matmul_node.name)
                            transpose_node.input_tensors[0] = new_node.output_tensors[0]
                            model.insert_nodes(matmul_node_idx + 1, [new_node])
                        else:
                            matmul_node_name = ret_old_nodes[j][2].input_tensors[0].source_op[0]
                            matmul_node = model.get_node_by_name(
                                ret_old_nodes[j][2].input_tensors[0].source_op[0])
                            weight = matmul_node.input_tensors[1].data.shape[1]

                            attr = OrderedDict()
                            attr['dst_shape'] = '-1,-1,' + str(weight)
                            attr['dims'] = 0
                            model.nodes[reshape_node_idx].attr = attr

        return model
