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
"""The AttentionBlock_Resize2Gather Pattern."""

from .pattern import Pattern, pattern_registry
from collections import OrderedDict
import numpy as np
from .. import graph_utils as util
from ..ops import Tensor
from .. import logger


@pattern_registry(pattern_type='AttentionBlock_Resize2Gather')
class AttentionBlock_Resize2Gather(Pattern):
    """The AttentionBlock_Resize2Gather pattern.

    Fuse the original sub-graph into the custom acceleration 'AttentionBlock_Resize2Gather' graph.
    The search strategy is based on the following pattern mapping configs for the stable diffusionV1-5.
    """
    def __call__(self, model):
        """The __call__ function of this pattern class."""
        pattern_mapping_config = {
            'AttentionBlock_Resize2Gather': [
                {
                    'patterns': {
                        'in': [[(0, 'Add'), (1, 'Div'), (2, 'Resize'), (3, 'Conv')]],
                        'out': [[(0, 'Add'), (1, 'Div'), (2, 'Gather'), (3, 'Gather'), (4, 'Conv')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 0,
                        1: 1,
                        2: 'attentionBlock_Resize2Gather/extend_4th_dim',
                        3: 'attentionBlock_Resize2Gather/extend_3th_dim',
                        4: 3
                    },
                    'input_tensors': {
                        0: [[{
                            0: [0]
                        }, {
                            0: [1]
                        }], [[0, 1], 2]],
                        1: [[{
                            1: [1]
                        }], [[1], 2]],
                        2: [[], [[], 1]],
                        3: [[], [[], 1]],
                        4: [[{
                            3: [1]
                        }, {
                            3: [2]
                        }], [[1, 2], 3]],
                    },
                    'output_tensors': {
                        0: [[{
                            0: [0]
                        }], [[0], 1]],
                        1: [[{
                            1: [0]
                        }], [[0], 1]],
                        2: [[], [[], 1]],
                        3: [[], [[], 1]],
                        4: [[{
                            3: [0]
                        }], [[0], 1]],
                    },
                    'returns': [0, 1, 2, 3]
                },
            ]
        }

        for i in range(len(pattern_mapping_config['AttentionBlock_Resize2Gather'])):
            pattern_dict = pattern_mapping_config['AttentionBlock_Resize2Gather'][i]
            model, new_node_names, ret_old_nodes = util.pattern_mapping("AttentionBlock_Resize2Gather",
                                                                        pattern_dict, model)

            if len(new_node_names) != 0:
                logger.info('AttentionBlock_Resize2Gather mathched...')
                logger.debug('AttentionBlock_Resize2Gather = {}'.format(new_node_names))
                for j in range(len(new_node_names)):
                    # restor the add node
                    assert ret_old_nodes[j][0].op_type == 'Add'
                    add_node_idx = model.get_node_id(new_node_names[j][0])
                    model.nodes[add_node_idx].attr = ret_old_nodes[j][0].attr

                    # restore the Div node
                    assert ret_old_nodes[j][1].op_type == 'Div'
                    div_node_idx = model.get_node_id(new_node_names[j][1])
                    model.nodes[div_node_idx].attr = ret_old_nodes[j][1].attr

                    # restore the conv node
                    assert ret_old_nodes[j][3].op_type == 'Conv'
                    conv_node_idx = model.get_node_id(new_node_names[j][4])
                    model.nodes[conv_node_idx].attr = ret_old_nodes[j][3].attr

                    first_gather_node = model.get_node_by_name(new_node_names[j][2])
                    second_gather_node = model.get_node_by_name(new_node_names[j][3])

                    indices = (np.arange(16, dtype=np.int32) / 2).astype(np.int32)
                    extended_tensor_16 = Tensor(name='extend_range_to_16',
                                                source_op=[],
                                                dest_op=[],
                                                shape=[indices.shape[0]],
                                                data=indices,
                                                dtype='s32')

                    indices = (np.arange(128, dtype=np.int32) / 2).astype(np.int32)
                    extended_tensor_128 = Tensor(name='extend_range_to_128',
                                                 source_op=[],
                                                 dest_op=[],
                                                 shape=[indices.shape[0]],
                                                 data=indices,
                                                 dtype='s32')

                    indices = (np.arange(256, dtype=np.int32) / 2).astype(np.int32)
                    extended_tensor_256 = Tensor(name='extend_range_to_256',
                                                 source_op=[],
                                                 dest_op=[],
                                                 shape=[indices.shape[0]],
                                                 data=indices,
                                                 dtype='s32')

                    indices = (np.arange(512, dtype=np.int32) / 2).astype(np.int32)
                    extended_tensor_512 = Tensor(name='extend_range_to_512',
                                                 source_op=[],
                                                 dest_op=[],
                                                 shape=[indices.shape[0]],
                                                 data=indices,
                                                 dtype='s32')

                    extended_tensor_list = []
                    # hard code
                    if model.nodes[conv_node_idx].input_tensors[1].data.shape[0] == 1280:
                        extended_tensor_list.append(extended_tensor_16)
                    else:
                        extended_tensor_list.append(extended_tensor_128)
                    extended_tensor_list.append(extended_tensor_256)
                    extended_tensor_list.append(extended_tensor_512)

                    first_gather_node.input_tensors.insert(0, extended_tensor_list[j])
                    attr = OrderedDict()
                    attr['batch_dims'] = 3
                    attr['axis'] = 0
                    first_gather_node.attr = attr

                    second_gather_node.input_tensors.insert(0, extended_tensor_list[j])
                    attr = OrderedDict()
                    attr['batch_dims'] = 2
                    attr['axis'] = 0
                    second_gather_node.attr = attr

        return model
