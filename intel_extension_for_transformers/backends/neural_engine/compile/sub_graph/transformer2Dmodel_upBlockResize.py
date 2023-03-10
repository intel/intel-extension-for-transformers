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
"""The Transformer2DModel_UpBlockResize Pattern."""

from .pattern import Pattern, pattern_registry
from collections import OrderedDict
import numpy as np
from .. import graph_utils as util
from ..ops import Tensor


@pattern_registry(pattern_type='Transformer2DModel_UpBlockResize')
class Transformer2DModel_UpBlockResize(Pattern):
    """The Transformer2DModel_UpBlockResize pattern.

    Fuse the original sub-graph into the custom acceleration 'Transformer2DModel_UpBlockResize' graph.
    The search strategy is based on the following pattern mapping configs for the stable diffusionV1-5.
    """
    def __call__(self, model):
        """The __call__ function of this pattern class."""
        pattern_mapping_config = {
            # for Reshape 1 - 8, Reshape 1/3/5/7, Reshape2/4/6/8
            'Transformer2DModel_UpBlockResize': [
                {
                    'patterns': {
                        'in': [[(0, 'Add'), (1, 'Resize'), (2, 'Conv')]],
                        'out': [[(0, 'Add'), (1, 'Gather'), (2, 'Gather'), (3, 'Conv')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 0,
                        1: 'Transformer2DModel_UpBlockResize/extend_4th_dim',
                        2: 'Transformer2DModel_UpBlockResize/extend_3th_dim',
                        3: 2
                    },
                    'input_tensors': {
                        0: [[{
                            0: [0]
                        }, {
                            0: [1]
                        }], [[0, 1], 2]],
                        1: [[], [[], 1]],
                        2: [[], [[], 1]],
                        3: [[{
                            2: [1]
                        }, {
                            2: [2]
                        }], [[1, 2], 3]],
                    },
                    'output_tensors': {
                        0: [[{
                            0: [0]
                        }], [[0], 1]],
                        1: [[], [[], 1]],
                        2: [[], [[], 1]],
                        3: [[{
                            2: [0]
                        }], [[0], 1]],
                    },
                    'returns': [0, 1, 2]
                },
            ]
        }

        for i in range(len(pattern_mapping_config['Transformer2DModel_UpBlockResize'])):
            pattern_dict = pattern_mapping_config['Transformer2DModel_UpBlockResize'][i]
            model, new_node_names, ret_old_nodes = util.pattern_mapping("Transformer2DModel_UpBlockResize",
                                                                        pattern_dict, model)

            if len(new_node_names) != 0:
                print('i = ', i, 'Transformer2DModel_UpBlockResize = ', new_node_names)
                for j in range(len(new_node_names)):
                    # restor the add node
                    assert ret_old_nodes[j][0].op_type == 'Add'
                    add_node_idx = model.get_node_id(new_node_names[j][0])
                    model.nodes[add_node_idx].attr = ret_old_nodes[j][0].attr

                    # restore the conv node
                    assert ret_old_nodes[j][2].op_type == 'Conv'
                    conv_node_idx = model.get_node_id(new_node_names[j][3])
                    model.nodes[conv_node_idx].attr = ret_old_nodes[j][2].attr

                    first_gather_node = model.get_node_by_name(new_node_names[j][1])
                    second_gather_node = model.get_node_by_name(new_node_names[j][2])

                    indices = (np.arange(32, dtype=np.int32) / 2).astype(np.int32)
                    extended_tensor_32 = Tensor(name='extented_range_to_32',
                                                source_op=[],
                                                dest_op=[],
                                                shape=[indices.shape[0]],
                                                data=indices,
                                                dtype='s32')

                    indices = (np.arange(64, dtype=np.int32) / 2).astype(np.int32)
                    extended_tensor_64 = Tensor(name='extented_range_to_64',
                                                source_op=[],
                                                dest_op=[],
                                                shape=[indices.shape[0]],
                                                data=indices,
                                                dtype='s32')

                    extended_tensor_list = []
                    extended_tensor_list.append(extended_tensor_32)
                    extended_tensor_list.append(extended_tensor_64)

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
