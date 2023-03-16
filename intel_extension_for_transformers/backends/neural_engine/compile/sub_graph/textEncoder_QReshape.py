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
"""The TextEncoder_QReshape Pattern."""

from .pattern import Pattern, pattern_registry
from collections import OrderedDict
from .. import graph_utils as util
from .. import logger


@pattern_registry(pattern_type='TextEncoder_QReshape')
class TextEncoder_QReshape(Pattern):
    """The TextEncoder_QReshape pattern.

    Fuse the original sub-graph into the custom acceleration 'TextEncoder_QReshape' graph.
    The search strategy is based on the following pattern mapping configs for the stable diffusionV1-5.
    """
    def __call__(self, model):
        """The __call__ function of this pattern class."""
        pattern_mapping_config = {
            'TextEncoder_QReshape': [
                # for text encoder q_proj reshape(Reshape_2 & Reshape_8)
                {
                    'patterns': {
                        'in': [[(0, 'Unsqueeze'), (2, 'Concat'), (3, 'Reshape'), (4, 'Transpose')],
                               [(), (1, 'Unsqueeze'), (2, 'Concat')]],
                        'out': [[(0, 'Reshape'), (1, 'Transpose')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 3,
                        1: 4
                    },
                    'input_tensors': {
                        0: [[{
                            3: [0]
                        }, {
                            'input_data': [0]
                        }], [[0, 1], 2]],
                        1: [[], [[], 1]],
                    },
                    'output_tensors': {
                        0: [[{
                            3: [0]
                        }], [[0], 1]],
                        1: [[{
                            4: [0]
                        }], [[0], 1]],
                    },
                    'returns': [2, 4]
                },
            ]
        }

        for i in range(len(pattern_mapping_config['TextEncoder_QReshape'])):
            pattern_dict = pattern_mapping_config['TextEncoder_QReshape'][i]
            model, new_node_names, ret_old_nodes = util.pattern_mapping("TextEncoder_QReshape", pattern_dict,
                                                                        model)

            if len(new_node_names) != 0:
                logger.info('TextEncoder_QReshape mathched...')
                logger.debug('TextEncoder_QReshape = {}'.format(new_node_names))
                for j in range(len(new_node_names)):

                    pack_node = ret_old_nodes[j][0]
                    head_size = int(pack_node.input_tensors[-1].data)
                    head_size_idx = 3


                    head_num_idx = False
                    if pack_node.input_tensors[1].data != None:
                        head_num_idx = 1
                        head_num = int(pack_node.input_tensors[1].data)
                    elif pack_node.input_tensors[2].data != None:
                        head_num_idx = 2
                        head_num = int(pack_node.input_tensors[2].data)

                    assert head_num_idx != False

                    attr = OrderedDict()
                    attr['dst_shape'] = []
                    for i in range(len(pack_node.input_tensors)):
                        if i == head_num_idx:
                            attr['dst_shape'].append(str(head_num))
                        elif i == head_size_idx:
                            attr['dst_shape'].append(str(head_size))
                        else:
                            attr['dst_shape'].append('-1')

                    attr['dst_shape'] = ','.join(attr['dst_shape'])
                    attr['dims'] = '0,1'
                    
                    reshape_node_idx = model.get_node_id(new_node_names[j][0])
                    model.nodes[reshape_node_idx].attr = attr

                    assert ret_old_nodes[j][1].op_type == 'Transpose'
                    mat_node_idx = model.get_node_id(new_node_names[j][1])
                    model.nodes[mat_node_idx].attr = ret_old_nodes[j][1].attr

                return model

        return model
