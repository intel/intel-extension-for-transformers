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

"""The ResnetBlock_InstanceNormReshapeFromConv Pattern."""

from .pattern import Pattern, pattern_registry
from collections import namedtuple, OrderedDict
import copy
from .. import graph_utils as util


@pattern_registry(pattern_type='ResnetBlock_InstanceNormReshapeFromConv')
class ResnetBlock_InstanceNormReshapeFromConv(Pattern):
    """The ResnetBlock_InstanceNormReshapeFromConv pattern.

    Fuse the original sub-graph into the custom acceleration 'ResnetBlock_InstanceNormReshapeFromConv' graph.
    """

    def __call__(self, model):
        """The __call__ function of this pattern class."""
        pattern_mapping_config = {
            'ResnetBlock_InstanceNormReshapeFromConv': [
                {
                    'patterns': {
                        'in': [[(0, 'Conv'), (1, 'Reshape'), (2, 'InstanceNormalization'),(4, 'Reshape')],
                        [(0, 'Conv'), (3, 'Shape'), (4, 'Reshape')]],
                        'out': [[(0, 'Conv'), (1, 'Reshape'),(2, 'InstanceNormalization'), (3, 'Reshape')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 0,
                        1: 1,
                        2: 2,
                        3: 4
                    },
                    'input_tensors': {
                        0: [[{0: [0]}, {0: [1]}, {0: [2]}], [[0, 1, 2], 3]],
                        1: [[{'input_data': [0]}], [[1], 2]],
                        2: [[{2: [1]}, {2: [2]}], [[1, 2], 3]],
                        3: [[{'input_data': [0]}], [[1], 2]],
                    },
                    'output_tensors': {
                        0: [[{0: [0]}], [[0], 1]],
                        1: [[{1: [0]}], [[0], 1]],
                        2: [[{2: [0]}], [[0], 1]],
                        3: [[{4: [0]}], [[0], 1]],
                    },
                    'returns': [0, 1, 2, 3, 4]
                },
            ]
        }

        def _set_attr(batch_size, out_channel, height, weight, node_names, model):
            attr = OrderedDict()
            attr['dst_shape'] = str(batch_size) + ',' + str(out_channel) + ',' + str(height) + ',' + str(weight)
            attr['dims'] = '0, 2, 3'

            reshape_node_idx = model.get_node_id(node_names[3])
            assert  model.nodes[reshape_node_idx].op_type == 'Reshape'
            model.nodes[reshape_node_idx].attr = attr

        for i in range(len(pattern_mapping_config['ResnetBlock_InstanceNormReshapeFromConv'])):
            pattern_dict = pattern_mapping_config['ResnetBlock_InstanceNormReshapeFromConv'][i]
            model, new_node_names, ret_old_nodes = util.pattern_mapping(
                "ResnetBlock_InstanceNormReshapeFromConv", pattern_dict, model)

            print('i = ', i, 'ResnetBlock_InstanceNormReshapeFromConv = ', new_node_names)
            if len(new_node_names) != 0:
                for j in range(len(new_node_names)):
                    first_ret_old_node = ret_old_nodes[j][0]
                    assert first_ret_old_node.op_type == 'Conv'
                    conv_node_idx = model.get_node_id(new_node_names[j][0])
                    model.nodes[conv_node_idx].attr = first_ret_old_node.attr

                    second_ret_old_node = ret_old_nodes[j][1]
                    assert second_ret_old_node.op_type == 'Reshape'
                    new_node_idx = model.get_node_id(new_node_names[j][1])


                    if second_ret_old_node.attr['dst_shape'].split(',')[0] == '0':
                        attr = OrderedDict()
                        shape_0 = '-1'
                        shape_1 = second_ret_old_node.attr['dst_shape'].split(',')[1]
                        shape_2 = second_ret_old_node.attr['dst_shape'].split(',')[2]
                        attr['dst_shape'] = shape_0 + ',' + shape_1 + ',' + shape_2
                        attr['dims'] = 0
                        model.nodes[new_node_idx].attr = attr
                    else:
                        model.nodes[new_node_idx].attr = second_ret_old_node.attr                 


                    # the third new node
                    instanceNormalization_node_idx = model.get_node_id(new_node_names[j][2])
                    model.nodes[instanceNormalization_node_idx].op_type = 'LayerNorm'
                    attr_3 = OrderedDict()
                    attr_3['epsilon'] = 9.999999960041972e-13
                    model.nodes[instanceNormalization_node_idx].attr = attr_3

                    forth_ret_old_node = ret_old_nodes[j][4]
                    assert forth_ret_old_node.op_type == 'Reshape'
                    out_channel = first_ret_old_node.input_tensors[1].data.shape[0]
                    new_node_idx = model.get_node_id(new_node_names[j][3])
                    batch = model.nodes[new_node_idx].input_tensors[1].shape[0]
                    height = model.nodes[new_node_idx].input_tensors[1].shape[2]
                    weight = model.nodes[new_node_idx].input_tensors[1].shape[3]

                    _set_attr(batch, out_channel, height, weight, new_node_names[j], model)
                return model

        return model
