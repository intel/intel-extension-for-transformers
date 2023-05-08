#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2022 Intel Corporation
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
"""The ExplicitNHWCTransposeForConv Pattern."""

from .pattern import Pattern, pattern_registry
from collections import namedtuple, OrderedDict
from .. import graph_utils as util
from .. import logger


@pattern_registry(pattern_type='ExplicitNHWCTransposeForConv')
class ExplicitNHWCTransposeForConv(Pattern):
    """The ExplicitNHWCTransposeForConv pattern.

    Fuse the original sub-graph into the custom acceleration 'ExplicitNHWCTransposeForConv' graph.
    The search strategy is based on the following pattern mapping configs for different models.
    """

    def __call__(self, model):
        """The __call__ function of this pattern class."""
        pattern_mapping_config = {
            'ExplicitNHWCTransposeForConv': [
                {
                    'patterns': {
                        'in': [[(0, 'Conv')]],
                        'out': [[(0, 'Transpose'), (1, 'Conv'), (2, 'Transpose')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 'reorder_pre_for_conv',
                        1: 0,
                        2: 'reorder_post_for_conv',
                    },
                    'input_tensors': {
                        0: [[{
                            0: [0]
                        }], [[0], 1]],
                        1: [[{
                            0: [1]
                        }, {
                            0: [2]
                        }], [[1, 2], 3]],
                        2: [[], [[], 1]]
                    },
                    'output_tensors': {
                        0: [[], [[], 1]],
                        1: [[], [[], 1]],
                        2: [[{
                            0: [0]
                        }], [[0], 1]]
                    },
                    'returns': [0]
                },
            ]
        }

        pattern_dict = pattern_mapping_config['ExplicitNHWCTransposeForConv'][0]
        model, new_node_names, ret_old_nodes = util.pattern_mapping("ExplicitNHWCTransposeForConv", pattern_dict, model)
        if len(new_node_names) != 0:
            logger.info('ExplicitNHWCTransposeForConv mathched...')
            logger.debug('ExplicitNHWCTransposeForConv = {}'.format(new_node_names))
            for i in range(len(new_node_names)):
                conv_attr = OrderedDict()
                conv_node_idx = model.get_node_id(new_node_names[i][1])
                conv_attr = ret_old_nodes[i][0].attr
                conv_node = model.nodes[conv_node_idx]
                conv_node.attr = conv_attr
                conv_node.attr['src_perm'] = '0,3,1,2'
                conv_node.attr['dst_perm'] = '0,2,3,1'

                # the first node
                attr = OrderedDict()
                reorder_pre_node = model.get_node_by_name(new_node_names[i][0])
                attr['src_perm'] = '0,1,2,3'
                attr['dst_perm'] = '0,2,3,1'
                reorder_pre_node.attr = attr

                # the thrid node
                attr_2 = OrderedDict()
                reorder_post_node = model.get_node_by_name(new_node_names[i][2])
                attr_2['src_perm'] = '0,1,2,3'
                attr_2['dst_perm'] = '0,3,1,2'
                reorder_post_node.attr = attr_2

                if 'output_dtype' in conv_attr:
                    if 'fp32' in conv_attr['output_dtype']:
                        reorder_pre_node.attr['output_dtype'] = 'bf16'
                        conv_node.attr['output_dtype'] = 'bf16'
                        reorder_post_node.attr['output_dtype'] = 'fp32'
                    else:
                        reorder_post_node.attr['output_dtype'] = conv_attr['output_dtype']
                        reorder_pre_node.attr['output_dtype'] = conv_attr['output_dtype']

        if len(new_node_names) != 0:
            remove_node_name = []
            for transpose_node in model.nodes:
                if transpose_node.op_type == 'Transpose':
                    node_id = model.get_node_id(transpose_node.name)
                    second_transpose_node = model.nodes[node_id + 1]
                    if second_transpose_node.op_type == 'Transpose' \
                        and second_transpose_node.input_tensors[0].name == transpose_node.output_tensors[0].name:
                        if transpose_node.attr['dst_perm'] == '0,3,1,2' \
                            and second_transpose_node.attr['dst_perm'] == '0,2,3,1':
                            remove_node_name.append(transpose_node.name)
                            remove_node_name.append(second_transpose_node.name)

                            pre_node = model.nodes[node_id - 1]
                            target_node = model.nodes[node_id + 2]
                            pre_node.output_tensors[0] = transpose_node.output_tensors[0]
                            target_node.input_tensors[0] = transpose_node.output_tensors[0]

            model.remove_nodes(remove_node_name)

        return model
