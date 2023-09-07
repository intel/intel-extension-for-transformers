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
"""The TextEncoder_SoftmaxReshape Pattern."""

from .pattern import Pattern, pattern_registry
from collections import OrderedDict
import copy
from .. import graph_utils as util
from .. import logger


@pattern_registry(pattern_type='TextEncoder_SoftmaxReshape')
class TextEncoder_SoftmaxReshape(Pattern):
    """The TextEncoder_SoftmaxReshape pattern.

    Fuse the original sub-graph into the custom acceleration 'TextEncoder_SoftmaxReshape' graph.
    The search strategy is based on the following pattern mapping configs for the stable diffusionV1-5.
    """
    def __call__(self, model):
        """The __call__ function of this pattern class."""
        pattern_mapping_config = {
            # for text encoder self_attn/out_proj/Add Reshape, Reshape 7
            'TextEncoder_SoftmaxReshape': [
                {
                    'patterns': {
                        'in': [[(0, 'Reshape'), (1, 'Shape'), (2, 'Gather'), (3, 'Unsqueeze'), (6, 'Concat'),
                                (7, 'Reshape'), (8, 'Softmax')], [(), (4, 'Unsqueeze'), (6, 'Concat')],
                               [(), (5, 'Unsqueeze'), (6, 'Concat')]],
                        'out': [[(0, 'Reshape'), (1, 'Reshape'), (2, 'Softmax')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 0,
                        1: 7,
                        2: 8
                    },
                    'input_tensors': {
                        0: [[{
                            0: [0]
                        }, {
                            0: [1]
                        }], [[0, 1], 2]],
                        1: [[{
                            7: [0]
                        }], [[0], 2]],
                        2: [[], [[], 1]],
                    },
                    'output_tensors': {
                        0: [[{
                            0: [0]
                        }], [[0], 1]],
                        1: [[{
                            7: [0]
                        }], [[0], 1]],
                        2: [[{
                            8: [0]
                        }], [[0], 1]]
                    },
                    'returns': [0, 6, 8]
                },
            ]
        }

        def _set_attr(node_names, model):
            attr = OrderedDict()
            reshape_node = model.get_node_by_name('position_embeddings/after/reshape')
            # 12 or 16
            mul_size = int(reshape_node.input_tensors[0].shape[1] / 64)
            # mul_size + max_seq + seq 12,77,77
            attr['dst_shape'] = str(mul_size) + ',-1,-1'
            attr['dims'] = '1,1'
            reshape_node_idx = model.get_node_id(node_names[1])
            model.nodes[reshape_node_idx].attr = attr

        for i in range(len(pattern_mapping_config['TextEncoder_SoftmaxReshape'])):
            pattern_dict = pattern_mapping_config['TextEncoder_SoftmaxReshape'][i]
            model, new_node_names, ret_old_nodes = util.pattern_mapping("TextEncoder_SoftmaxReshape",
                                                                        pattern_dict, model)

            logger.info('TextEncoder_SoftmaxReshape mathched...')
            logger.debug('TextEncoder_SoftmaxReshape = {}'.format(new_node_names))
            if len(new_node_names) != 0:
                for j in range(len(new_node_names)):
                    first_ret_old_node = ret_old_nodes[j][0]
                    assert first_ret_old_node.op_type == 'Reshape'
                    reshape_new_node_idx = model.get_node_id(new_node_names[j][0])
                    model.nodes[reshape_new_node_idx].attr = first_ret_old_node.attr
                    _set_attr(new_node_names[j], model)

                    transpose_node_name = model.nodes[reshape_new_node_idx].output_tensors[0].dest_op[1]
                    transpose_node_idx = model.get_node_id(transpose_node_name)

                    # delete the node before insert
                    copy_node = copy.deepcopy(model.nodes[reshape_new_node_idx])
                    model.remove_nodes([new_node_names[j][0]])
                    model.insert_nodes(transpose_node_idx, [copy_node])

        return model
