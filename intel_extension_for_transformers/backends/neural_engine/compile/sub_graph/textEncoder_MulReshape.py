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
"""The TextEncoder_MulReshape Pattern."""

from .pattern import Pattern, pattern_registry
from collections import OrderedDict
from .. import graph_utils as util
from .. import logger


@pattern_registry(pattern_type='TextEncoder_MulReshape')
class TextEncoder_MulReshape(Pattern):
    """The TextEncoder_MulReshape pattern.

    Fuse the original sub-graph into the custom acceleration 'TextEncoder_MulReshape' graph.
    This pattern is used for the reshape node after the QKV.
    """
    def __call__(self, model):
        """The __call__ function of this pattern class."""
        pattern_mapping_config = {
            'TextEncoder_MulReshape': [
                # text encoder Reshape_3/4/5/7
                {
                    'patterns': {
                        'in': [[(0, 'Mul'), (1, 'Unsqueeze'), (2, 'Concat'), (3, 'Reshape')]],
                        'out': [[(0, 'Reshape')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 3,
                    },
                    'input_tensors': {
                        0: [[{
                            3: [0]
                        }], [[0], 1]],
                    },
                    'output_tensors': {
                        0: [[{
                            3: [0]
                        }], [[0], 1]],
                    },
                    'returns': [2, 0]
                },
            ]
        }

        def _set_attr(head_num, head_size, mul_size, node_names, model):
            attr = OrderedDict()
            attr['dst_shape'] = str(mul_size) + ',' + str(head_num) + ',' + str(head_size)

            reshape_node_idx = model.get_node_id(node_names[0])
            assert model.nodes[reshape_node_idx].op_type == 'Reshape'
            model.nodes[reshape_node_idx].attr = attr

        for i in range(len(pattern_mapping_config['TextEncoder_MulReshape'])):
            pattern_dict = pattern_mapping_config['TextEncoder_MulReshape'][i]
            model, new_node_names, ret_old_nodes = util.pattern_mapping("TextEncoder_MulReshape",
                                                                        pattern_dict, model)

            if len(new_node_names) != 0:
                logger.info('TextEncoder_MulReshape mathched...')
                logger.debug('TextEncoder_MulReshape = {}'.format(new_node_names))
                # except the reshape_7
                for j in range(len(new_node_names)):
                    pack_node = ret_old_nodes[j][0]
                    head_size = int(pack_node.input_tensors[-1].data)
                    head_num = int(pack_node.input_tensors[-2].data)
                    assert ret_old_nodes[j][1].op_type == 'Mul'
                    mul_size = int(ret_old_nodes[j][1].input_tensors[1].data)
                    _set_attr(head_num, head_size, mul_size, new_node_names[j], model)
                return model

        return model
