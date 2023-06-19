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

"""The TorchUnpackBaddbmm pattern."""

from .pattern import Pattern, pattern_registry
from collections import namedtuple, OrderedDict
import copy
from .. import graph_utils as util
from ..ops.tensor import Tensor
import numpy as np

@pattern_registry(pattern_type='TorchUnpackBaddbmm')
class TorchUnpackBaddbmm(Pattern):
    """The TorchUnpackBaddbmm pattern.

    Fuse the original sub-graph into the custom acceleration 'TorchUnpackBaddbmm' graph.
    The search strategy is based on the following pattern mapping configs for different models.
    """
    def __call__(self, model):
        """The __call__ function of this pattern class."""
        pattern_mapping_config = {
            'TorchUnpackBaddbmm': [
                {
                    'patterns': {
                        'in': [[(0,'Baddbmm')]],
                        'out': [[(0, 'Matmul'), (1, 'BinaryOp'), (2, 'BinaryAdd')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 0,
                        1: 'Baddbmm_div',
                        2: 'Baddbmm_add'
                    },
                    'input_tensors': {
                        0: [[{
                            0: [1]
                        }, {
                            0: [2]
                        }], [[1, 2], 2]],
                        1: [[], [[], 1]],#input tensor is from the last node
                        2: [[{
                            0: [0]
                        }], [[0], 2]],

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

        def _set_attr(new_node_names, ret_old_nodes, model):
            for i in range(len(new_node_names)):
                binary_node = model.get_node_by_name(new_node_names[i][1])
                new_attr = OrderedDict({'algorithm': 'div'})
                binary_node.attr = new_attr
                mm_other_tensor = Tensor(name=ret_old_nodes[i][0].name + '_mm_other',
                         source_op=[],
                         dest_op=[binary_node.name],
                         shape=[1],
                         data=np.array(1./ret_old_nodes[i][0].attr['alpha']).astype(np.float32),
                         dtype="fp32")
                binary_node.input_tensors.append(mm_other_tensor)
        pattern_dict = pattern_mapping_config['TorchUnpackBaddbmm'][0]
        model, new_node_names, ret_old_nodes = \
            util.pattern_mapping('TorchUnpackBaddbmm', pattern_dict, model)
        _set_attr(new_node_names, ret_old_nodes, model)
        return model