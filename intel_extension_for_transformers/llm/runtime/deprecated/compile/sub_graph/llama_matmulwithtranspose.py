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

"""The LlamaMatMulWithTranspose pattern."""

from .pattern import Pattern, pattern_registry
from collections import namedtuple, OrderedDict
from .. import graph_utils as util
from ..onnx_utils import bias_to_int32
import copy


@pattern_registry(pattern_type='LlamaMatMulWithTranspose')
class LlamaMatMulWithTranspose(Pattern):
    """The LlamaMatMulWithTranspose pattern.

    Fuse the original sub-graph into the custom acceleration 'LlamaMatMulWithTranspose' graph.
    The search strategy is based on the following pattern mapping configs for different models.
    """
    def __call__(self, model):
        """The __call__ function of this pattern class."""
        pattern_mapping_config = {
            'LlamaMatMulWithTranspose': [
                # llama 
                {
                    'patterns': {
                        'in': [[(0, 'Reorder'), (1, 'Matmul')]],
                        'out': [[(0, 'Matmul')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 1,
                    },
                    'input_tensors': {
                        0: [[{
                            1: [0]
                        }, {
                            0: [0]
                        }], [[0, 1], 2]]
                    },
                    'output_tensors': {
                         0: [[{
                            1: [0]
                        }], [[0], 1]]
                    },
                    'returns': [0]
                },
                {
                    'patterns': {
                        'in': [[(0, 'Matmul'), (1, 'Reorder'), (2, 'Reshape')]],
                        'out': [[(0, 'Matmul')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 0,
                    },
                    'input_tensors': {
                        0: [[{
                            0: [0]
                        }, {
                            0: [1]
                        }], [[0, 1], 2]]
                    },
                    'output_tensors': {
                         0: [[{
                            2: [0]
                        }], [[0], 1]]
                    },
                    'returns': [1]
                },
                
                {
                    'patterns': {
                        'in': [[(0, 'Add'), (1, 'Max'), (2, 'Softmax')]],
                        'out': [[(0, 'Add'), (1, 'Softmax')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 0,
                        1: 2
                    },
                    'input_tensors': {
                        0: [[{
                            0: [0]
                        }, {
                            0: [1]
                        }], [[0, 1], 2]],
                        1: [[], [[], 1]]
                    },
                    'output_tensors': {
                         0: [[], [[], 1]],
                         1: [[{
                            2: [0]
                        }], [[0], 1]]
                    },
                    'returns': [1]
                },
            ]
        }

        pattern_dict = pattern_mapping_config['LlamaMatMulWithTranspose'][0]
        model, new_node_names, ret_old_nodes = util.pattern_mapping("LlamaMatMulWithTranspose", 
                                                                    pattern_dict, model)
        if len(new_node_names) != 0:
            for i in range(len(new_node_names)):
                mat_node_idx = model.get_node_by_name(new_node_names[i][0])
                attr = OrderedDict()
                attr['src1_perm'] = '0, 1, 3, 2'
                mat_node_idx.attr = attr

        pattern_dict = pattern_mapping_config['LlamaMatMulWithTranspose'][1]
        model, new_node_names, ret_old_nodes = util.pattern_mapping("LlamaMatMulWithTranspose", 
                                                                    pattern_dict, model)
        if len(new_node_names) != 0:
            for i in range(len(new_node_names)):
                mat_node_idx = model.get_node_by_name(new_node_names[i][0])
                attr = OrderedDict()
                attr['dst_perm'] = "0, 2, 1, 3"
                attr['reshape'] = "-1, 4096"
                mat_node_idx.attr = attr

        pattern_dict = pattern_mapping_config['LlamaMatMulWithTranspose'][2]
        model, new_node_names, ret_old_nodes = util.pattern_mapping("LlamaMatMulWithTranspose", 
                                                                    pattern_dict, model)

        return model
