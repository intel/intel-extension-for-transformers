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

"""The CastTo Pattern."""

from .pattern import Pattern, pattern_registry
from .. import graph_utils as util
import copy


@pattern_registry(pattern_type='CastTo')
class CastTo(Pattern):
    """The CastTo pattern.

    Fuse the original sub-graph into the custom acceleration 'CastTo' graph.
    The search strategy is based on the following pattern mapping configs for different models.
    """

    def __call__(self, model):
        """The __call__ function of this pattern class."""
        pattern_mapping_config = {
            'SoftmaxCast': [
                # softmax + cast
                {
                    'patterns': {
                        'in': [[(0, 'Softmax'), (1, 'Cast')]],
                        'out':[[(0, 'Softmax')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 1
                    },
                    'input_tensors': {
                        0: [[{
                            0: [0]
                        }], [[0], 1]]
                    },
                    'output_tensors': {
                        0: [[{
                            1: [0]
                        }], [[0], 1]]
                    },
                    'returns': [0, 1]
                },
            ],
            'GreaterCastReduceSum': [
                # greater + cast + reduce_sum
                {
                    'patterns': {
                        'in': [[(0, 'Greater'), (1, 'Cast'), (2, 'ReduceSum')]],
                        'out':[[(0, 'Greater'), (1, 'ReduceSum')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 1,
                        1: 2
                    },
                    'input_tensors': {
                        0: [[{
                            0: [0]
                        }], [[0], 1]],
                        1: [[], [[], 1]]
                    },
                    'output_tensors': {
                        0: [[{
                            0: [0]
                        }], [[0], 1]],
                        1: [[{
                            2: [0]
                        }], [[0], 1]]
                    },
                    'returns': [0, 1, 2]
                },
            ],
            'RangeCastLess': [
                # range + cast + less
                {
                    'patterns': {
                        'in': [[(0, 'Cast'), (1, 'Range'), (2, 'Less')]],
                        'out':[[(0, 'Range'), (1, 'Less')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 1,
                        1: 2
                    },
                    'input_tensors': {
                        0: [[{
                            0: [0]
                        }], [[0], 1]],
                        1: [[{
                            2: [1]
                        }], [[1], 2]]
                    },
                    'output_tensors': {
                        0: [[], [[], 1]],
                        1: [[{
                            2: [0]
                        }], [[0], 1]]
                    },
                    'returns': [1, 0, 2]
                },
            ],
            'LessCastEqual': [
                # less + cast + equal
                {
                    'patterns': {
                        'in': [[(0, 'Less'), (1, 'Cast'), (2, 'Equal')]],
                        'out':[[(0, 'Less'), (1, 'Equal')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 1,
                        1: 2
                    },
                    'input_tensors': {
                        0: [[{
                            0: [0]
                        }, {
                            0: [1]
                        }], [[0, 1], 2]],
                        1: [[{
                            2: [1]
                        }], [[1], 2]]
                    },
                    'output_tensors': {
                        0: [[{
                            0: [0]
                        }], [[0], 1]],
                        1: [[{
                            2: [0]
                        }], [[0], 1]]
                    },
                    'returns': [0, 1, 2]
                },
            ],
            'BatchMatMulCast': [
                # batch_matmul + cast
                {
                    'patterns': {
                        'in': [[(0, 'BatchMatMul'), (1, 'Cast')]],
                        'out':[[(0, 'BatchMatMul')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 1
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
                            1: [0]
                        }], [[0], 1]]
                    },
                    'returns': [0, 1]
                },
            ],
            'MatMulWithBiasCast': [
                # matmul_with_bias + cast
                {
                    'patterns': {
                        'in': [[(0, 'MatMulWithBias'), (1, 'Cast')]],
                        'out':[[(0, 'MatMulWithBias')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 1
                    },
                    'input_tensors': {
                        0: [[{
                            0: [0]
                        }, {
                            0: [1]
                        }, {
                            0: [2]
                        }], [[0, 1, 2], 3]]
                    },
                    'output_tensors': {
                        0: [[{
                            1: [0]
                        }], [[0], 1]]
                    },
                    'returns': [0, 1]
                },
            ],
        }

        def _set_attr(attr_list, cast_dtype, node_names, model):
            assert len(attr_list) == len(node_names)
            for i in range(len(node_names)):
                node_idx = model.get_node_id(node_names[i])
                node_attr = attr_list[i]
                node_type = model.nodes[node_idx].op_type
                if i == 0:
                    for k, v in cast_dtype.items():
                        if k[:3] == "dst":
                            if v == "fp32":
                                node_attr['dtype'] = v
                            else:
                                if node_type in ["Greater", "Less"]:
                                    if v in ["int32", "int64"]:
                                        node_attr['dtype'] = "s8"
                        else:
                            if node_type in ["Range"]:
                                if v in ["int64"]:
                                    node_attr[k] = "int32"
                            else:
                                node_attr[k] = v
                model.nodes[node_idx].attr = copy.deepcopy(node_attr)

        _fusion_order = [('SoftmaxCast', 2),
                         ('BatchMatMulCast', 2),
                         ('MatMulWithBiasCast', 2),
                         ('GreaterCastReduceSum', 2),
                         ('RangeCastLess', 2),
                         ('LessCastEqual', 2),
                        ]
        for pattern_key, keep_attr_start in _fusion_order:
            pattern_dict = pattern_mapping_config[pattern_key][0]
            model, new_node_names, ret_old_nodes = \
                util.pattern_mapping(pattern_key, pattern_dict, model)
            if len(new_node_names) != 0:
                for j in range(len(new_node_names)):
                    node = ret_old_nodes[j][0]
                    node_attr = copy.deepcopy(node.attr)
                    tensor_info = {}
                    cast_dtype = {}
                    for idx, t in enumerate(node.input_tensors):
                        tensor_info[t.name] = 'src' + str(idx) + '_dtype'
                    tensor_info[node.output_tensors[0].name] = 'dst'
                    for n in ret_old_nodes[j][1:keep_attr_start]:
                        if n.output_tensors[0].name in tensor_info:
                            cast_dtype[tensor_info[n.output_tensors[0].name]] = n.attr['DstT']
                        else:
                            cast_dtype[tensor_info[n.input_tensors[0].name]] = n.attr['DstT']
                    attr_list = [node_attr]
                    attr_list.extend([n.attr for n in ret_old_nodes[j][keep_attr_start:]])
                    _set_attr(attr_list, cast_dtype, new_node_names[j], model)

        return model
