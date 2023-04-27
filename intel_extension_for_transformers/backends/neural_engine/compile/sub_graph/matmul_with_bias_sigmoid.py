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
"""The MatMulWithBiasSigmoid pattern."""

from .pattern import Pattern, pattern_registry
from collections import OrderedDict
from .. import graph_utils as util
from .. import logger
from .subgraph_matcher import EXECUTOR_TYPE


@pattern_registry(pattern_type='MatMulWithBiasSigmoid')
class MatMulWithBiasSigmoid(Pattern):
    """The MatMulWithBiasSigmoid pattern.

    Fuse the original sub-graph into the custom acceleration 'MatMulWithBiasSigmoid' graph.
    The search strategy is based on the following pattern mapping configs for different models.
    """
    def __call__(self, model):
        """The __call__ function of this pattern class."""
        pattern_mapping_config = {
            'MatMulWithBiasSigmoid': [
                # unet inner_product + swish
                {
                    'patterns': {
                        'in': [[(0, 'MatMulWithBias'), (1, ['Sigmoid']), (2, 'Mul')]],
                        'out': [[(0, 'MatMulWithBiasSwish')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 2
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
                            2: [0]
                        }], [[0], 1]]
                    },
                    'returns': [0, 1, 2]
                },
                # inner_product + sigmoid
                {
                    'patterns': {
                        'in': [[(0, 'MatMulWithBias'), (1, ['Sigmoid'])]],
                        'out': [[(0, 'MatMulWithBiasSigmoid')]]
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
            ]
        }

        def _set_attr(new_node_names, ret_old_nodes, model):
            for i in range(len(new_node_names)):
                mat_node_idx = model.get_node_id(new_node_names[i][0])
                attr = OrderedDict()
                if 'src0_perm' in ret_old_nodes[i][0].attr.keys():
                    attr['src0_perm'] = ret_old_nodes[i][0].attr['src0_perm']
                if 'src1_perm' in ret_old_nodes[i][0].attr.keys():
                    attr['src1_perm'] = ret_old_nodes[i][0].attr['src1_perm']
                op_t_list = [n.op_type for n in ret_old_nodes[i]]
                op_t_list[0] = EXECUTOR_TYPE.get(op_t_list[0], op_t_list[0])
                if op_t_list == ['InnerProduct', 'Sigmoid']:
                    attr['append_op'] = 'sigmoid'
                if op_t_list == ['InnerProduct', 'Sigmoid', 'Mul']:
                    attr['append_op'] = 'swish'
                model.nodes[mat_node_idx].attr = attr

        for i in range(len(pattern_mapping_config['MatMulWithBiasSigmoid'])):
            pattern_dict = pattern_mapping_config['MatMulWithBiasSigmoid'][i]
            model, new_node_names, ret_old_nodes = util.pattern_mapping("MatMulWithBiasSigmoid",
                                                                        pattern_dict, model)
            if len(new_node_names) != 0:
                _set_attr(new_node_names, ret_old_nodes, model)

        return model
