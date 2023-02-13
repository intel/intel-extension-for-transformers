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

"""The MatMulWithBiasAdd Pattern."""

from .pattern import Pattern, pattern_registry
from collections import namedtuple, OrderedDict
from .. import graph_utils as util
from ..onnx_utils import bias_to_int32
import copy


@pattern_registry(pattern_type='MatMulWithBiasAdd')
class MatMulWithBiasAdd(Pattern):
    """The MatMulWithBiasAdd pattern.

    Fuse the original sub-graph into the custom acceleration 'MatMulWithBiasAdd' graph.
    The search strategy is based on the following pattern mapping configs for different models.
    """
    def __call__(self, model):
        """The __call__ function of this pattern class."""
        pattern_mapping_config = {
            'MatMulWithBiasAdd': [
                {
                    'patterns': {
                        'in': [[(0, 'MatMulWithBias'), (1, ['Add', 'AddV2'])]],
                        'out': [[(0, 'MatMulWithBiasAdd')]]
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
                        }, {
                            1: [0, 1]
                        }], [[0, 1, 2, 3], 4]]
                    },
                    'output_tensors': {
                        0: [[{
                            1: [0]
                        }], [[0], 1]]
                    },
                    'returns': [0]
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
                attr['append_op'] = 'sum'
                model.nodes[mat_node_idx].attr = attr

        pattern_dict = pattern_mapping_config['MatMulWithBiasAdd'][0]
        model, new_node_names, ret_old_nodes = util.pattern_mapping("MatMulWithBiasAdd",
                                                                    pattern_dict, model)
        if len(new_node_names) != 0:
            _set_attr(new_node_names, ret_old_nodes, model)

            return model

        return model
