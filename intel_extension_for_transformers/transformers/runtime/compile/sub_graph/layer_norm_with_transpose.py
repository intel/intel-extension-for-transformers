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

"""The LayerNormWithTranspose Pattern."""

from .pattern import Pattern, pattern_registry
from collections import namedtuple, OrderedDict
from .. import graph_utils as util
import copy


@pattern_registry(pattern_type='LayerNormWithTranspose')
class LayerNormWithTranspose(Pattern):
    """The LayerNormWithTranspose pattern.

    Fuse the original sub-graph into the custom acceleration 'LayerNormWithTranspose' graph.
    The search strategy is based on the following pattern mapping configs for different models.
    """
    def __call__(self, model):
        """The __call__ function of this pattern class."""
        pattern_mapping_config = {
            'LayerNormWithTranspose': [
                {
                    'patterns': {
                        'in': [[(0, 'MatMulWithBiasAdd'), (1, 'LayerNorm'), (2, 'Transpose')]],
                        'out': [[(0, 'MatMulWithBiasAdd'), (1, 'LayerNorm'), (2, 'Reshape'),
                                 (3, 'Transpose')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 0,
                        1: 1,
                        2: 'reshape_3d_before_transpose',
                        3: 2,
                    },
                    'input_tensors': {
                        0: [[{
                            0: [0]
                        }, {
                            0: [1]
                        }, {
                            0: [2]
                        }, {
                            0: [3]
                        }], [[0, 1, 2, 3], 4]],
                        1: [[{
                            1: [1]
                        }, {
                            1: [2]
                        }], [[1, 2], 3]],
                        2: [[{
                            'input_data': [0]
                        }], [[1], 2]],
                        3: [[], [[], 1]]
                    },
                    'output_tensors': {
                        0: [[], [[], 1]],
                        1: [[], [[], 1]],
                        2: [[], [[], 1]],
                        3: [[{
                            2: [0]
                        }], [[0], 1]]
                    },
                    'returns': [0, 1, 2]
                },
            ]
        }

        def _set_attr(old_nodes, node_names, model):
            reshape_attr = OrderedDict()
            hidden_size = str(model.inquire_config_item("hidden_size"))
            reshape_attr['dst_shape'] = '-1,-1,' + hidden_size
            reshape_attr['dims'] = '1,0'

            reshape_idx = 0
            idx = 0
            for n in old_nodes:
                if model.get_node_by_name(node_names[idx]).op_type == "Reshape":
                    reshape_idx = idx
                    idx += 1
                model.get_node_by_name(new_node_names[j][idx]).attr = copy.deepcopy(n.attr)
                idx += 1
            model.get_node_by_name(new_node_names[j][reshape_idx]).attr = reshape_attr

        pattern_dict = pattern_mapping_config['LayerNormWithTranspose'][0]
        model, new_node_names, ret_old_nodes = util.pattern_mapping("LayerNormWithTranspose",
                                                                    pattern_dict, model)
        if len(new_node_names) != 0:
            for j in range(len(new_node_names)):
                _set_attr(ret_old_nodes[j], new_node_names[j], model)

            return model

        return model
