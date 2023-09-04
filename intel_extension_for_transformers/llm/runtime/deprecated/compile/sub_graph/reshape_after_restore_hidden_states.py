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

"""The ReshapeAfterRestoreHiddenStates Pattern."""

from .pattern import Pattern, pattern_registry
from collections import namedtuple, OrderedDict
from .. import graph_utils as util
import numpy as np
import copy


@pattern_registry(pattern_type='ReshapeAfterRestoreHiddenStates')
class ReshapeAfterRestoreHiddenStates(Pattern):
    """The ReshapeAfterRestoreHiddenStates pattern.

    Fuse the original sub-graph into the custom acceleration 'ReshapeAfterRestoreHiddenStates' graph.
    The search strategy is based on the following pattern mapping configs for different models.
    """
    def __call__(self, model):
        """The __call__ function of this pattern class."""
        pattern_mapping_config = {
            'ReshapeAfterRestoreHiddenStates': [
                # minilmv2-lat-roberta
                # reshape to 3d when do last scatter
                # and becomes to 2d when do inner product
               {
                    'patterns': {
                        'in': [[(0, 'ScatterElements'), (1, 'MatMulWithBias')]],
                        'out': [[(0, 'ScatterElements'), (1, 'Reshape'), (2, 'MatMulWithBias')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 0,
                        1: 'reshape_to_2d_before_inner_product_in_last_restoration',
                        2: 1, 
                    },
                    'input_tensors': {
                        0: [[{0: [0]}, {0: [1]}, {0: [2]}], [[0, 1, 2], 3]],
                        1: [[], [[], 1]],
                        2: [[{1: [1]}, {1: [2]}], [[1, 2], 3]],
                    },
                    'output_tensors': {
                        0: [[], [[], 1]],
                        1: [[], [[], 1]],
                        2: [[{1: [0]}], [[0], 1]],
                    },
                    'returns': [0, 1]
                },
            ]
        }

        def _set_attr(se_attr, mat_attr, hidden_size, node_names, model):
            attr1 = OrderedDict()
            attr1['dst_shape'] = '-1,' + str(hidden_size)
            
            scatter_elements_node_idx = model.get_node_id(node_names[0])
            model.nodes[scatter_elements_node_idx].attr = se_attr
            reshape_2d_node_idx = model.get_node_id(node_names[1])
            model.nodes[reshape_2d_node_idx].attr = attr1
            mat_node_idx = model.get_node_id(node_names[2])
            model.nodes[mat_node_idx].attr = mat_attr
        
        # minilmv2-lat-roberta
        pattern_dict = pattern_mapping_config['ReshapeAfterRestoreHiddenStates'][0]
        model, new_node_names, ret_old_nodes = util.pattern_mapping(
                'ReshapeAfterRestoreHiddenStates', pattern_dict, model)
        if len(new_node_names) != 0:
            for i in range(len(new_node_names)):
                hidden_size = int(ret_old_nodes[i][1].input_tensors[1].shape[0])
                se_attr = ret_old_nodes[i][0].attr
                mat_attr = ret_old_nodes[i][1].attr
                mat_node = model.get_node_by_name(new_node_names[i][2])
                reshape_node = model.get_node_by_name(new_node_names[i][1])
                mat_node.input_tensors[0].name = ret_old_nodes[i][1].input_tensors[0].name
                reshape_node.output_tensors[0].name =  mat_node.input_tensors[0].name
                _set_attr(se_attr, mat_attr, hidden_size, new_node_names[i], model)
            
            return model
        
        return model