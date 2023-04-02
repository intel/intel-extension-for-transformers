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

"""The ReshapeBeforeRestoreHiddenStates Pattern."""

from .pattern import Pattern, pattern_registry
from collections import namedtuple, OrderedDict
from .. import graph_utils as util
import numpy as np
import copy


@pattern_registry(pattern_type='ReshapeBeforeRestoreHiddenStates')
class ReshapeBeforeRestoreHiddenStates(Pattern):
    """The ReshapeBeforeRestoreHiddenStates pattern.

    Fuse the original sub-graph into the custom acceleration 'ReshapeBeforeRestoreHiddenStates' graph.
    The search strategy is based on the following pattern mapping configs for different models.
    """
    def __call__(self, model):
        """The __call__ function of this pattern class."""
        pattern_mapping_config = {
            'ReshapeBeforeRestoreHiddenStates': [
                # minilmv2-lat-roberta
                # reshape to 3d when do scatter
               {
                    'patterns': {
                        'in': [[(0, 'LayerNorm'), (1, 'ScatterElements')]],
                        'out': [[(0, 'LayerNorm'), (1, 'Reshape'), (2, 'ScatterElements')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 0,
                        1: 'reshape_to_3d_after_layer_norm_in_restoration',
                        2: 1,
                    },
                    'input_tensors': {
                        0: [[{0: [0]}, {0: [1]}, {0: [2]}], [[0, 1, 2], 3]],
                        1: [[{'input_data': [0]}], [[1], 2]],
                        2: [[{1: [0]}, {1: [1]}], [[0, 1], 3]],
                    },
                    'output_tensors': {
                        0: [[{0: [0]}], [[0], 1]],
                        1: [[], [[], 1]],
                        2: [[{1: [0]}], [[0], 1]],
                    },
                    'returns': [0, 1]
                },
            ]
        }

        def _set_attr(ln_attr, se_attr, hidden_size, node_names, model):
            attr1 = OrderedDict()
            attr1['dst_shape'] = '-1,-1,' + str(hidden_size)
            attr1['dims'] = 0
            
            ln_node_idx = model.get_node_id(node_names[0])
            model.nodes[ln_node_idx].attr = ln_attr
            reshape_3d_node_idx = model.get_node_id(node_names[1])
            model.nodes[reshape_3d_node_idx].attr = attr1
            scatter_elements_node_idx = model.get_node_id(node_names[2])
            model.nodes[scatter_elements_node_idx].attr = se_attr
        
        # minilmv2-lat-roberta
        layer_norm_idx = []
        remove_list = []
        pattern = pattern_mapping_config['ReshapeBeforeRestoreHiddenStates'][0]['patterns']['in']
        patterns_nodes_name = util.search_pattern(pattern, model)
        for pattern_nodes_name in patterns_nodes_name:
            layer_norm_idx.append(model.get_node_id(pattern_nodes_name[0]))
        pattern_dict = pattern_mapping_config['ReshapeBeforeRestoreHiddenStates'][0]
        model, new_node_names, ret_old_nodes = util.pattern_mapping(
                'ReshapeBeforeRestoreHiddenStates', pattern_dict, model)
        if len(new_node_names) != 0:
            for i in range(len(new_node_names)):
                hidden_size = int(ret_old_nodes[i][0].input_tensors[-1].shape[0])
                ln_attr = ret_old_nodes[i][0].attr
                se_attr = ret_old_nodes[i][1].attr
                _set_attr(ln_attr, se_attr, hidden_size, new_node_names[i], model)
                import copy
                ln_node = copy.deepcopy(model.get_node_by_name(new_node_names[i][0]))
                model.remove_nodes([new_node_names[i][0]])
                model.insert_nodes(layer_norm_idx[i] + i, [ln_node])
                
                remove_list.append(new_node_names[i][0])
                
            
            # model.remove_nodes(remove_list)
            return model
        
        return model