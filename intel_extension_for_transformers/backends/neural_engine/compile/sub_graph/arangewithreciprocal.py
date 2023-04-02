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

"""The ArangewithReciprocal Pattern."""

from .pattern import Pattern, pattern_registry
from collections import namedtuple, OrderedDict
from .. import graph_utils as util
from ..ops import Tensor
import copy
import numpy as np

@pattern_registry(pattern_type='ArangewithReciprocal')
class ArangewithReciprocal(Pattern):
    """The ArangewithReciprocal pattern.

    Fuse the original sub-graph into the custom acceleration 'ArangewithReciprocal' graph.
    The search strategy is based on the following pattern mapping configs for different models.
    """
    def __call__(self, model):
        """The __call__ function of this pattern class."""
        pattern_mapping_config = {
            'ArangewithReciprocal': [
                {
                    'patterns': {
                        'in': [[(0, 'Shape'), (1, 'Arange'), (2, 'Div'), (3, 'Pow'), 
                                (4, 'Reciprocal'), (5, 'Mul')]
                                ],
                        'out': [[(0, 'Range'), (1, 'Div'), (2, 'Pow'), 
                                (3, 'Div')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 1,
                        1: 2,
                        2: 3,
                        3: 5
                        
                    },
                    'input_tensors': {
                        0: [[{
                            'input_data': [0]
                        }
                        ], [[0], 1]],
                        1: [[], [[], 1]],
                        2: [[{
                            3: [0]
                        }], [[0], 2]],
                        3: [[{
                            5: [1]
                        }
                        ], [[0], 2]],
                    },
                    'output_tensors': {
                        0: [[], [[], 1]],
                        1: [[], [[], 1]],
                        2: [[], [[], 1]],
                        3: [[{
                            5: [0]
                        }], [[0], 1]]
                    },
                    'returns': [0]
                },
                

                
            ]
        }


        def _set_attr(new_node_names, ret_old_nodes, model):
            for i in range(len(new_node_names)):
                
                slice_node = model.get_node_by_name(ret_old_nodes[i][0].input_tensors[0].source_op[0])
                fixed_pos_embedding_dim = int(slice_node.attr['ends'])
                range_node_idx = model.get_node_id(new_node_names[i][0])
                attr = OrderedDict()
                attr['end'] = fixed_pos_embedding_dim
                attr['step'] = 2
                model.nodes[range_node_idx].attr = attr
                div_dim_node = model.get_node_by_name(new_node_names[i][1])
                div_dim = np.array([fixed_pos_embedding_dim], dtype=np.float32)
                div_dim_tensor = Tensor(name=div_dim_node.input_tensors[0].name + "_dim",
                                        source_op=[],
                                        dest_op=[div_dim_node.name],
                                        data = div_dim,
                                        shape = [1, 1],
                                        dtype="fp32")
                div_dim_node.input_tensors.append(div_dim_tensor)
                div_dim_node.attr = OrderedDict({'algorithm': 'div'})
                model.get_node_by_name(new_node_names[i][3]).attr = \
                                     OrderedDict({'algorithm': 'div'})
                reciprocal_node = model.get_node_by_name(new_node_names[i][3])
                reciprocal_node.input_tensors[0].data = np.array([1], dtype=np.float32)
                
        if model.framework_modeling_config['framework'] == 'torch':
            pattern_dict = pattern_mapping_config['ArangewithReciprocal'][0]
            model, new_node_names, ret_old_nodes = util.pattern_mapping("ArangewithReciprocal", 
                                                                        pattern_dict, model)
            if len(new_node_names) != 0:
                _set_attr(new_node_names, ret_old_nodes, model)
                return model

        return model
