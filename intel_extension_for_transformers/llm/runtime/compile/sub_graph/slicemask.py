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

"""The SliceMask Pattern."""

from .pattern import Pattern, pattern_registry
from collections import namedtuple, OrderedDict
from .. import graph_utils as util
import copy


@pattern_registry(pattern_type='SliceMask')
class SliceMask(Pattern):
    """The SliceMask pattern.

    Fuse the original sub-graph into the custom acceleration 'SliceMask' graph.
    The search strategy is based on the following pattern mapping configs for different models.
    """
    def __call__(self, model):
        """The __call__ function of this pattern class."""
        pattern_mapping_config = {
            'SliceMask': [
                {
                    'patterns': {
                        'in': [[(0, 'Shape'), (2, 'Sub'), (3, 'Int'), (6, 'Slice'), 
                                (7, 'Slice'), (8, 'Where'), (9, 'Add')],
                               [(), (1, 'Shape'), (2, 'Sub')],
                               [(), (4, 'Slice'), (5, 'Slice'), (6, 'Slice')]
                                ],
                        'out': [[(0, 'Slice'), (1, 'Slice'), (2, 'BinaryAdd')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 4,
                        1: 6,
                        2: 9
                    },
                    'input_tensors': {
                        0: [[{
                            4: [0]
                        }], [[0], 1]],
                        1: [[], [[], 1]],
                        2: [[{
                            8: [1]
                        }, {
                            9: [1]
                        }
                        ], [[0, 2], 3]]
                    },
                    'output_tensors': {
                        0: [[], [[], 1]],
                        1: [[], [[], 1]],
                        2: [[{
                            9: [0]
                        }], [[0], 1]]
                    },
                    'returns': []
                },
                
                {
                    'patterns': {
                        'in': [[(0, 'Shape'), (2, 'Sub'), (3, 'Slice'), (4, 'Slice'), 
                                (5, 'Where')],
                               [(), (1, 'Shape'), (2, 'Sub')],
                                ],
                        'out': [[(0, 'SliceMask'), (1, 'BinaryAdd')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 4,
                        1: 5,
                    },
                    'input_tensors': {
                        0: [[{
                            3: [0]
                        },{
                            'input_data': [0]
                        }, {
                            'input_data': [2]
                        }
                        ], [[0, 1, 2], 3]],
                        1: [[{
                            5: [1]
                        }], [[0], 2]]
                    },
                    'output_tensors': {
                        0: [[], [[], 1]],
                        1: [[{
                            5: [0]
                        }], [[0], 1]]
                    },
                    'returns': []
                },
            ]
        }

        if model.framework_modeling_config['framework'] != 'torch':
            return model

        def _set_attr(new_node_names, ret_old_nodes, model):
            for i in range(len(new_node_names)):
                binary_node_idx = model.get_node_id(new_node_names[i][2])
                attr = OrderedDict()
                model.nodes[binary_node_idx].attr = attr
                slice_node = model.get_node_by_name(new_node_names[i][0])
                import numpy as np
                
                slice_node.input_tensors[0].data = np.array(slice_node.input_tensors[0].data, dtype=np.float32)
                slice_node.input_tensors[0].data.dtype = np.float32
                slice_node.input_tensors[0].data[np.where(slice_node.input_tensors[0].data==0)] = -10000
                attr_slice1 = OrderedDict()
                attr_slice1['starts'] = 0
                attr_slice1['ends'] = 128
                attr_slice1['axes'] = 2
                attr_slice1['steps'] = 1
                slice_node.attr = attr_slice1
                attr_slice2 = OrderedDict()
                attr_slice2['starts'] = 0
                attr_slice2['ends'] = 128
                attr_slice2['axes'] = 3
                attr_slice2['steps'] = 1
                model.get_node_by_name(new_node_names[i][1]).attr = attr_slice2
        pattern_dict = pattern_mapping_config['SliceMask'][0]
        model, new_node_names, ret_old_nodes = util.pattern_mapping("SliceMask", 
                                                                    pattern_dict, model)
        if len(new_node_names) != 0:
            _set_attr(new_node_names, ret_old_nodes, model)

        # TODO
        def _set_attr1(new_node_names, ret_old_nodes, model):
            for i in range(len(new_node_names)):
                binary_node_idx = model.get_node_id(new_node_names[i][1])
                attr = OrderedDict()
                model.nodes[binary_node_idx].attr = attr
                slice_node = model.get_node_by_name(new_node_names[i][0])
                import numpy as np
                
                slice_node.input_tensors[0].data = np.array(slice_node.input_tensors[0].data, dtype=np.float32)
                slice_node.input_tensors[0].data.dtype = np.float32
                
                slice_node.input_tensors[0].data[np.where(slice_node.input_tensors[0].data==0)] = -1600000
                slice_node.input_tensors[0].data[np.where(slice_node.input_tensors[0].data==1)] = 0
                attr_slice1 = OrderedDict()
                attr_slice1['starts'] = 0
                attr_slice1['ends_with_tensor'] = 1
                attr_slice1['ends_with_tensor_alg'] = "sub"
                attr_slice1['axes'] = "2, 3"
                attr_slice1['steps'] = 1
                slice_node.attr = attr_slice1
        pattern_dict = pattern_mapping_config['SliceMask'][1]
        model, new_node_names, ret_old_nodes = util.pattern_mapping("SliceMask", 
                                                                    pattern_dict, model)
        if len(new_node_names) != 0:
            _set_attr1(new_node_names, ret_old_nodes, model)
        

        return model