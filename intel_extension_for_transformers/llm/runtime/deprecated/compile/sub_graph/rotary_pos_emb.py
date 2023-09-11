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

"""The RoraryPosEmb Pattern."""

from .pattern import Pattern, pattern_registry
from collections import namedtuple, OrderedDict
from .. import graph_utils as util
from ..ops import Tensor
import numpy as np
import copy


@pattern_registry(pattern_type='RoraryPosEmb')
class RoraryPosEmb(Pattern):
    """The RoraryPosEmb pattern.

    Fuse the original sub-graph into the custom acceleration 'RoraryPosEmb' graph.
    The search strategy is based on the following pattern mapping configs for different models.
    """
    def __call__(self, model):
        """The __call__ function of this pattern class."""
        pattern_mapping_config = {
            'RoraryPosEmb': [
                {
                    'patterns': {
                        'in': [[(0, 'Shape'), (3, 'View'), (4, 'Unsqueeze')],
                               [(), (1, 'View'), (2, 'Repeat'), (3, 'View')]
                                ],
                        'out': [[(0, 'Reshape'), (1, 'Gather')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 0,
                        1: 2,
                    },
                    'input_tensors': {
                        0: [[{
                            0: [0]
                        }], [[0], 1]],
                        1: [[], [[], 1]]
                    },
                    'output_tensors': {
                        0: [[], [[], 1]],
                        1: [[{
                            4: [0]
                        }], [[0], 1]]
                    },
                    'returns': []
                },
                {
                    'patterns': {
                        'in': [[(0, 'Add'), (1, 'Slice'), (2, 'Unsqueeze'), 
                                (3, 'Slice')]
                                ],
                        'out': [[(0, 'Reshape'), (1, 'Slice')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 1,
                        1: 3,
                    },
                    'input_tensors': {
                        0: [[{
                            1: [0]
                        }], [[0], 1]],
                        1: [[{
                            'input_data': [2]
                        },
                             {
                            'input_data': [0]
                        }], [[1,2], 3]]
                    },
                    'output_tensors': {
                        0: [[], [[], 1]],
                        1: [[{
                            3: [0]
                        }], [[0], 1]]
                    },
                    'returns': [0]
                },

                {
                    'patterns': {
                        'in': [[(0, 'Slice'), (1, 'Slice'), (2, 'Slice'), (3, 'Slice'), 
                                (4, 'Neg'), (5, 'Stack'), (6, 'Flatten')]
                                ],
                        'out': [[(0, 'Slice'), (1, 'Mul'), (2, 'Reshape'), (3, 'Concat'), (4, 'Reshape')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 3,
                        1: 4,
                        2: 0,
                        3: 5,
                        4: 6
                    },
                    'input_tensors': {
                        0: [[{
                            0: [0]
                        }], [[0], 1]],
                        1: [[], [[], 1]],
                        2: [[], [[], 1]],
                        3: [[{
                            5: [1]
                        }], [[1], 2]],
                        4: [[], [[], 1]],
                    },
                    'output_tensors': {
                        0: [[], [[], 1]],
                        1: [[], [[], 1]],
                        2: [[], [[], 1]],
                        3: [[], [[], 1]],
                        4: [[{
                            6: [0]
                        }], [[0], 1]]
                    },
                    'returns': [0, 3, 5]
                },
                
                {
                    'patterns': {
                        'in': [[(0, 'Shape'), (2, 'Add'), (3, 'Arange')],
                               [(), (1, 'Shape'), (2, 'Add')]
                                ],
                        'out': [[(0, 'Range')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 3,
                    },
                    'input_tensors': {
                        0: [[{
                            'input_data': [0]
                        },{
                            'input_data': [2]
                        }], [[0, 1], 2]]
                    },
                    'output_tensors': {
                        0: [[{
                            3: [0]
                        }], [[0], 1]]
                    },
                    'returns': [0]
                }
            ]
        }
        if model.framework_modeling_config['framework'] != 'torch':
            return model
        def _set_attr(new_node_names, ret_old_nodes, model):
            for i in range(len(new_node_names)):
                in_reshape_node = model.get_node_by_name(new_node_names[i][0])
                in_reshape_node.attr = OrderedDict({"dst_shape" : "-1, 1"})
                gather_node = model.get_node_by_name(new_node_names[i][1])
                gather_node.attr = OrderedDict({'axis': '0', 'batch_dims': '1'})
                idx_value = np.array([0, 0], dtype=np.int32)
                idx_tensor = Tensor(name=new_node_names[i][1] + "_idx",
                                        source_op=[],
                                        dest_op=[new_node_names[i][1]],
                                        data = idx_value,
                                        shape = [2],
                                        dtype="int32")
                gather_node.input_tensors.insert(0, idx_tensor)
            
            #  batch_dims: 0
                
        pattern_dict = pattern_mapping_config['RoraryPosEmb'][0]
        model, new_node_names, ret_old_nodes = util.pattern_mapping("RoraryPosEmb", 
                                                                    pattern_dict, model)
        if len(new_node_names) != 0:
            _set_attr(new_node_names, ret_old_nodes, model)
            

        def _set_attr1(new_node_names, ret_old_nodes, model):
            remove_shape_list = []
            for i in range(len(new_node_names)):
                for in_tensor in ret_old_nodes[i][0].input_tensors:
                    shape_node = model.get_node_by_name(in_tensor.source_op[0])
                    if len(shape_node.output_tensors[0].dest_op) == 0:
                        remove_shape_list.append(shape_node.name)
                reshape_node = model.get_node_by_name(new_node_names[i][0])
                reshape_node.attr = OrderedDict({"dst_shape" : "1, -1, 1, 64"})
                slice_node = model.get_node_by_name(new_node_names[i][1])
                slice_node.attr = OrderedDict({"starts_with_tensor" : "1",
                                               "ends_add" : "1", "axes" : "1", "steps" : "1"})
            model.remove_nodes(remove_shape_list)
                
        pattern_dict = pattern_mapping_config['RoraryPosEmb'][1]
        model, new_node_names, ret_old_nodes = util.pattern_mapping("RoraryPosEmb", 
                                                                    pattern_dict, model)
        if len(new_node_names) != 0:
            _set_attr1(new_node_names, ret_old_nodes, model)

        def _set_attr2(new_node_names, ret_old_nodes, model):
            for i in range(len(new_node_names)):
                slice_node = model.get_node_by_name(new_node_names[i][0])
                slice_node.attr = ret_old_nodes[i][1].attr
                neg_value = np.array([-1], dtype=np.float32)
                neg_tensor = Tensor(name=new_node_names[i][1] + "_neg",
                                        source_op=[],
                                        dest_op=[new_node_names[i][1]],
                                        data = neg_value,
                                        shape = [1],
                                        dtype="fp32")
                mul_node = model.get_node_by_name(new_node_names[i][1])
                mul_node.input_tensors.append(neg_tensor)
                mul_node.attr = OrderedDict({'algorithm': 'mul'})
                old_slice_node = model.get_node_by_name(ret_old_nodes[i][0].input_tensors[0].source_op[0])
                concat_node = model.get_node_by_name(new_node_names[i][3])
                first_slice_node = model.get_node_by_name(concat_node.input_tensors[1].source_op[0])
                old_slice_node.output_tensors[0].dest_op.append(first_slice_node.name)
                first_slice_node.input_tensors[0] = old_slice_node.output_tensors[0]
                concat_node.attr = OrderedDict({'axis': '4'})
                reshape_node = model.get_node_by_name(new_node_names[i][2])
                reshape_node.attr = OrderedDict({'unsqueeze': '-1'})
                
                reshape_output = Tensor(name=concat_node.input_tensors[1].name + "_reshape",
                                        source_op=[concat_node.name + "_reshape"],
                                        dest_op=[concat_node.name],
                                        dtype=concat_node.input_tensors[1].dtype)
                
                reshape_op = util.construct_node(
                    node_name=concat_node.name + "_reshape",
                    op_type='Reshape',
                    input_tensors=[concat_node.input_tensors[1]],
                    output_tensors=[reshape_output],
                    attr=OrderedDict({'unsqueeze': '-1'}))
                concat_node.input_tensors[1].source_op = [reshape_op.name]
                concat_node.input_tensors[1] = reshape_output
                insert_idx = model.get_node_id(new_node_names[i][3])
                model.insert_nodes(insert_idx, [reshape_op])
                
                last_reshape_node = model.get_node_by_name(new_node_names[i][4])
                last_reshape_node.attr = OrderedDict({'mul': '3, 4'})
        pattern_dict = pattern_mapping_config['RoraryPosEmb'][2]
        model, new_node_names, ret_old_nodes = util.pattern_mapping("RoraryPosEmb", 
                                                                    pattern_dict, model)
        if len(new_node_names) != 0:
            _set_attr2(new_node_names, ret_old_nodes, model)
            
        return model
