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


@pattern_registry(pattern_type='NeoxRoraryPosEmb')
class NeoxRoraryPosEmb(Pattern):
    """The NeoxRoraryPosEmb pattern.

    Fuse the original sub-graph into the custom acceleration 'NeoxRoraryPosEmb' graph.
    The search strategy is based on the following pattern mapping configs for different models.
    """
    def __call__(self, model):
        """The __call__ function of this pattern class."""
        pattern_mapping_config = {
            'NeoxRoraryPosEmb': [
                {
                    'patterns': {
                        'in': [[(0, 'Shape'), (1, 'Add'), (2, 'Slice'), (3, 'Slice'),
                                (4, 'Slice')]
                                ],
                        'out': [[(0, 'Slice')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 4,
                    },
                    'input_tensors': {
                        0: [[{
                            2: [0]
                        },{
                           'input_data': [3]
                        }], [[0, 1], 2]]
                    },
                    'output_tensors': {
                        0: [[{
                            4: [0]
                        }], [[0], 1]]
                    },
                    'returns': [3]
                },
                {
                    'patterns': {
                        'in': [[(0, 'Shape'), (1, 'Add')]],
                    },
                },
                
                {
                    'patterns': {
                        'in': [[(0, 'Shape'), (1, 'Div'), (2, 'Slice'), (4, 'Neg'), (5, 'Concat')],
                               [(1, 'Div'), (3, 'Slice'), (5, 'Concat')]],
                    },
                },
                {
                    'patterns': {
                        'in': [[(0, 'Slice'), (1, 'Repeat'), (4, 'Gather'), (5, 'Reshape')],
                               [(), (2, 'prim::gather_indices'), (3, 'Repeat'), (4, 'Gather')]],
                        'out': [[(0, 'Slice')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 5,
                    },
                    'input_tensors': {
                        0: [[{
                            0: [0]
                        },{
                           'input_data': [3]
                        }], [[0, 1], 2]]
                    },
                    'output_tensors': {
                        0: [[{
                            5: [0]
                        }], [[0], 1]]
                    },
                    'returns': [0, 1, 3]
                },
            ]
        }
        
        def _set_attr(new_node_names, ret_old_nodes, model):
            remove_shape_list = []
            for i in range(len(new_node_names)):
                attr_slice = OrderedDict()
                attr_slice['axes'] = 1
                attr_slice['steps'] = ret_old_nodes[i][0].attr['steps']
                attr_slice['starts_with_tensor'] = 1
                attr_slice['ends_with_tensor'] = 1
                slice_node = model.get_node_by_name(new_node_names[i][0])
                if 'ends_with_tensor' not in slice_node.attr:
                    slice_node.input_tensors[0].data = slice_node.input_tensors[0].data.transpose(0,2,1,3)
                    shape = slice_node.input_tensors[0].shape
                    slice_node.input_tensors[0].shape = [shape[0], shape[2], shape[1], shape[3]]
                slice_node.attr = attr_slice

                mask_tensor = None
                for tensor in model.nodes[0].output_tensors:
                    if 'mask' in tensor.name:
                        mask_tensor = copy.deepcopy(tensor)
                        break
                slice_node.input_tensors.append(mask_tensor)
        
        if model.framework_modeling_config['framework'] != 'torch':
            return model
 
        pattern_dict = pattern_mapping_config['NeoxRoraryPosEmb'][0]
        model, new_node_names, ret_old_nodes = util.pattern_mapping("NeoxRoraryPosEmb", 
                                                                    pattern_dict, model)
        if len(new_node_names) != 0:
            _set_attr(new_node_names, ret_old_nodes, model)
            # remove unused shape and add
            pattern = pattern_mapping_config['NeoxRoraryPosEmb'][1]['patterns']['in']
            patterns_nodes_name = util.search_pattern(pattern, model)
            remove_node_list = []
            for pattern_nodes_name in patterns_nodes_name:
                add_node = model.get_node_by_name(pattern_nodes_name[1])
                is_remove = True
                output_name = add_node.output_tensors[0].name
                for node in model.nodes:
                    for input_tensor in node.input_tensors:
                        if output_name == input_tensor.name:
                            is_remove = False
                if is_remove:
                    remove_node_list.extend([pattern_nodes_name[0], pattern_nodes_name[1]])
            model.remove_nodes(remove_node_list)

        # rotate_half pattern for llama 
        pattern = pattern_mapping_config['NeoxRoraryPosEmb'][2]['patterns']['in']
        patterns_nodes_name = util.search_pattern(pattern, model)
        remove_node_list = []

        for pattern_nodes_name in patterns_nodes_name:
            slice1_node = model.get_node_by_name(pattern_nodes_name[2])
            source_slice_node = model.get_node_by_name(slice1_node.input_tensors[0].source_op[0])
            slice_sign = source_slice_node.attr['ends'] - source_slice_node.attr['starts']
            slice1_node.attr['starts'] = slice_sign//2
            slice1_node.input_tensors = [slice1_node.input_tensors[0]]
            slice2_node = model.get_node_by_name(pattern_nodes_name[3])
            slice2_node.attr['ends'] = slice_sign//2
            slice2_node.input_tensors = [slice2_node.input_tensors[0]]

            remove_node_list.extend([pattern_nodes_name[0], pattern_nodes_name[1]])
        model.remove_nodes(remove_node_list)


        pattern_dict = pattern_mapping_config['NeoxRoraryPosEmb'][3]
        model, new_node_names, ret_old_nodes = util.pattern_mapping("NeoxRoraryPosEmb", 
                                                                    pattern_dict, model)
        remove_node_list = []
        for i in range(len(new_node_names)):
            attr_slice = OrderedDict()
            attr_slice['axes'] = 1
            attr_slice['steps'] = 1
            attr_slice['starts_with_tensor'] = 1
            attr_slice['ends_with_tensor'] = 1
            slice_node = model.get_node_by_name(new_node_names[i][0])
            if 'ends_with_tensor' not in slice_node.attr:
                slice_node.input_tensors[0].data = slice_node.input_tensors[0].data.transpose(0,2,1,3)
                shape = slice_node.input_tensors[0].shape
                slice_node.input_tensors[0].shape = [shape[0], shape[2], shape[1], shape[3]]
            slice_node.attr = attr_slice
            mask_tensor = None
            for tensor in model.nodes[0].output_tensors:
                if 'mask' in tensor.name:
                    mask_tensor = copy.deepcopy(tensor)
                    break
            slice_node.input_tensors.append(mask_tensor)

            ori_slice_node = ret_old_nodes[i][0]
            add_node = model.get_node_by_name(ori_slice_node.input_tensors[1].source_op[0])
            remove_node_list.append(add_node.name)

            repeat_node = ret_old_nodes[i][1]
            shape_node = model.get_node_by_name(repeat_node.input_tensors[1].source_op[0])
            remove_node_list.append(shape_node.name)

            repeat2_node = ret_old_nodes[i][2]
            shape_node2 = model.get_node_by_name(repeat2_node.input_tensors[1].source_op[0])
            shape_node3 = model.get_node_by_name(repeat2_node.input_tensors[2].source_op[0])
            remove_node_list.append(shape_node2.name)
            remove_node_list.append(shape_node3.name)
        model.remove_nodes(remove_node_list)

        return model
