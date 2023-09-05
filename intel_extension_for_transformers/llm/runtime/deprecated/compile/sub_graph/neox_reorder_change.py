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

"""The NeoxReorderChange pattern."""

from .pattern import Pattern, pattern_registry
from collections import namedtuple, OrderedDict
from .. import graph_utils as util
from ..onnx_utils import bias_to_int32
import copy
from ..ops import Tensor

@pattern_registry(pattern_type='NeoxReorderChange')
class NeoxReorderChange(Pattern):
    """The NeoxReorderChange pattern.

    Fuse the original sub-graph into the custom acceleration 'NeoxReorderChange' graph.
    The search strategy is based on the following pattern mapping configs for different models.
    """
    def __call__(self, model):
        """The __call__ function of this pattern class."""
        pattern_mapping_config = {
            'NeoxReorderChange': [
                    {'patterns': {
                        'in': [[(0, 'Slice'), (1, 'Reorder'), (2, 'Slice'), (3, 'Concat'), (4, 'Matmul')]],
                    }},
                    {'patterns': {
                        'in': [[(0, 'Slice'), (1, 'Reorder'), (2, 'Slice'), (3, 'Concat'), (4, 'Concat'),
                            (5, 'Reorder'), (6, 'Matmul')]],
                    }},
                    {'patterns': {
                        'in': [[(0, 'Slice'), (1, 'Reorder'), (2, 'Concat'), (4, 'Matmul')],
                               [(), (3, 'Softmax'), (4, 'Matmul')]],
                    }},
                    {'patterns': {
                        'in': [[(0, 'Softmax'), (1, 'Matmul'), (2, 'Reorder'), (3, 'View')]],
                    }},
                    {'patterns': {
                        'in': [[(0, 'Concat'), (1, 'Shape'), (3, 'Mul')],
                            [(), (0, 'Concat'), (2, 'Shape'), (3, 'Mul')]],
                        'out': [[(0, 'Concat')]]
                    }},
                    {   'patterns': {
                            'in': [[(0, 'Softmax'), (1, 'Matmul'), (2, 'Reorder'), (3, 'Reshape')]],
                            'out': [[(0, 'Softmax'), (1, 'Matmul')]]
                        },
                        'search_mode': 'op_type',
                        'node_names': {
                            0: 0,
                            1: 1
                        },
                        'input_tensors': {
                            0: [[
                                {0: [0]},
                                ],
                                [[0], 1]],
                            1: [[
                                {1: [0]},
                                {1: [1]}
                                ],
                                [[0, 1], 2]],
                        },
                        'output_tensors': {
                            0 : [[{0: [0]}], [[0], 1]],
                            1 : [[{3: [0]}], [[0], 1]]
                        },
                        'returns': [2, 3]
                    },
            ]
        }
        if model.framework_modeling_config['framework'] != 'torch':
            return model

        remove_list = []
        patterns_nodes_name = util.search_pattern(pattern_mapping_config
                ['NeoxReorderChange'][0]['patterns']['in'], model)
        if len(patterns_nodes_name) != 0:
            for nodes_name in patterns_nodes_name:
                reorder_node = model.get_node_by_name(nodes_name[1])
                if reorder_node.op_type != 'Reorder':
                    continue
                source_node = model.get_node_by_name(nodes_name[0])
                source_node.output_tensors[0] = copy.deepcopy(reorder_node.output_tensors[0])
                source_node.output_tensors[0].source_op = source_node.name
                for dest_name in reorder_node.output_tensors[0].dest_op:
                    dest_node = model.get_node_by_name(dest_name)
                    for in_tensor in dest_node.input_tensors:
                        if reorder_node.output_tensors[0].name in in_tensor.source_op:
                            in_tensor.source_op.remove(reorder_node.output_tensors[0].name)
                            in_tensor.source_op.append(source_node.name)
                remove_list.append(reorder_node.name)
                mm_node = model.get_node_by_name(nodes_name[4])
                mm_node.attr['src0_perm'] = '0,2,1,3'

        patterns_nodes_name = util.search_pattern(pattern_mapping_config
                ['NeoxReorderChange'][1]['patterns']['in'], model)
        if len(patterns_nodes_name) != 0:
            for nodes_name in patterns_nodes_name:
                reorder_node = model.get_node_by_name(nodes_name[1])
                if reorder_node.op_type != 'Reorder':
                    continue
                source_node = model.get_node_by_name(nodes_name[0])
                source_node.output_tensors[0] = copy.deepcopy(reorder_node.output_tensors[0])
                source_node.output_tensors[0].source_op = source_node.name
                for dest_name in reorder_node.output_tensors[0].dest_op:
                    dest_node = model.get_node_by_name(dest_name)
                    for in_tensor in dest_node.input_tensors:
                        if reorder_node.output_tensors[0].name in in_tensor.source_op:
                            in_tensor.source_op.remove(reorder_node.output_tensors[0].name)
                            in_tensor.source_op.append(source_node.name)
                remove_list.append(reorder_node.name)

                reorder_node = model.get_node_by_name(nodes_name[5])
                if reorder_node.op_type != 'Reorder':
                    continue
                source_node = model.get_node_by_name(nodes_name[4])
                source_node.output_tensors[0].dest_op = [nodes_name[6]]
                mm_node = model.get_node_by_name(nodes_name[6])
                mm_node.input_tensors[1] = copy.deepcopy(source_node.output_tensors[0])

                remove_list.append(reorder_node.name)

                cat_node = model.get_node_by_name(nodes_name[4])
                cat_node.attr['axis'] = 1
                mm_node.attr['src1_perm'] = '0,2,3,1'

        patterns_nodes_name = util.search_pattern(pattern_mapping_config
                ['NeoxReorderChange'][2]['patterns']['in'], model)
        if len(patterns_nodes_name) != 0:
            for nodes_name in patterns_nodes_name:
                reorder_node = model.get_node_by_name(nodes_name[1])
                if reorder_node.op_type != 'Reorder':
                    continue
                source_node = model.get_node_by_name(nodes_name[0])
                source_node.output_tensors[0] = copy.deepcopy(reorder_node.output_tensors[0])
                source_node.output_tensors[0].source_op = source_node.name
                for dest_name in reorder_node.output_tensors[0].dest_op:
                    dest_node = model.get_node_by_name(dest_name)
                    for in_tensor in dest_node.input_tensors:
                        if reorder_node.output_tensors[0].name in in_tensor.source_op:
                            in_tensor.source_op.remove(reorder_node.output_tensors[0].name)
                            in_tensor.source_op.append(source_node.name)
                remove_list.append(reorder_node.name)
                cat_node = model.get_node_by_name(nodes_name[2])
                cat_node.attr['axis'] = 1
                mm_node = model.get_node_by_name(nodes_name[4])
                mm_node.attr['src1_perm'] = '0,2,1,3'


        model.remove_nodes(remove_list)
        remove_list.clear()
        patterns_nodes_name = util.search_pattern(pattern_mapping_config
                ['NeoxReorderChange'][3]['patterns']['in'], model)
        if len(patterns_nodes_name) != 0:
            for nodes_name in patterns_nodes_name:
                view_node = model.get_node_by_name(nodes_name[3])
                if view_node.op_type != 'View':
                    continue
                view_node.op_type = 'Reshape'
                view_node.input_tensors = view_node.input_tensors[0:1]
                view_node.attr['dst_shape'] = '-1,-1,-1'
                view_node.attr['dims'] = '0,1'
                view_node.attr['mul'] = '0,1'

        patterns_nodes_name = util.search_pattern(pattern_mapping_config
                ['NeoxReorderChange'][4]['patterns']['in'], model)
        if len(patterns_nodes_name) != 0:
            for nodes_name in patterns_nodes_name:
                remove_list.append(nodes_name[1])
                remove_list.append(nodes_name[2])
                remove_list.append(nodes_name[3])
        model.remove_nodes(remove_list)

        # embedding reshape
        node_idx = 0
        while node_idx < len(model.nodes):
            node = model.nodes[node_idx]
            if node.op_type == 'Gather':
                temp_tensor = Tensor(name=node.name + "_out_ts",
                                    source_op=[node.name],
                                    dest_op=[node.name + "_reshape"],
                                    dtype=node.output_tensors[0].dtype)
                node.output_tensors[0].source_op = [node.name + "_reshape"]
                reshape_node = util.construct_node(node_name=node.name + "_reshape",
                                                    op_type='Reshape',
                                                    input_tensors=[copy.deepcopy(temp_tensor)],
                                                    output_tensors=[copy.deepcopy(node.output_tensors[0])])
                node.output_tensors[0] = copy.deepcopy(temp_tensor)
                attr = OrderedDict()
                attr['dst_shape'] = '-1,-1,-1'
                attr['dims'] = '0,1'
                attr['mul'] = '0,1'
                reshape_node.attr = attr
                model.insert_nodes(node_idx + 1, [reshape_node])
                node_idx += 1
            node_idx += 1

        # fuse matmul+reorder+reshape
        patterns_nodes_name = util.search_pattern(pattern_mapping_config
                ['NeoxReorderChange'][5]['patterns']['in'], model)
        remove_list = []
        if len(patterns_nodes_name) != 0:
            for nodes_name in patterns_nodes_name:
                mm_node = model.get_node_by_name(nodes_name[1])
                reorder_node = model.get_node_by_name(nodes_name[2])
                reshape_node = model.get_node_by_name(nodes_name[3])
                dest_node = model.get_node_by_name(reshape_node.output_tensors[0].dest_op[0])
                mm_node.attr['dst_perm'] = reorder_node.attr['dst_perm']
                mm_node.attr['reshape'] = reshape_node.attr['shape'][3:]
                mm_node.output_tensors[0] = copy.deepcopy(reshape_node.output_tensors[0])
                mm_node.output_tensors[0].source_op.append(mm_node.name)
                if reshape_node.name in mm_node.output_tensors[0].source_op:
                    mm_node.output_tensors[0].source_op.remove(reshape_node.name)
                dest_node.input_tensors[0].source_op.append(mm_node.name)
                if reshape_node.name in dest_node.input_tensors[0].source_op:
                    dest_node.input_tensors[0].source_op.remove(reshape_node.name)
                remove_list += reorder_node.output_tensors[0].dest_op
                remove_list.append(reorder_node.name)

        model.remove_nodes(remove_list)
        return model
