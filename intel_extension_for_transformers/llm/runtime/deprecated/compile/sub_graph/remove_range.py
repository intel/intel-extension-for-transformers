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

"""The RemoveRange Pattern."""

from .pattern import Pattern, pattern_registry
from collections import namedtuple, OrderedDict
from .. import graph_utils as util
from .subgraph_matcher import EXECUTOR_TYPE
import numpy as np
import copy

@pattern_registry(pattern_type='RemoveRange')
class RemoveRange(Pattern):
    """The RemoveRange pattern.

    Fuse the original sub-graph into the custom acceleration 'RemoveRange' graph.
    The search strategy is based on the following pattern mapping configs for different models.
    """

    def __call__(self, model):
        """The __call__ function of this pattern class."""
        pattern_mapping_config = {
            'RemoveRange': [
                {
                    'patterns': {
                        'in': [[(0, 'Range'), (1, 'Div'), (2, 'Pow'), (3, 'Div'), (4, 'Reshape'), (7, 'Matmul')],
                               [(), (5, 'Range'), (6, 'Reshape'), (7, 'Matmul')]
                        ],
                    },
                },
                
                {
                    'patterns': {
                        'in': [[(0, 'Cos'), (1, 'Reshape'), (2, 'Gather'), (3, 'Reshape'), 
                                (4, 'Slice')]
                        ],
                    },
                },
                {
                    'patterns': {
                        'in': [[(0, 'Sin'), (1, 'Reshape'), (2, 'Gather'), (3, 'Reshape'), 
                                (4, 'Slice')]
                        ],
                    },
                },   
                
            ],
        }

        if model.framework_modeling_config['framework'] != 'torch':
            return model

        pattern = pattern_mapping_config['RemoveRange'][0]['patterns']['in']
        patterns_nodes_name = util.search_pattern(pattern, model)
        once_flag = False
        for pattern_nodes_name in patterns_nodes_name:
            if once_flag == False:
                once_flag = True
                continue
            remove_list = []
            for node in pattern_nodes_name:
                if not (isinstance(node, list)):
                    remove_list.append(node)
            model.remove_nodes(remove_list)
            
        pattern = pattern_mapping_config['RemoveRange'][1]['patterns']['in']
        patterns_nodes_name = util.search_pattern(pattern, model)
        once_flag2 = False
        
        keep_list = []
        for pattern_nodes_name in patterns_nodes_name:
            if once_flag2 == False :
                once_flag2 = True
                keep_list = pattern_nodes_name
                continue
            remove = model.get_node_by_name(pattern_nodes_name[-2])
            keep = model.get_node_by_name(keep_list[-2])
            
            next_node = model.get_node_by_name(remove.output_tensors[0].dest_op[0])
            keep.output_tensors[0].dest_op.append(next_node.name)
            next_node.input_tensors[1] = keep.output_tensors[0]
            for node_name in pattern_nodes_name:
                if node_name not in keep_list and not (isinstance(node_name, list)):
                    model.remove_nodes([node_name])
            
        pattern = pattern_mapping_config['RemoveRange'][2]['patterns']['in']
        patterns_nodes_name = util.search_pattern(pattern, model)
        once_flag2 = False
        
        keep_list = []
        for pattern_nodes_name in patterns_nodes_name:
            if once_flag2 == False :
                once_flag2 = True
                keep_list = pattern_nodes_name
                continue
            remove = model.get_node_by_name(pattern_nodes_name[-2])
            keep = model.get_node_by_name(keep_list[-2])
            
            next_node = model.get_node_by_name(remove.output_tensors[0].dest_op[0])
            keep.output_tensors[0].dest_op.append(next_node.name)
            next_node.input_tensors[1] = keep.output_tensors[0]
            for node_name in pattern_nodes_name:
                if node_name not in keep_list and not (isinstance(node_name, list)):
                    model.remove_nodes([node_name])
            
        return model
