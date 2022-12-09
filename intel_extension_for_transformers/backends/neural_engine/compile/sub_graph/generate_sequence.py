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

"""The GenerateSequence Pattern."""

from .pattern import Pattern, pattern_registry
from collections import namedtuple, OrderedDict
from .. import graph_utils as util


@pattern_registry(pattern_type='GenerateSequence')
class GenerateSequence(Pattern):
    """The GenerateSequence pattern.

    Fuse the original sub-graph into the custom acceleration 'GenerateSequence' graph.
    The fusion strategy is based on 'AddClsToken' pattern map configurations and different kinds of models.
    """

    def __call__(self, model):
        """The __call__ function of this pattern class."""
        pattern_mapping_config = {
            'GenerateSequence': [{
                'patterns': {
                    'in': [[(0, "Shape"), (1, 'Gather'), (2, "Unsqueeze"), (3, "Concat"),
                            (4, "Shape"), (5, "ConstantOfShape"), (6, "Expand"), (7, "Tile")],
                           [(), (8, 'Shape'), (9, 'Gather'), (10, 'Cast'), (11, 'Range'),
                            (12, "Unsqueeze"), (6, "Expand")],
                           [(1, 'Gather'), (13, "Unsqueeze"), (14, "Concat"), (7, "Tile")]],
                    'out': [[(0, 'Range')]]
                },
                'search_mode': 'op_type',
                'node_names': {
                    0: 11
                },
                'input_tensors': {
                    0: [[{
                        'input_data': [0]
                    }], [[0], 1]],
                },
                'output_tensors': {
                    0: [[{
                        7: [0]
                    }], [[0], 1]],
                },
                'returns': [11]
            }]
        }

        for i in range(len(pattern_mapping_config['GenerateSequence'])):
            pattern_dict = pattern_mapping_config['GenerateSequence'][i]
            model, new_node_names, ret_old_nodes = util.pattern_mapping(
                "GenerateSequence", pattern_dict, model)
            if len(new_node_names) != 0:
                for j in range(len(new_node_names)):
                    old_node = ret_old_nodes[j][0]
                    attr = OrderedDict()
                    attr["start"] = int(old_node.input_tensors[0].data)
                    attr["step"] = int(old_node.input_tensors[2].data)
                    new_node_idx = model.get_node_id(new_node_names[j][0])
                    model.nodes[new_node_idx].attr = attr
                return model

        return model
