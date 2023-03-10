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

"""The AttentionBlock_WeightReshapeTo4D Pattern."""

from .pattern import Pattern, pattern_registry
from collections import namedtuple, OrderedDict
from .. import graph_utils as util
from .subgraph_matcher import EXECUTOR_TYPE
import numpy as np
import copy

@pattern_registry(pattern_type='AttentionBlock_WeightReshapeTo4D')
class AttentionBlock_WeightReshapeTo4D(Pattern):
    """The AttentionBlock_WeightReshapeTo4D pattern.

    Fuse the original sub-graph into the custom acceleration 'AttentionBlock_WeightReshapeTo4D' graph.
    The search strategy is based on the following pattern mapping configs for different models.
    """

    def __call__(self, model):
        """The __call__ function of this pattern class."""
        pattern_mapping_config = {
            'AttentionBlock_WeightReshapeTo4D': [
                {
                    'patterns': {
                        'in': [[(0, 'Reshape'), (1, 'Mul'), (2, 'Add'), (3, 'Sigmoid')]],
                    },
                },
            ]
        }

        pattern = pattern_mapping_config['AttentionBlock_WeightReshapeTo4D'][0]['patterns']['in']
        patterns_nodes_name = util.search_pattern(pattern, model)
        print('AttentionBlock_WeightReshapeTo4D = ', patterns_nodes_name)

        if len(patterns_nodes_name) != 0:
            for j in range(len(patterns_nodes_name)):
                add_node = model.get_node_by_name(patterns_nodes_name[j][2])
                add_node.input_tensors[1].shape = [1, 512, 1, 1]

        return model

