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

"""The Gelu Pattern."""

from .pattern import Pattern, pattern_registry
from collections import namedtuple, OrderedDict
import copy
from .. import graph_utils as util


@pattern_registry(pattern_type='Gelu')
class Gelu(Pattern):
    """The Gelu pattern.

    Fuse the original sub-graph into the custom acceleration 'Gelu' graph.
    The search strategy is based on the following pattern mapping configs for different models.
    """
    def __call__(self, model):
        """The __call__ function of this pattern class."""
        pattern_mapping_config = {
            'Gelu': [
                {
                    'patterns': {
                        'in': [[(0,'Pow'), (1, 'Mul'), (2, 'AddV2'), (3, 'Mul'), (4, 'Tanh'),
                                (5, 'AddV2'), (6, 'Mul'), (7, 'Mul')]],
                        'out': [[(0, 'Gelu')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 7
                    },
                    'input_tensors': {
                        0: [[{
                            0: [0]
                        }], [[0], 1]]
                    },
                    'output_tensors': {
                        0: [[{
                            7: [0]
                        }], [[0], 1]]
                    },
                    'returns': []
                },

                # distilbert_base
                {
                    'patterns': {
                        'in': [[(0, 'Div'), (1, 'Erf'), (2, 'Add'), (3, 'Mul'), (4, 'Mul')]],
                        'out': [[(0, 'Gelu')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 4
                    },
                    'input_tensors': {
                        0: [[{
                            0: [0]
                        }], [[0], 1]]
                    },
                    'output_tensors': {
                        0: [[{
                            4: [0]
                        }], [[0], 1]]
                    },
                    'returns': []
                },

                # gpt_neox
                {
                    'patterns': {
                        'in': [[(0,'Mul'), (1, 'Mul'), (2, 'Add'), (3, 'Mul'), (4, 'Tanh'),
                                (5, 'Add'), (6, 'Mul')],
                                [(), (7, 'Mul'), (3, 'Mul')],
                                 [(),(8, 'Mul'),(6, 'Mul')]],
                        'out': [[(0, 'Gelu')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 6
                    },
                    'input_tensors': {
                        0: [[{
                            0: [0]
                        }], [[0], 1]]
                    },
                    'output_tensors': {
                        0: [[{
                            6: [0]
                        }], [[0], 1]]
                    },
                    'returns': []
                },
            ]
        }

        for i in range(len(pattern_mapping_config['Gelu'])):
            pattern_dict = pattern_mapping_config['Gelu'][i]
            model, new_node_names, ret_old_nodes = util.pattern_mapping("Gelu", 
                                                                        pattern_dict, model)
            if len(new_node_names) != 0:
                for j in range(len(new_node_names)):
                    gelu_node = model.get_node_by_name(new_node_names[j][0])
                    attr = OrderedDict()
                    attr['algorithm'] = 'gelu_tanh'
                    gelu_node.attr = attr

        return model
