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

"""The RmsNorm Pattern."""

from .pattern import Pattern, pattern_registry
from collections import namedtuple, OrderedDict
from .. import graph_utils as util


@pattern_registry(pattern_type='RmsNorm')
class RmsNorm(Pattern):
    """The RmsNorm pattern.

    Fuse the original sub-graph into the custom acceleration 'RmsNorm' graph.
    The search strategy is based on the following pattern mapping configs for different models.
    """
    def __call__(self, model):
        """The __call__ function of this pattern class."""
        pattern_mapping_config = {
            'RmsNorm': [

                # bert_base has two layer_norm patterns
                {
                    'patterns': {
                        'in': [[(0, 'Pow'), (1, 'ReduceMean'), (2, 'Add'), (3, 'Rsqrt'),
                                (4, 'Mul'), (5, 'Mul')]
                              ],
                        'out': [[(0, 'RmsNorm')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 5
                    },
                    'input_tensors': {
                        0: [[{
                            0: [0]
                        }, {
                            5: [0]
                        }], [[0, 1], 2]]
                    },
                    'output_tensors': {
                        0: [[{
                            5: [0]
                        }], [[0], 1]]
                    },
                    'returns': [2]
                },

            ]
        }

        def _set_attr(epsilon, node_names, model):
            attr = OrderedDict()
            attr['epsilon'] = float(epsilon.input_tensors[1].data)
            ln_node_idx = model.get_node_id(node_names[0])
            model.nodes[ln_node_idx].attr = attr
            
            if len(model.nodes[ln_node_idx].input_tensors) == 2:
                hidden_size = model.nodes[ln_node_idx].input_tensors[1].data.shape[0]
                model.add_config_item("hidden_size", hidden_size)

        # import pdb;pdb.set_trace()
        pattern_dict = pattern_mapping_config['RmsNorm'][0]
        model, new_node_names, ret_old_nodes = util.pattern_mapping("RmsNorm", 
                                                                    pattern_dict, model)
        if len(new_node_names) != 0:
            for i in range(len(new_node_names)):
                epsilon = ret_old_nodes[i][0]
                _set_attr(epsilon, new_node_names[i], model)

            return model

        return model
