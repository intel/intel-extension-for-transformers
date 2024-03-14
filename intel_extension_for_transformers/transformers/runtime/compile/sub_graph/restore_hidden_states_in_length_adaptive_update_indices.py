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

"""The RestoreHiddenStatesInLengthAdaptive pattern."""

from .pattern import Pattern, pattern_registry
from collections import namedtuple, OrderedDict
from .. import graph_utils as util
import numpy as np
import copy


@pattern_registry(pattern_type='RestoreHiddenStatesInLengthAdaptiveUpdateIndices')
class RestoreHiddenStatesInLengthAdaptive(Pattern):
    """The RestoreHiddenStatesInLengthAdaptive pattern.

    Fuse the original sub-graph into the custom acceleration 'RestoreHiddenStatesInLengthAdaptive' graph.
    The search strategy is based on the following pattern mapping configs for different models.
    """

    def __call__(self, model):
        """The __call__ function of this pattern class."""
        pattern_mapping_config = {
            'RestoreHiddenStatesInLengthAdaptiveUpdateIndices': [
                # minilmv2-lat-roberta
                {
                    'patterns': {
                        'in': [[(0, 'Shape'), (1, 'Gather'), (2, 'Unsqueeze'), (3, 'Concat'),
                                (4, 'Reshape'), (5, 'Shape'), (6, 'ConstantOfShape'), (7, 'Mul'),
                                (8, 'Equal'), (9, 'Where'), (11, 'Expand'),
                                (12, 'ScatterElements')],
                               [(), (10, 'Unsqueeze'), (11, 'Expand')]],
                        'out': [[(0, 'Reshape'), (1, 'ExpandIndices'), (2, 'ScatterElements')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 'reshape_to_3d_before_restoration',
                        1: 11,
                        2: 12,
                    },
                    'input_tensors': {
                        0: [[{
                            0: [0]
                        }, {
                            'input_data': [0]
                        }], [[0, 1], 2]],
                        1: [[{
                            10: [0]
                        }], [[0], 2]],
                        2: [[{
                            12: [0],
                        }, {
                            12: [2]
                        }], [[0, 2], 3]],
                    },
                    'output_tensors': {
                        0: [[], [[], 1]],
                        1: [[], [[], 1]],
                        2: [[{
                            12: [0]
                        }], [[0], 1]],
                    },
                    'returns': [10, 12]
                },
                {
                    'patterns': {
                        'in': [[(0, 'Shape'), (1, 'Gather'), (2, 'Unsqueeze'), (3, 'Concat'),
                                (4, 'Reshape'), (5, 'Equal'), (6, 'Where'), (8, 'Expand'),
                                (9, 'ScatterElements')],
                               [(), (7, 'Unsqueeze'), (8, 'Expand')]],
                        'out': [[(0, 'Reshape'), (1, 'ExpandIndices'), (2, 'ScatterElements')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 'reshape_to_3d_before_restoration',
                        1: 8,
                        2: 9,
                    },
                    'input_tensors': {
                        0: [[{
                            0: [0]
                        }, {
                            'input_data': [0]
                        }], [[0, 1], 2]],
                        1: [[{
                            7: [0]
                        }], [[0], 2]],
                        2: [[{
                            9: [0],
                        }, {
                            9: [2]
                        }], [[0, 2], 3]],
                    },
                    'output_tensors': {
                        0: [[], [[], 1]],
                        1: [[], [[], 1]],
                        2: [[{
                            9: [0]
                        }], [[0], 1]],
                    },
                    'returns': [7, 9]
                },
            ]
        }

        def _set_attr(input_indices, se_attr, node_names, model):
            attr = OrderedDict()
            attr['dst_shape'] = '-1,-1,-1'
            attr['dims'] = '0,1'
            attr1 = OrderedDict()
            attr1['position'] = util.list2str(input_indices)

            reshape_3d_node_idx = model.get_node_id(node_names[0])
            model.nodes[reshape_3d_node_idx].attr = attr
            expand_indices_node_idx = model.get_node_id(node_names[1])
            model.nodes[expand_indices_node_idx].attr = attr1
            se_node_idx = model.get_node_id(node_names[2])
            model.nodes[se_node_idx].attr = se_attr

        # minilmv2-lat-roberta
        for i in range(len(pattern_mapping_config['RestoreHiddenStatesInLengthAdaptiveUpdateIndices'])):
            pattern_dict = pattern_mapping_config['RestoreHiddenStatesInLengthAdaptiveUpdateIndices'][i]
            model, new_node_names, ret_old_nodes = util.pattern_mapping(
                'RestoreHiddenStatesInLengthAdaptiveUpdateIndices', pattern_dict, model)
            if len(new_node_names) != 0:
                for i in range(len(new_node_names)):
                    attr = OrderedDict()
                    input_indices = []
                    unsqueeze_node = ret_old_nodes[i][0]
                    input_indices.append(int(unsqueeze_node.attr['axes']))
                    se_attr = ret_old_nodes[i][1].attr
                    _set_attr(input_indices, se_attr, new_node_names[i], model)
                    # the first scatter elements operation need the output of embedding layer norm
                    # but its output shape is [bsxseq_len, hidden_size]
                    # so the first scatter node need modify this tensor to 3d tensor
                    # whose shape is [bs, seq_len, hidden_size]
                    reshape_3d_node = model.get_node_by_name(new_node_names[i][0])
                    embedding_ln_out_tensor = copy.deepcopy(reshape_3d_node.output_tensors[0])
                    scatter_node = model.get_node_by_name(new_node_names[i][2])
                    # check if one input tensor is from embedding_layer_norm node
                    if scatter_node.input_tensors[0].name == reshape_3d_node.input_tensors[0].name:
                        model.change_node_input_tensors(new_node_names[i][2], 0,
                            tensor=embedding_ln_out_tensor, mode='modify')
                return model

        return model
