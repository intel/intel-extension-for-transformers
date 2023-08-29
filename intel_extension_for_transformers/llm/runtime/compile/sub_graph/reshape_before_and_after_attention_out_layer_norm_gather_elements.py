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

"""The ReshapeBeforeAndAfterAttentionOutLayerNormGatherElements Pattern."""

from .pattern import Pattern, pattern_registry
from collections import namedtuple, OrderedDict
from .. import graph_utils as util
import numpy as np
import copy


@pattern_registry(pattern_type='ReshapeBeforeAndAfterAttentionOutLayerNormGatherElements')
class ReshapeBeforeAndAfterAttentionOutLayerNormGatherElements(Pattern):
    """The ReshapeBeforeAndAfterAttentionOutLayerNormGatherElements pattern.

    Fuse the original sub-graph into the custom acceleration graph.
    The search strategy is based on the following pattern mapping configs for different models.
    """
    def __call__(self, model):
        """The __call__ function of this pattern class."""
        pattern_mapping_config = {
            'ReshapeBeforeAndAfterAttentionOutLayerNormGatherElements': [
                # minilmv2-lat-roberta
                {
                    'patterns': {
                        'in': [[(0, 'LayerNorm'), (1, 'ExpandIndices'), (2, 'GatherElements')]],
                        'out': [[(0, 'LayerNorm'), (1, 'Reshape'), (2, 'ExpandIndices'),
                                 (3, 'GatherElements'), (4, 'Reshape')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 0,
                        1: 'reshape_to_3d_after_layer_norm_in_attention',
                        2: 1,
                        3: 2,
                        4: 'reshape_to_2d_after_gather_elements_in_attention',
                    },
                    'input_tensors': {
                        0: [[{
                            0: [0]
                        }, {
                            0: [1]
                        }, {
                            0: [2]
                        }], [[0, 1, 2], 3]],
                        1: [[{
                            'input_data': [0]
                        }], [[1], 2]],
                        2: [[{
                            1: [0]
                        }], [[0], 2]],
                        3: [[{
                            2: [0]
                        }], [[0], 2]],
                        4: [[], [[], 1]],
                    },
                    'output_tensors': {
                        0: [[], [[], 1]],
                        1: [[{
                            0: [0]
                        }], [[0], 1]],
                        2: [[], [[], 1]],
                        3: [[], [[], 1]],
                        4: [[{
                            2: [0]
                        }], [[0], 1]],
                    },
                    'returns': [0, 1, 2]
                },
            ]
        }

        def _set_attr(ln_attr, ki_attr, ge_attr, hidden_size, node_names, model):
            attr1 = OrderedDict()
            attr1['dst_shape'] = '-1,-1,' + str(hidden_size)
            attr1['dims'] = '0'
            attr2 = OrderedDict()
            attr2['dst_shape'] = '-1,' + str(hidden_size)

            ln_node_idx = model.get_node_id(node_names[0])
            model.nodes[ln_node_idx].attr = ln_attr
            reshape_3d_node_idx = model.get_node_id(node_names[1])
            model.nodes[reshape_3d_node_idx].attr = attr1
            keep_indices_node_idx = model.get_node_id(node_names[2])
            model.nodes[keep_indices_node_idx].attr = ki_attr
            gather_elements_node_idx = model.get_node_id(node_names[3])
            model.nodes[gather_elements_node_idx].attr = ge_attr
            reshape_2d_node_idx = model.get_node_id(node_names[4])
            model.nodes[reshape_2d_node_idx].attr = attr2

        # minilmv2-lat-roberta
        pattern_dict = pattern_mapping_config[
            'ReshapeBeforeAndAfterAttentionOutLayerNormGatherElements'][0]
        model, new_node_names, ret_old_nodes = util.pattern_mapping(
            'ReshapeBeforeAndAfterAttentionOutLayerNormGatherElements', pattern_dict, model)
        if len(new_node_names) != 0:
            for i in range(len(new_node_names)):
                hidden_size = int(ret_old_nodes[i][0].input_tensors[-1].shape[0])
                ln_attr = ret_old_nodes[i][0].attr
                ki_attr = ret_old_nodes[i][1].attr
                ge_attr = ret_old_nodes[i][2].attr
                _set_attr(ln_attr, ki_attr, ge_attr, hidden_size, new_node_names[i], model)

            return model

        return model
