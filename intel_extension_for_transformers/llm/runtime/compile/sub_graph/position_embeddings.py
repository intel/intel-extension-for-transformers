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

"""The PositionEmbeddings pattern."""

from .pattern import Pattern, pattern_registry
from collections import namedtuple, OrderedDict
from .. import graph_utils as util


@pattern_registry(pattern_type='PositionEmbeddings')
class PositionEmbeddings(Pattern):
    """The PositionEmbeddings pattern.

    Fuse the original sub-graph into the custom acceleration 'PositionEmbeddings' graph.
    The search strategy is based on the following pattern mapping configs for different models.
    """
    def __call__(self, model):
        """The __call__ function of this pattern class."""
        pattern_mapping_config = {
            'PositionEmbeddings': [
                {
                    'patterns': {
                        'in': [[(0, 'Shape'), (1, 'StridedSlice'), (2, 'Pack'), (3, 'Slice'),
                                (5, 'Reshape')], [(1, 'StridedSlice'), (4, 'Pack'),
                                                  (5, 'Reshape')]],
                        'out': [[(0, 'Reshape'), (1, 'Reshape')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 'position_embeddings/after/reshape',
                        1: 5
                    },
                    'input_tensors': {
                        0: [[{
                            3: [0]
                        }, {
                            'input_data': [0]
                        }], [[0, 1], 2]],
                        1: [[], [[], 1]]
                    },
                    'output_tensors': {
                        0: [[], [[], 1]],
                        1: [[{
                            5: [0]
                        }], [[0], 1]]
                    },
                    'returns': [3]
                },

                # distil_bert_base
                {
                    'patterns': {
                        'in': [[(0, 'Shape'), (1, 'Gather'), (2, 'Cast'), (3, 'Range'),
                                (4, 'Unsqueeze'), (6, 'Expand'), (7, 'Gather')],
                                [(), (5, 'Shape'), (6, 'Expand')]],
                        'out': [[(0, 'Reshape'), (1, 'Reshape')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 'position_embeddings/after/reshape',
                        1: 7
                    },
                    'input_tensors': {
                        0: [[{
                            7: [0]
                        }, {
                            'input_data': [0]
                        }], [[0, 1], 2]],
                        1: [[], [[], 1]]
                    },
                    'output_tensors': {
                        0: [[], [[], 1]],
                        1: [[{
                            7: [0]
                        }], [[0], 1]]
                    },
                    'returns': [7]
                },

                # geminet
                {
                    'patterns': {
                        'in': [[(0, 'Shape'), (1, 'Gather'), (2, 'Unsqueeze'),
                                (9, 'ConstantOfShape'), (10, 'NonZero'), (11, 'Transpose'),
                                (12, 'Squeeze'), (13, 'Unsqueeze'), (14, 'Expand'),
                                (15, 'Gather')],
                                [(0, 'Shape'), (3, 'Gather'), (4, 'Unsqueeze'), (5, 'Concat'),
                                (6, 'Reshape'), (7, 'Equal'), (8, 'Where'), (14, 'Expand')]],
                        'out': [[(0, 'Reshape'), (1, 'Reshape')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 'position_embeddings/after/reshape',
                        1: 15
                    },
                    'input_tensors': {
                        0: [[{
                            15: [0]
                        }, {
                            'input_data': [0]
                        }], [[0, 1], 2]],
                        1: [[], [[], 1]]
                    },
                    'output_tensors': {
                        0: [[], [[], 1]],
                        1: [[{
                            15: [0]
                        }], [[0], 1]]
                    },
                    'returns': [15]
                },

                # bert_base_sparse
                {
                    'patterns': {
                        'in': [[(0, 'Shape'), (1, 'Gather'), (2, 'Add'), (3, 'Unsqueeze'),
                                (4, 'Slice'), (5, 'Gather')]],
                        'out': [[(0, 'Reshape'), (1, 'Reshape')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 'position_embeddings/after/reshape',
                        1: 5
                    },
                    'input_tensors': {
                        0: [[{
                            5: [0]
                        }, {
                            'input_data': [0]
                        }], [[0, 1], 2]],
                        1: [[], [[], 1]]
                    },
                    'output_tensors': {
                        0: [[], [[], 1]],
                        1: [[{
                            5: [0]
                        }], [[0], 1]]
                    },
                    'returns': [5]
                },

                # bert_mini_int8
                {
                    'patterns': {
                        'in': [[(0, 'Shape'), (1, 'Gather'), (2, 'Unsqueeze'),
                                (3, 'Slice'), (4, 'Gather')]],
                        'out': [[(0, 'Reshape'), (1, 'Reshape')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 'position_embeddings/after/reshape',
                        1: 4
                    },
                    'input_tensors': {
                        0: [[{
                            4: [0]
                        }, {
                            'input_data': [0]
                        }], [[0, 1], 2]],
                        1: [[], [[], 1]]
                    },
                    'output_tensors': {
                        0: [[], [[], 1]],
                        1: [[{
                            4: [0]
                        }], [[0], 1]]
                    },
                    'returns': [4]
                },

                # opennmt encoder
                {
                    'patterns': {
                        'in': [[(0, 'Shape'), (1, 'Gather'), (2, 'Unsqueeze'), (3, 'Slice'),
                                (4, 'Add')]],
                        'out': [[(0, 'Reshape'), (1, 'Add')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 3,
                        1: 4,
                    },
                    'input_tensors': {
                        0: [[{
                            3: [0]
                        }, {
                            'input_data': [0]
                        }], [[0, 1], 2]],
                        1: [[{
                            4: [0]
                        }], [[0], 2]]
                    },
                    'output_tensors': {
                        0: [[], [[], 1]],
                        1: [[{
                            4: [0]
                        }], [[0], 1]]
                    },
                    'returns': [3]
                },

                # bert_base_mrpc
                {
                    'patterns': {
                        'in': [[(0, 'Slice'), (1, 'Reshape')]],
                        'out': [[(0, 'Reshape'), (1, 'Reshape')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 'position_embeddings/after/reshape',
                        1: 1
                    },
                    'input_tensors': {
                        0: [[{
                            0: [0]
                        }, {
                            'input_data': [0]
                        }], [[0, 1], 2]],
                        1: [[], [[], 1]]
                    },
                    'output_tensors': {
                        0: [[], [[], 1]],
                        1: [[{
                            1: [0]
                        }], [[0], 1]]
                    },
                    'returns': [0]
                },
            ]
        }

        def _set_attr(hidden_size, node_names, model, batch_idx=0):
            attr1 = OrderedDict()
            if batch_idx == 0:
                attr1['dst_shape'] = '1,-1,' + str(hidden_size)
                attr1['dims'] = 1
            else:
                attr1['dst_shape'] = '-1,1,' + str(hidden_size)
                attr1['dims'] = 0
            if batch_idx == 0:
                attr2 = OrderedDict()
                attr2['dst_shape'] = '1,-1'

            reshape_0_node_idx = model.get_node_id(node_names[0])
            model.nodes[reshape_0_node_idx].attr = attr1

            if batch_idx == 0:
                reshape_1_node_idx = model.get_node_id(node_names[1])
                model.nodes[reshape_1_node_idx].attr = attr2

        def _remove_assert(pattern, model):
            rm_rets = util.search_pattern(pattern, model)
            for ret in rm_rets:
                model.remove_nodes(ret[:-1])
            return model

        for i in range(0, len(pattern_mapping_config['PositionEmbeddings'])-1):
            pattern_dict = pattern_mapping_config['PositionEmbeddings'][i]
            model, new_node_names, ret_old_nodes = util.pattern_mapping("PositionEmbeddings", 
                                                                        pattern_dict, model)
            if len(new_node_names) != 0:
                for j in range(len(new_node_names)):
                    slice_node = ret_old_nodes[j][0]
                    hidden_size = int(slice_node.input_tensors[0].shape[-1])
                    batch_idx = 0
                    if model.get_node_by_name(new_node_names[j][0]).op_type == "Reshape" and \
                        model.get_node_by_name(new_node_names[j][1]).op_type != "Reshape":
                        batch_idx = 1
                    _set_attr(hidden_size, new_node_names[j], model, batch_idx=batch_idx)

                return model

        # bert_base_mrpc
        pattern_dict = pattern_mapping_config['PositionEmbeddings'][-1]
        model, new_node_names, ret_old_nodes = util.pattern_mapping("PositionEmbeddings", 
                                                                    pattern_dict, model)
        if len(new_node_names) != 0:
            for i in range(len(new_node_names)):
                slice_node = ret_old_nodes[i][0]
                hidden_size = int(slice_node.input_tensors[0].shape[-1])
                _set_attr(hidden_size, new_node_names[i], model)
            p = [[(0, 'LessEqual'), (1, 'All'), (2, 'Assert')]]
            model = _remove_assert(p, model)

            return model

        return model
