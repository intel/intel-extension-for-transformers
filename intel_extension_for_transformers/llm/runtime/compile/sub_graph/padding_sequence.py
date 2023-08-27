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

"""The PaddingSequence pattern."""

from .pattern import Pattern, pattern_registry
from collections import namedtuple, OrderedDict
import copy
from .. import graph_utils as util
from ..ops.tensor import Tensor


@pattern_registry(pattern_type='PaddingSequence')
class PaddingSequence(Pattern):
    """The PaddingSequence pattern.

    Fuse the original sub-graph into the custom acceleration 'PaddingSequence' graph.
    The search strategy is based on the following pattern mapping configs for different models.
    """
    def __call__(self, model):
        """The __call__ function of this pattern class."""
        pattern_mapping_config = {
            'PaddingSequence': [
                {
                    'patterns': {
                        'in': [[(1, 'Shape'), (2, 'StridedSlice'), (5, 'Pack'), (6, 'Reshape'),
                                (7, 'ExpandDims'), (8, 'Sub'), (9, 'Mul'), (10, ['Add', 'AddV2'])],
                               [(), (3, 'Shape'), (4, 'StridedSlice'), (5, 'Pack')],
                               [(), (0, 'Cast'), (6, 'Reshape')]],
                        'out': [[(0, 'AddV2')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 10
                    },
                    'input_tensors': {
                        0: [[{
                            10: [0]
                        }, {
                            'padding_sequence': [0]
                        }], [[0, 1], 2]]
                    },
                    'output_tensors': {
                        0: [[{
                            10: [0]
                        }], [[0], 1]]
                    },
                    'returns': []
                },

                # bert_base_mrpc
                {
                    'patterns': {
                        'in': [[(0, 'Shape'), (1, 'StridedSlice'), (2, 'Pack'), (3, 'Fill'),
                                (7, 'Mul'), (8, 'ExpandDims'), (9, 'Sub'), (10, 'Mul'),
                                (11, ['Add', 'AddV2'])],
                               [(1, 'StridedSlice'), (4, 'Pack'), (5, 'Reshape'), (6, 'Cast'),
                                (7, 'Mul')]],
                        'out': [[(0, 'AddV2')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 11
                    },
                    'input_tensors': {
                        0: [[{
                            11: [0]
                        }, {
                            'padding_sequence': [0]
                        }], [[0, 1], 2]]
                    },
                    'output_tensors': {
                        0: [[{
                            11: [0]
                        }], [[0], 1]]
                    },
                    'returns': []
                },
                # geminet, bert_related in huggingface
                {
                    'patterns': {
                        'in': [[(0, 'Unsqueeze'), (1, 'Unsqueeze'), (2, 'Cast'), (3, 'Sub'),
                                (4, 'Mul')]],
                        'out': [[(0, 'PaddingSequence')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 4
                    },
                    'input_tensors': {
                        0: [[{
                            'input_data': [-1]
                        }], [[0], 1]]
                    },
                    'output_tensors': {
                        0: [[{
                            4: [0]
                        }], [[0], 1]]
                    },
                    'returns': []
                },
                # distil_bert_base
                {
                    'patterns': {
                        'in': [[(1, 'Unsqueeze'), (3, 'Concat'), (4, 'Reshape'), (6, 'Expand'),
                                (7, 'Cast'), (8, 'Where')], [(), (2, 'Unsqueeze'), (3, 'Concat')],
                               [(), (5, 'Shape'), (6, 'Expand')], [(), (0, 'Equal'),
                                                                   (4, 'Reshape')]],
                        'out': [[(0, 'AddV2')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 8
                    },
                    'input_tensors': {
                        0: [[{
                            5: [0]
                        }, {
                            'padding_sequence': [0]
                        }], [[0, 1], 2]]
                    },
                    'output_tensors': {
                        0: [[{
                            8: [0]
                        }], [[0], 1]]
                    },
                    'returns': [2]
                },

                # distil_bert_base_int8
                {
                    'patterns': {
                        'in': [[(0, 'Unsqueeze'), (1, 'Concat'), (2, 'Reshape'), (3, 'Expand'),
                                (4, 'Where')], [(), (5, 'Unsqueeze'), (1, 'Concat')],
                               [(), (6, 'Shape'), (3, 'Expand')], [(), (7, 'Equal'),
                                                                   (2, 'Reshape')]],
                        'out': [[(0, 'AddV2')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 4
                    },
                    'input_tensors': {
                        0: [[{
                            6: [0]
                        }, {
                            'padding_sequence': [0]
                        }], [[0, 1], 2]]
                    },
                    'output_tensors': {
                        0: [[{
                            4: [0]
                        }], [[0], 1]]
                    },
                    'returns': [5]
                },

                # opennmt encoder
                {
                    'patterns': {
                        'in': [[(0, 'Input'), (1, 'ReduceMax'), (2, 'Cast'), (3, 'Range'),
                                (5, 'Expand'), (6, 'Tile'), (8, 'Less'), (9, 'Unsqueeze'),
                                (10, 'Not'), (11, 'Unsqueeze'), (12, 'Cast'), (13, 'Where')],
                               [(), (4, 'ConstantOfShape'), (5, 'Expand')],
                               [(0, 'Input'), (7, 'Unsqueeze'), (8, 'Less')]],
                        'out': [[(0, 'AddV2')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 13
                    },
                    'input_tensors': {
                        0: [[{
                            13: [2]
                        }, {
                            'padding_sequence': [0]
                        }], [[0, 1], 2]]
                    },
                    'output_tensors': {
                        0: [[{
                            13: [0]
                        }], [[0], 1]]
                    },
                    'returns': [0]
                },

                # opennmt decoder
                {
                    'patterns': {
                        'in': [[(0, 'Input'), (1, 'Shape'), (2, 'Gather'), (3, 'Cast'),
                                (4, 'Range'), (6, 'Expand'), (7, 'Tile'), (9, 'Less'),
                                (10, 'Unsqueeze'), (11, 'Not'), (12, 'Unsqueeze'), (13, 'Cast'),
                                (14, 'Where')],
                               [(), (5, 'ConstantOfShape'), (6, 'Expand')],
                               [(0, 'Input'), (8, 'Unsqueeze'), (9, 'Less')]],
                        'out': [[(0, 'AddV2')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 14
                    },
                    'input_tensors': {
                        0: [[{
                            14: [2]
                        }, {
                            'padding_sequence': [0]
                        }], [[0, 1], 2]]
                    },
                    'output_tensors': {
                        0: [[{
                            14: [0]
                        }], [[0], 1]]
                    },
                    'returns': [0]
                },
            ]
        }

        def _make_padding_sequence_node(tensor_idx, hidden_size, model, seq_len_first=False):
            # Models with different size may have different nums of self-attention head.
            # In Google Bert, the nums_head = hidden_size // 64.
            # But in some other models, like minilm, they has smaller attention head channel.
            # Use this dict to maintain the attr of padding_sequence operation.
            # The last key is for debug and tell the info of unknown model.
            heads_num_dict = {1024: 16, 768: 12, 384: 12, 256: 4, 128: 2, 32: 2, -1: -1}
            node_name = 'padding_sequence'
            heads_num = int(heads_num_dict[hidden_size])
            input_data = model.nodes[0]
            if len(input_data.output_tensors) > tensor_idx:
                input_tensors = [copy.deepcopy(input_data.output_tensors[tensor_idx])]
            else:
                input_tensors = [Tensor()]
            output_tensors = [Tensor(name=node_name + ':0', source_op=[node_name], dest_op=[])]
            attr = OrderedDict()
            if not is_lat_model(model):
                attr['dst_shape'] = '-1,' + str(heads_num) + ',0,-1'
            else:
                attr['dst_shape'] = '-1,' + str(1) + ',1,-1'

            if not seq_len_first:
                attr['dims'] = 1
            else:
                attr['dims'] = 0
                attr['seq_len_first'] = True
            padding_sequence_node = util.construct_node(node_name,
                                                        'PaddingSequence',
                                                        input_tensors=input_tensors,
                                                        output_tensors=output_tensors,
                                                        attr=attr)

            model.insert_nodes(1, [padding_sequence_node])

            return model

        def get_hidden_size(model, p=None, mat_idx=0):
            if p == None:
                p = [[(0, 'MatMul'), (1, ['Add', 'AddV2']), (2, ['Add', 'AddV2']),
                      (3, 'LayerNorm')]]
            match_result = util.search_pattern(p, model)
            if len(match_result) != 0:
                mat_node = model.get_node_by_name(match_result[0][mat_idx])
                hidden_size = int(mat_node.input_tensors[1].shape[-1])
            else:
                hidden_size = -1
            return hidden_size

        def is_lat_model(model, p=None):
            if p == None:
                p = [[(0, 'TopK'),(1, 'Unsqueeze'),(2, 'Unsqueeze'),(3, 'Expand'),
                    (4, 'GatherElements')]]
            match_result = util.search_pattern(p, model)
            return len(match_result) != 0

        pattern_dict = pattern_mapping_config['PaddingSequence'][0]
        model = _make_padding_sequence_node(2, 1024, model)
        model, new_node_names, ret_old_nodes = util.pattern_mapping("PaddingSequence",
                                                                    pattern_dict, model)
        if len(new_node_names) != 0:
            return model
        else:
            model.remove_nodes(['padding_sequence'])

        pattern_dict = pattern_mapping_config['PaddingSequence'][1]
        model = _make_padding_sequence_node(2, 768, model)
        model, new_node_names, ret_old_nodes = util.pattern_mapping("PaddingSequence",
                                                                    pattern_dict, model)
        if len(new_node_names) != 0:
            return model
        else:
            model.remove_nodes(['padding_sequence'])

        pattern_dict = pattern_mapping_config['PaddingSequence'][2]
        hidden_size = get_hidden_size(model)
        if len(model.nodes[0].output_tensors) == 3:
            model = _make_padding_sequence_node(2, hidden_size, model)
        else:
            model = _make_padding_sequence_node(1, hidden_size, model)
        model, new_node_names, ret_old_nodes = util.pattern_mapping("PaddingSequence",
                                                                    pattern_dict, model)

        if len(new_node_names) != 0:
            assert hidden_size!=-1, "Wrong hidden size in padding_sequence!"
            ps_attr = model.get_node_by_name('padding_sequence').attr
            for j in range(len(new_node_names)):
                ps_node_id = model.get_node_id(new_node_names[j][0])
                model.nodes[ps_node_id].attr = ps_attr
            # remove fake node
            model.remove_nodes(['padding_sequence'])
            return model
        else:
            model.remove_nodes(['padding_sequence'])

        pattern_dict = pattern_mapping_config['PaddingSequence'][3]
        model = _make_padding_sequence_node(1, 768, model)
        model, new_node_names, ret_old_nodes = util.pattern_mapping("PaddingSequence",
                                                                    pattern_dict, model)
        if len(new_node_names) != 0:
            # remove shape+gather in distil_bert_base
            for i in range(len(ret_old_nodes)):
                gather_node_name = ret_old_nodes[i][0].input_tensors[0].source_op[0]
                gather_node = model.get_node_by_name(gather_node_name)
                shape_node_name = gather_node.input_tensors[0].source_op[0]
                model.remove_nodes([gather_node_name, shape_node_name])
            return model
        else:
            model.remove_nodes(['padding_sequence'])

        pattern_dict = pattern_mapping_config['PaddingSequence'][4]
        model = _make_padding_sequence_node(1, 768, model)
        model, new_node_names, ret_old_nodes = util.pattern_mapping("PaddingSequence",
                                                                    pattern_dict, model)
        if len(new_node_names) != 0:
            # remove gather in distil_bert_base
            for i in range(len(ret_old_nodes)):
                gather_node_name = ret_old_nodes[i][0].input_tensors[0].source_op[0]
                model.remove_nodes([gather_node_name])
            return model
        else:
            model.remove_nodes(['padding_sequence'])

        for idx in [5, 6]:
            pattern_dict = pattern_mapping_config['PaddingSequence'][idx]
            model = _make_padding_sequence_node(0, 1024, model, seq_len_first=True)
            model, new_node_names, ret_old_nodes = util.pattern_mapping(
                "PaddingSequence", pattern_dict, model)
            if len(new_node_names) != 0:
                assert ret_old_nodes[0][0].op_type == 'Input'
                model.insert_nodes(0, [ret_old_nodes[0][0]])
                return model
            else:
                model.remove_nodes(['padding_sequence'])

        return model
