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

"""The TorchPaddingSequence pattern."""

from .pattern import Pattern, pattern_registry
from collections import namedtuple, OrderedDict
import copy
from .. import graph_utils as util
from ..ops.tensor import Tensor


@pattern_registry(pattern_type='TorchPaddingSequence')
class TorchPaddingSequence(Pattern):
    """The TorchPaddingSequence pattern.

    Fuse the original sub-graph into the custom acceleration 'TorchPaddingSequence' graph.
    The search strategy is based on the following pattern mapping configs for different models.
    """
    def __call__(self, model):
        """The __call__ function of this pattern class."""
        pattern_mapping_config = {
            'TorchPaddingSequence': [
                {
                    'patterns': {
                        'in': [[(0, 'View'), (1, 'Slice'), (2, 'Unsqueeze'), (3, 'Unsqueeze'),
                                (4, 'Slice'), (5, 'Rsub'), (6, 'Mul')]
                               ],
                        'out': [[(0, 'PaddingSequence')]]
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
                }

            ]
        }

        def _make_padding_sequence_node(tensor_idx, hidden_size, model):
            # Models with different size may have different nums of self-attention head.
            # In Google Bert, the nums_head = hidden_size // 64.
            # But in some other models, like minilm, they has smaller attention head channel.
            # Use this dict to maintain the attr of padding_sequence operation.
            # The last key is for debug and tell the info of unknown model.
            heads_num_dict = {4096: 16, 1024: 16, 768: 12, 384: 12, 256: 4, 128: 2, -1: -1}
            node_name = 'padding_sequence'
            heads_num = int(heads_num_dict[hidden_size])
            input_data = model.nodes[0]
            if len(input_data.output_tensors) > tensor_idx:
                input_tensors = [copy.deepcopy(input_data.output_tensors[tensor_idx])]
            else:
                input_tensors = [Tensor()]
            output_tensors = [Tensor(name=node_name + ':0', source_op=[node_name], dest_op=[])]
            attr = OrderedDict()
            attr['dst_shape'] = '-1,' + str(heads_num) + ',0,-1'
            attr['dims'] = 1
            padding_sequence_node = util.construct_node(node_name,
                                                        'PaddingSequence',
                                                        input_tensors=input_tensors,
                                                        output_tensors=output_tensors,
                                                        attr=attr)

            model.insert_nodes(1, [padding_sequence_node])

            return model

        def get_hidden_size(model, p=None, mat_idx=0):
            if p == None:
                p = [[(0, 'InnerProduct'), (1, 'Pow')]]
            match_result = util.search_pattern(p, model)
            if len(match_result) != 0:
                mat_node = model.get_node_by_name(match_result[0][mat_idx])
                # import pdb;pdb.set_trace()
                hidden_size = 4096
                #hidden_size = int(mat_node.input_tensors[1].shape[-1])
                
                #hidden_size = 768
            else:
                hidden_size = -1
            return hidden_size

        if model.framework_modeling_config['framework'] == 'torch':
            pattern_dict = pattern_mapping_config['TorchPaddingSequence'][0]
            hidden_size = get_hidden_size(model)
            model = _make_padding_sequence_node(1, hidden_size, model)
            model, new_node_names, ret_old_nodes = util.pattern_mapping("TorchPaddingSequence",
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


        return model
