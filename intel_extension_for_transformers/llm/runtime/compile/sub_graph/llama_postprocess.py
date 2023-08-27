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

"""The LlamaPostprocess pattern."""

from .pattern import Pattern, pattern_registry
from collections import namedtuple, OrderedDict
from .. import graph_utils as util
from ..ops import Tensor
import numpy as np


@pattern_registry(pattern_type='LlamaPostprocess')
class LlamaPostprocess(Pattern):
    """The LlamaPostprocess pattern.

    Fuse the original sub-graph into the custom acceleration 'LlamaPostprocess' graph.
    The search strategy is based on the following pattern mapping configs for different models.
    """
    def __call__(self, model):
        """The __call__ function of this pattern class."""
        pattern_mapping_config = {
            'LlamaPostprocess': [
                # llama postprocess
                {
                    'patterns': {
                        'in': [[(0, 'Gather'), (1, 'RmsNorm')]
                              ],
                        'out': [[(0, 'Reshape'), (1, 'Gather'),(2, 'Reshape'),
                                 (3, 'RmsNorm')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 'llama_embeddings_before/reshape',
                        1: 0,
                        2: 'llama_embeddings_after/reshape',
                        3: 1
                    },
                    'input_tensors': {
                        0: [[{
                            0: [1]
                        }], [[0], 1]],
                        1: [[{
                            0: [0]
                        }], [[1], 2]],
                        2: [[{
                            0: [1]
                        }], [[1], 2]],
                        3: [[{
                            1: [1]
                        }], [[1], 2]]
                    },
                    'output_tensors': {
                        0: [[], [[], 1]],
                        1: [[], [[], 1]],
                        2: [[], [[], 1]],
                        3: [[{
                            1: [0]
                        }], [[0], 1]],
                    },
                    'returns': [1, 0]
                }
            ]
        }

        # pattern_dict = pattern_mapping_config['LlamaPostprocess'][0]
        # model, new_node_names, ret_old_nodes = \
        #     util.pattern_mapping('LlamaPostprocess', pattern_dict, model)
        # if len(new_node_names) != 0:
        #     for i in range(len(new_node_names)):
        #         reshape_node1 = model.get_node_by_name(new_node_names[i][0])
        #         reshape_node1.attr = OrderedDict({'dst_shape': '-1'})
        #         reshape_node2 = model.get_node_by_name(new_node_names[i][2])
        #         reshape_node2.attr = OrderedDict({'dst_shape': '-1, -1, -1', 'dims': '0, 1',
        #                                           'mul': '0, 1'})
        #         gather_node = model.get_node_by_name(new_node_names[i][1])
        #         gather_node.attr = OrderedDict({'axis': '0', 'batch_dims': '0'})
        #         binary_node = model.get_node_by_name(ret_old_nodes[i][1].output_tensors[0].dest_op[0])
        #         binary_node.input_tensors[0] = reshape_node2.output_tensors[0]
        #         rmsnorm_node = model.get_node_by_name(new_node_names[i][3])
        #         rmsnorm_node.attr = ret_old_nodes[i][0].attr

        remove_shape = []
        for node in model.nodes:
            if node.op_type == 'Output':
                pre_node = model.get_node_by_name(node.input_tensors[0].source_op[0])
                output_name = node.input_tensors[0].name
                reshape_output = Tensor(name=output_name + "_reshape",
                                    source_op=[node.name + "_reshape_node"],
                                    dest_op=[],
                                    dtype='fp32')
                reshape_op = util.construct_node(
                    node_name=node.name + "_reshape_node",
                    op_type='Reshape',
                    input_tensors=[node.input_tensors[0],
                                   model.get_node_by_name('input_data').output_tensors[0]],
                    output_tensors=[reshape_output],
                    attr=OrderedDict({'dst_shape': '-1,-1,32000', 'dims': '0,1'}))
                insert_idx = model.get_node_id(node.name)
                model.insert_nodes(insert_idx, [reshape_op])
                node.input_tensors[0] = reshape_output
                break
            
            if node.op_type == 'RmsNorm':
                pre_node = model.get_node_by_name(node.input_tensors[0].source_op[0])
                if 'output_dtype' in pre_node.attr:
                    pre_node.attr['output_dtype'] = 'fp32'
                if pre_node.op_type == "Reshape":
                    prepre_node = model.get_node_by_name(pre_node.input_tensors[0].source_op[0])
                    prepre_node.attr = None
                    prepre_node.attr = OrderedDict({'axis': '0', 'batch_dims':'0'})
                    prepre_node = model.get_node_by_name(prepre_node.input_tensors[0].source_op[0])
                    prepre_node.attr = None
                    prepre_node.attr = OrderedDict({'dst_shape': '-1'})
            
            if node.op_type == 'Shape':
                remove_shape.append(node.name)
        model.remove_nodes(remove_shape)        
        return model