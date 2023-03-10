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
"""The Transformer2Dmodel_EncoderHiddenStatesReshape pattern."""

from .pattern import Pattern, pattern_registry
from collections import OrderedDict
from .. import graph_utils as util
from ..ops import Tensor


@pattern_registry(pattern_type='Transformer2Dmodel_EncoderHiddenStatesReshape')
class Transformer2Dmodel_EncoderHiddenStatesReshape(Pattern):
    """The Transformer2Dmodel_EncoderHiddenStatesReshape pattern.

    Fuse the original sub-graph into the custom acceleration 'Transformer2Dmodel_EncoderHiddenStatesReshape' graph.
    The search strategy is based on the following pattern mapping configs for the stable textEncoderV1-5.
    """
    def __call__(self, model):
        """The __call__ function of this pattern class."""
        pattern_mapping_config = {
            'Transformer2Dmodel_EncoderHiddenStatesReshape': [
                {
                    'patterns': {
                        'in': [[(0, 'Input'), (1, 'MatMul')]],
                    },
                },
            ]
        }

        pattern = pattern_mapping_config['Transformer2Dmodel_EncoderHiddenStatesReshape'][0]['patterns']['in']
        patterns_nodes_name = util.search_pattern(pattern, model)
        if len(patterns_nodes_name) != 0:
            for j in range(len(patterns_nodes_name)):

                input_node = model.get_node_by_name(patterns_nodes_name[j][0])
                encoder_hidden_states_tensor = input_node.output_tensors[2]

                matmul_node_name = patterns_nodes_name[j][1]
                matmul_node = model.get_node_by_name(matmul_node_name)

                matmul_node_idx = model.get_node_id(matmul_node.name)

                #create a new 2d reshape node
                new_node_name = 'encoder_hidden_states/reshape_2d'
                new_node_output_tensor_name = encoder_hidden_states_tensor.name + '_2d'

                input_tensors = [encoder_hidden_states_tensor]
                output_tensor = [
                    Tensor(name=new_node_output_tensor_name,
                           source_op=[new_node_name],
                           dest_op=[matmul_node.name],
                           dtype=matmul_node.output_tensors[0].dtype)
                ]
                new_node = util.construct_node(node_name=new_node_name,
                                               op_type='Reshape',
                                               input_tensors=input_tensors,
                                               output_tensors=output_tensor)
                attr = OrderedDict()
                attr['dst_shape'] = '-1,768'
                new_node.attr = attr

                model.insert_nodes(matmul_node_idx, [new_node])
                matmul_node.input_tensors[0] = new_node.output_tensors[0]

        return model
