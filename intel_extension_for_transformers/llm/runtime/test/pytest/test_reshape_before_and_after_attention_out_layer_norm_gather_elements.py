#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2022 Intel Corporation
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

import unittest
from collections import OrderedDict
from intel_extension_for_transformers.backends.neural_engine.compile.ops.op import OPERATORS, Operator
from intel_extension_for_transformers.backends.neural_engine.compile.ops.tensor import Tensor
from intel_extension_for_transformers.backends.neural_engine.compile.graph import Graph
from intel_extension_for_transformers.backends.neural_engine.compile.sub_graph.reshape_before_and_after_attention_out_layer_norm_gather_elements import ReshapeBeforeAndAfterAttentionOutLayerNormGatherElements
import numpy as np


class TestAttentionReshape(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        pass

    @classmethod
    def tearDownClass(self):
        pass

    def test_attention_reshape_0(self):
        graph = Graph()
        graph.framework_modeling_config['framework'] = 'onnxruntime'
        input_data_node = OPERATORS['Input']()
        input_tensors = []
        output_tensors = [Tensor(name='input_data'), Tensor(), Tensor()]
        input_data_node.construct('input_data', 'Input', input_tensors=input_tensors,
                                output_tensors=output_tensors)

        layernorm_node = OPERATORS['LayerNorm']()
        input_tensors = [Tensor(name = 'input59', data=np.array(1), shape=[1,12,384,384]),
                Tensor(name='weight', data=np.array(1), shape=[1,12,384,384]), Tensor(name='bias', data=np.array(1), shape=[1,12,384,384])]
        output_tensors = [Tensor(name='layernorm:0', source_op=['layernorm'], dest_op=['expandindices']),
                Tensor(name='layernorm:0', source_op=['layernorm'], dest_op=['gatherelements'])]
        layernorm_node.construct('layernorm', 'LayerNorm', input_tensors=input_tensors,
                                output_tensors=output_tensors, attr=OrderedDict({
                                    'epsilon': '9.99e-06'}))

        expandindices_node = OPERATORS['ExpandIndices']()
        input_tensors = [Tensor(name='expandindices0', data=np.array(1)),
                Tensor(name='layernorm:0', source_op=['layernorm'], dest_op=['expandindices'])]
        output_tensors = [Tensor(name='expandindices:0', source_op=['expandindices'], dest_op=['gatherelements'])]
        expandindices_node.construct('expandindices', 'ExpandIndices', input_tensors=input_tensors,
                                output_tensors=output_tensors, attr=OrderedDict({
                                    'position': '-1'}))

        gatherelements_node = OPERATORS['GatherElements']()
        input_tensors = [Tensor(name='layernorm:0', source_op=['layernorm'], dest_op=['gatherelements']),
            Tensor(name='expandindices:0', source_op=['expandindices'], dest_op=['gatherelements'])]
        output_tensors = [Tensor(name='gatherelements:0', source_op=['gatherelements'], dest_op=[])]
        gatherelements_node.construct('gatherelements', 'GatherElements', input_tensors=input_tensors,
                                output_tensors=output_tensors, attr=OrderedDict({
                                    'axis': '1'}))


        graph.insert_nodes(len(graph.nodes), [input_data_node, layernorm_node, expandindices_node, gatherelements_node])

        graph = ReshapeBeforeAndAfterAttentionOutLayerNormGatherElements()(graph)
        self.assertEqual(6, len(graph.nodes))
        self.assertEqual('reshape_to_3d_after_layer_norm_in_attention', graph.nodes[2].name)
        self.assertEqual('reshape_to_2d_after_gather_elements_in_attention', graph.nodes[5].name)


if __name__ == "__main__":
    unittest.main()
