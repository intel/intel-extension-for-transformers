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
from intel_extension_for_transformers.backends.neural_engine.compile.sub_graph.attention_output_layer_norm_length_adaptive_keep_indices import AttentionOutputLayerNormLengthAdaptiveExpandIndices
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

        constantofshape_node = OPERATORS['ConstantOfShape']()
        input_tensors = [Tensor(name = 'input0', data=np.array(1), shape=[1])]
        output_tensors = [Tensor(name='output0', source_op=['constantofshape'], dest_op=['mul']),
                          Tensor(name='constantofshape_output0', source_op=['constantofshape'], dest_op=['where'])]

        constantofshape_node.construct('constantofshape', 'ConstantOfShape', input_tensors=input_tensors,
                                output_tensors=output_tensors)

        mul_node = OPERATORS['Mul']()

        input_tensors =  [Tensor(name='output0', source_op=['constantofshape'], dest_op=['mul']),
                          Tensor(name ='mul_input1', data=np.array(1), shape=[1])]
        output_tensors = [Tensor(name='mul_output', source_op=['mul'], dest_op=['equal'])]
        mul_node.construct('mul', 'Mul', input_tensors=input_tensors, output_tensors=output_tensors)

        equal_node = OPERATORS['Equal']()
        input_tensors =  [Tensor(name ='equal_input0', data=np.array([-1,1,1,-1]), shape=[4]),
                Tensor(name='mul_output', source_op=['mul'], dest_op=['equal'])]
        output_tensors = [Tensor(name='equal_output', source_op=['equal'], dest_op=['where'])]
        equal_node.construct('equal', 'Equal', input_tensors=input_tensors, output_tensors=output_tensors)

        where_node = OPERATORS['Where']()
        input_tensors = [Tensor(name='equal_output', source_op=['equal'], dest_op=['where']),
                         Tensor(name='constantofshape_output0', source_op=['constantofshape'],dest_op=['where']),
                         Tensor(name ='where_input2', data=np.array([-1,1,1,-1]), shape=[1])]
        output_tensors = [Tensor(name='where_output', source_op=['where'], dest_op=['expand'])]
        where_node.construct('where', 'Where', input_tensors=input_tensors, output_tensors=output_tensors)


        unsqueeze_node = OPERATORS['Unsqueeze']()
        input_tensors = [Tensor(name ='unsqueeze_input0', data=np.array(1), shape=[2,12,384,384]),
                         Tensor(name='unsqueeze_input1', data=np.array(1), shape=[1])]
        output_tensors = [Tensor(name='unsqueeze_output', source_op=['unsqueeze'], dest_op=['expand'])]
        unsqueeze_node.construct('unsqueeze', 'Unsqueeze', input_tensors=input_tensors,
                                output_tensors=output_tensors,attr=OrderedDict({'axes': '1'}))

        expand_node = OPERATORS['Expand']()
        input_tensors = [Tensor(name='unsqueeze_output', source_op=['unsqueeze'], dest_op=['expand']),
                         Tensor(name='where_output', source_op=['where'], dest_op=['expand'])]
        output_tensors = [Tensor(name='expand_output', source_op=['expand'], dest_op=['gatherelements'])]
        expand_node.construct('expand', 'Expand', input_tensors=input_tensors,
                                output_tensors=output_tensors)

        gatherelements_node = OPERATORS['GatherElements']()
        input_tensors = [Tensor(name='gatherelements_input0',data = np.array(1),shape=[1]),
                         Tensor(name='expand_output', source_op=['expand'], dest_op=['gatherelements'])]
        output_tensors = [Tensor(name='gatherelements_output', source_op=['gatherelements'], dest_op=['empty'])]
        gatherelements_node.construct('gatherelements', 'GatherElements', input_tensors=input_tensors, output_tensors=output_tensors, attr=OrderedDict({'axis': '1'}))

        graph.insert_nodes(len(graph.nodes), [input_data_node, constantofshape_node, mul_node, equal_node, where_node,unsqueeze_node, expand_node,gatherelements_node])

        graph = AttentionOutputLayerNormLengthAdaptiveExpandIndices()(graph)

        self.assertEqual(3, len(graph.nodes))
        self.assertEqual('expand', graph.nodes[1].name)


if __name__ == "__main__":
    unittest.main()
