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
from intel_extension_for_transformers.backends.neural_engine.compile.sub_graph.gelu import Gelu


class TestGelu(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        pass

    @classmethod
    def tearDownClass(self):
        pass
    
    def test_gelu_1(self):
        graph = Graph()
        graph.framework_modeling_config['framework'] = 'onnxruntime'
        input_data_node = OPERATORS['Input']()
        input_tensors = []
        output_tensors = [Tensor(), Tensor(), Tensor()]
        input_data_node.construct('input_data', 'Input', input_tensors=input_tensors, 
                                output_tensors=output_tensors)

        div_node = OPERATORS['Div']()
        input_tensors = [Tensor()]
        output_tensors = [Tensor(name='div:0', source_op=['div'], dest_op=['erf'])]
        div_node.construct('div', 'Div', input_tensors=input_tensors, 
                                output_tensors=output_tensors)
        
        erf_node = OPERATORS['Erf']()
        input_tensors = [Tensor(name='div:0', source_op=['div'], dest_op=['erf'])]
        output_tensors = [Tensor(name='erf:0', source_op=['erf'], dest_op=['add'])]
        erf_node.construct('erf', 'Erf', input_tensors=input_tensors, 
                                output_tensors=output_tensors)
        
        add_node = OPERATORS['Add']()
        input_tensors = [Tensor(name='erf:0', source_op=['erf'], dest_op=['add'])]
        output_tensors = [Tensor(name='add:0', source_op=['add'], dest_op=['mul_1'])]
        add_node.construct('add', 'Add', input_tensors=input_tensors, 
                                output_tensors=output_tensors)
        
        mul_1_node = OPERATORS['Mul']()
        input_tensors = [Tensor(name='add:0', source_op=['add'], dest_op=['mul_1'])]
        output_tensors = [Tensor(name='mul_1:0', source_op=['mul_1'], dest_op=['mul_2'])]
        mul_1_node.construct('mul_1', 'Mul', input_tensors=input_tensors, 
                                output_tensors=output_tensors)
        
        mul_2_node = OPERATORS['Mul']()
        input_tensors = [Tensor(name='mul_1:0', source_op=['mul_1'], dest_op=['mul_2'])]
        output_tensors = [Tensor(name='mul_2:0', source_op=['mul_2'], dest_op=[])]
        mul_2_node.construct('mul_2', 'Mul', input_tensors=input_tensors, 
                                output_tensors=output_tensors)
        
        graph.insert_nodes(len(graph.nodes), [input_data_node, div_node, erf_node, add_node,
                                                mul_1_node, mul_2_node])
        graph = Gelu()(graph)
        self.assertEqual(2, len(graph.nodes))
        self.assertEqual('mul_2', graph.nodes[-1].name)
        self.assertEqual('Gelu', graph.nodes[-1].op_type)


if __name__ == "__main__":
    unittest.main()
