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
from intel_extension_for_transformers.backends.neural_engine.compile.sub_graph.matmul_with_bias_relu import MatMulWithBiasRelu
import numpy as np


class TestMatmulWithBiasRelu(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        pass

    @classmethod
    def tearDownClass(self):
        pass
    
    def test_matmul_with_bias_relu_1(self):
        graph = Graph()
        graph.framework_modeling_config['framework'] = 'onnxruntime'
        input_data_node = OPERATORS['Input']()
        input_tensors = []
        output_tensors = [Tensor(), Tensor(), Tensor()]
        input_data_node.construct('input_data', 'Input', input_tensors=input_tensors, 
                                output_tensors=output_tensors)

        mat_node = OPERATORS['MatMulWithBias']()
        input_tensors = [Tensor(data=np.array(1)), Tensor(data=np.array(1)), 
                            Tensor(data=np.array(1))]
        output_tensors = [Tensor(name='matmul:0', source_op=['matmul'], 
                                    dest_op=['relu'])]
        mat_node.construct('matmul', 'MatMulWithBias', input_tensors=input_tensors, 
                                output_tensors=output_tensors, attr=OrderedDict({
                                    'src1_perm': '1,0'}))
        
        tanh_node = OPERATORS['Relu']()
        input_tensors = [Tensor(name='matmul:0', source_op=['matmul'], 
                                    dest_op=['relu'])]
        output_tensors = [Tensor(name='relu:0', source_op=['relu'],
                                dest_op=[])]
        tanh_node.construct('relu', 'Relu', input_tensors=input_tensors, 
                                output_tensors=output_tensors)
        
        graph.insert_nodes(len(graph.nodes), [input_data_node, mat_node, tanh_node])
        graph = MatMulWithBiasRelu()(graph)
        self.assertEqual(2, len(graph.nodes))
        self.assertEqual('1,0', graph.nodes[1].attr['src1_perm'])
        self.assertEqual('relu', graph.nodes[1].name)
        self.assertEqual('relu', graph.nodes[1].attr['append_op'])


if __name__ == "__main__":
    unittest.main()
