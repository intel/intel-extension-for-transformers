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
from intel_extension_for_transformers.backends.neural_engine.compile.sub_graph.qkv_reshape import QKVReshape


class TestAttentionReshape(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        pass

    @classmethod
    def tearDownClass(self):
        pass
    
    def test_attention_reshape_1(self):
        graph = Graph()
        graph.framework_modeling_config['framework'] = 'onnxruntime'
        input_data_node = OPERATORS['Input']()
        input_tensors = []
        output_tensors = [Tensor(), Tensor(), Tensor()]
        input_data_node.construct('input_data', 'Input', input_tensors=input_tensors, 
                                output_tensors=output_tensors)

        matmul_node = OPERATORS['MatMulWithBias']()
        input_tensors = [Tensor(), Tensor(), Tensor()]
        output_tensors = [Tensor(name='matmul:0', source_op=['matmul'],
                                dest_op=['reshape'])]
        matmul_node.construct('matmul', 'MatMulWithBias', input_tensors=input_tensors, 
                                output_tensors=output_tensors, attr=OrderedDict(
                                    {'src1_perm': '0,1'}))
        
        reshape_node = OPERATORS['Reshape']()
        input_tensors = [Tensor(name='matmul:0', source_op=['matmul'],
                                dest_op=['reshape'])]
        output_tensors = [Tensor(name='reshape:0', source_op=['reshape'], dest_op=[])]
        reshape_node.construct('reshape', 'Reshape', input_tensors=input_tensors, 
                                output_tensors=output_tensors, attr=OrderedDict({
                                    'dst_shape': '0,0,12,64'}))
        
        graph.insert_nodes(len(graph.nodes), [input_data_node, matmul_node, reshape_node])
        graph = QKVReshape()(graph)
        self.assertEqual(3, len(graph.nodes))
        self.assertEqual('0,1', graph.nodes[1].attr['src1_perm'])
        self.assertEqual('-1,-1,12,64', graph.nodes[2].attr['dst_shape'])


if __name__ == "__main__":
    unittest.main()
