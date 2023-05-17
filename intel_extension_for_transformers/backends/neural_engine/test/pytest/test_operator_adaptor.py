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
from intel_extension_for_transformers.backends.neural_engine.compile.sub_graph.operator_adaptor import OperatorAdaptor
import numpy as np


class TestOperatorAdaptor(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        pass

    @classmethod
    def tearDownClass(self):
        pass

    def test_gather_sweep(self):
        graph = Graph()
        graph.framework_modeling_config['framework'] = 'onnxruntime'
        input_data_node = OPERATORS['Input']()
        input_tensors = []
        output_tensors = [Tensor(), Tensor(), Tensor()]
        input_data_node.construct('input_data', 'Input', input_tensors=input_tensors, 
                                output_tensors=output_tensors)

        gather1_node = OPERATORS['Gather']()
        input_tensors = [Tensor(name='g1_src0'),
                         Tensor(name='g1_idx:0', data=np.array(0).astype("int32"))]
        output_tensors = [Tensor(name='gather1:0', source_op=['gather1'], dest_op=[])]
        gather1_node.construct('gather1', 'Gather', input_tensors=input_tensors, 
                                output_tensors=output_tensors)
        gather1_node._filling_method = 'extract_from_onnxruntime'

        gather2_node = OPERATORS['Gather']()
        input_tensors = [Tensor(name='g2_idx:0', data=np.array(0).astype("int32")),
                         Tensor(name='g2_src1')]
        output_tensors = [Tensor(name='gather2:0', source_op=['gather2'], dest_op=[])]
        gather2_node.construct('gather2', 'Gather', input_tensors=input_tensors, 
                                output_tensors=output_tensors)
        gather2_node._filling_method = 'extract_from_onnxruntime'

        graph.insert_nodes(len(graph.nodes), [input_data_node, gather1_node, gather2_node])
        graph = OperatorAdaptor()(graph)
        self.assertEqual(3, len(graph.nodes))
        self.assertEqual(0, graph.nodes[1].input_tensors[1].data.item())
        self.assertEqual(0, graph.nodes[2].input_tensors[1].data.item())

    def test_reshape_non_2d_src_before_inner_product(self):
        graph = Graph()
        graph.framework_modeling_config['framework'] = 'onnxruntime'
        input_data_node = OPERATORS['Input']()
        input_tensors = []
        output_tensors = [Tensor(), Tensor(), Tensor()]
        input_data_node.construct('input_data', 'Input', input_tensors=input_tensors, 
                                output_tensors=output_tensors)

        transpose_node = OPERATORS['Transpose']()
        input_tensors = [Tensor(name='t_src0')]
        output_tensors = [Tensor(name='transpose:0', source_op=['transpose'], dest_op=['matmul'])]
        transpose_node.construct('transpose', 'Transpose', input_tensors=input_tensors, 
                                 output_tensors=output_tensors, attr=OrderedDict({
                                 'dst_perm': '1,0,2'}))

        matmul_node = OPERATORS['MatMul']()
        input_tensors = [Tensor(name='transpose:0', source_op=['transpose'], dest_op=['matmul']),
                         Tensor(name='m_src1', data=np.random.randn(768, 768).astype("float32"))]
        output_tensors = [Tensor(name='matmul:0', source_op=['matmul'], dest_op=[])]
        matmul_node.construct('matmul', 'MatMul', input_tensors=input_tensors, 
                                output_tensors=output_tensors)

        graph.insert_nodes(len(graph.nodes), [input_data_node, transpose_node, matmul_node])
        graph = OperatorAdaptor()(graph)
        self.assertEqual(4, len(graph.nodes))
        self.assertEqual('Reshape', graph.nodes[2].op_type)
        self.assertEqual('-1,768', graph.nodes[2].attr['dst_shape'])

    def test_ip_squeeze(self):
        graph = Graph()
        graph.framework_modeling_config['framework'] = 'torch'
        input_data_node = OPERATORS['Input']()
        input_tensors = []
        output_tensors = [Tensor(), Tensor(), Tensor()]
        input_data_node.construct('input_data', 'Input', input_tensors=input_tensors,
                                output_tensors=output_tensors)

        ip_node = OPERATORS['InnerProduct']()
        input_tensors = [Tensor(name='ip_src0'),
                         Tensor(name='ip_idx:0', data=np.array(1).astype("float32"))]
        output_tensors = [Tensor(name='ip:0', source_op=['ip'], dest_op=['squeeze'])]
        ip_node.construct('ip', 'InnerProduct', input_tensors=input_tensors,
                                output_tensors=output_tensors)

        squeeze_node = OPERATORS['Squeeze']()
        input_tensors = [Tensor(name='ip:0', source_op=['ip'], dest_op=['squeeze'])]
        output_tensors = [Tensor(name='squeeze:0', source_op=['squeeze'], dest_op=[])]
        squeeze_node.construct('squeeze', 'Squeeze', input_tensors=input_tensors,
                                output_tensors=output_tensors, attr=OrderedDict({'axes': 1}))

        graph.insert_nodes(len(graph.nodes), [input_data_node, ip_node, squeeze_node])
        graph = OperatorAdaptor()(graph)
        self.assertEqual(2, len(graph.nodes))
        self.assertEqual(1, graph.nodes[1].attr['squeeze_dims'])

if __name__ == "__main__":
    unittest.main()
