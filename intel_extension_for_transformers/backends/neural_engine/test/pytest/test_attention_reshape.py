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
from intel_extension_for_transformers.backends.neural_engine.compile.sub_graph.attention_reshape import AttentionReshape
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
        output_tensors = [Tensor(), Tensor(), Tensor()]
        input_data_node.construct('input_data', 'Input', input_tensors=input_tensors,
                                output_tensors=output_tensors)

        batchmatmul_node = OPERATORS['BatchMatMul']()
        input_tensors = [Tensor(name='b_src0', dest_op=['batchmatmul']),
                         Tensor(name='b_src1', dest_op=['batchmatmul'])]
        output_tensors = [Tensor(name='batchmatmul:0', source_op=['batchmatmul'],
                                 dest_op=['transpose'])]
        batchmatmul_node.construct('batchmatmul', 'BatchMatMul', input_tensors=input_tensors,
                                output_tensors=output_tensors, attr=OrderedDict({
                                    'src_perm': '0,1,2,3', 'dst_perm': '0,2,1,3'}))

        transpose_node = OPERATORS['Transpose']()
        input_tensors = [Tensor(name='batchmatmul:0', source_op=['batchmatmul'],
                                dest_op=['transpose'])]
        output_tensors = [Tensor(name='transpose:0', source_op=['transpose'],
                                 dest_op=['shape', 'reshape'])]
        transpose_node.construct('transpose', 'Transpose', input_tensors=input_tensors,
                                output_tensors=output_tensors, attr=OrderedDict({
                                    'src_perm': '0,1,2,3', 'dst_perm': '0,2,1,3'}))

        shape_node = OPERATORS['Shape']()
        input_tensors = [Tensor(name='transpose:0', source_op=['transpose'], dest_op=['shape'])]
        output_tensors = [Tensor(name='shape:0', source_op=['shape'], dest_op=['gather'])]
        shape_node.construct('shape', 'Shape', input_tensors=input_tensors,
                                output_tensors=output_tensors
                                    )

        gather_node = OPERATORS['Gather']()
        input_tensors = [Tensor(name='shape:0', source_op=['shape'], dest_op=['gather'])]
        output_tensors = [Tensor(name='gather:0', source_op=['gather'], dest_op=['unsqueeze'])]
        gather_node.construct('gather', 'Gather', input_tensors=input_tensors,
                                output_tensors=output_tensors, attr=OrderedDict({
                                    'batch_dims': '0', 'axis': '0'}))

        unsqueeze_node = OPERATORS['Unsqueeze']()
        input_tensors = [Tensor(name='gather:0', source_op=['gather'], dest_op=['unsqueeze'])]
        output_tensors = [Tensor(name='unsqueeze:0', data=np.array(1), source_op=['unsqueeze'], dest_op=['concat'])]
        unsqueeze_node.construct('unsqueeze', 'Unsqueeze', input_tensors=input_tensors,
                                output_tensors=output_tensors, attr=OrderedDict({
                                    'axis': '0'}))

        concat_node = OPERATORS['Concat']()
        input_tensors = [Tensor(name='concat0', data=np.array(1), shape=[1, 384]) ,Tensor(name='unsqueeze:0', source_op=['unsqueeze'], dest_op=['concat']), Tensor(name='concat2', shape=[1, 384] ,data=np.array(1))]
        output_tensors = [Tensor(name='concat:0', source_op=['concat'], dest_op=['reshape'])]
        concat_node.construct('concat', 'Concat', input_tensors=input_tensors,
                                output_tensors=output_tensors
                                    )

        reshape_node = OPERATORS['Reshape']()
        input_tensors = [Tensor(name='concat:0', source_op=['concat'], dest_op=['reshape']),
                Tensor(name='transpose:0', source_op=['transpose'], dest_op=['reshape'])]
        output_tensors = [Tensor(name='reshape:0', source_op=['reshape'], dest_op=['matmul'])]
        reshape_node.construct('reshape', 'Reshape', input_tensors=input_tensors,
                                output_tensors=output_tensors,
                                    )

        matmul_node = OPERATORS['MatMulWithBias']()
        input_tensors = [Tensor(name='reshape:0', source_op=['reshape'], dest_op=['matmul']),
                            Tensor(name='matmul1',data=np.array(1)), Tensor(name='matmul2',data=np.array(1))]
        output_tensors = [Tensor(name='matmul:0', source_op=['matmul'],
                                dest_op=[])]
        matmul_node.construct('matmul', 'MatMulWithBias', input_tensors=input_tensors,
                                output_tensors=output_tensors, attr=OrderedDict(
                                    {'src1_perm': '0,1'}))

        graph.insert_nodes(len(graph.nodes), [input_data_node, batchmatmul_node, transpose_node,
                                              shape_node, gather_node, unsqueeze_node, concat_node,
                                              reshape_node, matmul_node])
        graph = AttentionReshape()(graph)
        self.assertEqual(5, len(graph.nodes))
        self.assertEqual('-1,1', graph.nodes[3].attr['dst_shape'])
        self.assertEqual('0,1', graph.nodes[4].attr['src1_perm'])


    def test_attention_reshape_1(self):
        graph = Graph()
        graph.framework_modeling_config['framework'] = 'onnxruntime'
        input_data_node = OPERATORS['Input']()
        input_tensors = []
        output_tensors = [Tensor(), Tensor(), Tensor()]
        input_data_node.construct('input_data', 'Input', input_tensors=input_tensors,
                                output_tensors=output_tensors)

        reshape_node = OPERATORS['Reshape']()
        input_tensors = [Tensor(data=np.array(1))]
        output_tensors = [Tensor(name='reshape:0', source_op=['reshape'], dest_op=['matmul'])]
        reshape_node.construct('reshape', 'Reshape', input_tensors=input_tensors,
                                output_tensors=output_tensors, attr=OrderedDict({
                                    'dst_shape': '0,0,768'}))

        matmul_node = OPERATORS['MatMulWithBias']()
        input_tensors = [Tensor(name='reshape:0', source_op=['reshape'], dest_op=['matmul']),
                            Tensor(data=np.array(1)), Tensor(data=np.array(1))]
        output_tensors = [Tensor(name='matmul:0', source_op=['matmul'],
                                dest_op=[])]
        matmul_node.construct('matmul', 'MatMulWithBias', input_tensors=input_tensors,
                                output_tensors=output_tensors, attr=OrderedDict(
                                    {'src1_perm': '0,1'}))

        graph.insert_nodes(len(graph.nodes), [input_data_node, reshape_node, matmul_node])
        graph = AttentionReshape()(graph)
        self.assertEqual(3, len(graph.nodes))
        self.assertEqual('-1,768', graph.nodes[1].attr['dst_shape'])
        self.assertEqual('0,1', graph.nodes[2].attr['src1_perm'])


if __name__ == "__main__":
    unittest.main()
