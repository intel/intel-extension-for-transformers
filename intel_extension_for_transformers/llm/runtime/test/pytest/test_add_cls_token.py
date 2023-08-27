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
from intel_extension_for_transformers.backends.neural_engine.compile.sub_graph.add_cls_token import AddClsToken
import numpy as np


class TestAddClsToken(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        pass

    @classmethod
    def tearDownClass(self):
        pass

    def test_add_cls_token_0(self):
        graph = Graph()
        graph.framework_modeling_config['framework'] = 'onnxruntime'
        input_data_node = OPERATORS['Input']()
        input_tensors = []
        output_tensors = [Tensor(), Tensor(), Tensor()]
        input_data_node.construct('input_data', 'Input', input_tensors=input_tensors,
                                output_tensors=output_tensors)

        transpose_node = OPERATORS['Transpose']()
        input_tensors = [Tensor(data=np.array(1))]
        output_tensors = [Tensor(name='transpose:0', source_op=['transpose'], dest_op=['shape']),
                            Tensor(name='transpose:0', source_op=['transpose'], dest_op=['concat1'])]
        transpose_node.construct('transpose', 'Transpose', input_tensors=input_tensors,
                                    output_tensors=output_tensors, attr=OrderedDict({
                                        'src_perm': '0,1,2', 'dst_perm': '0,2,1'}))

        shape_node = OPERATORS['Shape']()
        input_tensors = [Tensor(name='transpose:0', source_op=['transpose'], dest_op=['shape'])]
        output_tensors = [Tensor(name='shape:0', source_op=['shape'], dest_op=['gather'])]
        shape_node.construct('shape', 'Shape', input_tensors=input_tensors,
                                output_tensors=output_tensors)

        gather_node = OPERATORS['Gather']()
        input_tensors = [Tensor(name='shape:0', source_op=['shape'], dest_op=['gather']),
                            Tensor(name='gather:0_1', data=np.array(0), shape=[1])]
        output_tensors = [Tensor(name='gather:0', source_op=['gather'], dest_op=['unsqueeze'])]
        gather_node.construct('gather', 'Gather', input_tensors=input_tensors,
                                output_tensors=output_tensors, attr=OrderedDict({
                                    'axis': 0}))

        unsqueeze_node = OPERATORS['Unsqueeze']()
        input_tensors = [Tensor(name='gather:0', source_op=['gather'], dest_op=['unsqueeze']),
                            Tensor(name='unsqueeze:0_1', data=np.array(0), shape=[1])]
        output_tensors = [Tensor(name='unsqueeze:0', source_op=['unsqueeze'], dest_op=['concat'])]
        unsqueeze_node.construct('unsqueeze', 'Unsqueeze', input_tensors=input_tensors,
                                output_tensors=output_tensors)

        concat_node = OPERATORS['Concat']()
        input_tensors = [Tensor(name='unsqueeze:0', source_op=['unsqueeze'], dest_op=['concat']),
                            Tensor(name='concat:0_1', data=np.array(-1), shape=[1]),
                            Tensor(name='concat:0_2', data=np.array(-1), shape=[1])]
        output_tensors = [Tensor(name='concat:0', source_op=['concat'], dest_op=['reshape'])]
        concat_node.construct('concat', 'Concat', input_tensors=input_tensors,
                                output_tensors=output_tensors, attr=OrderedDict({
                                    'axis': 0}))

        reshape_node = OPERATORS['Reshape']()
        input_tensors = [Tensor(name='concat:0', source_op=['concat'], dest_op=['reshape']),
                            Tensor(name='reshape:0_1', data=np.array(-1), shape=[1])]
        output_tensors = [Tensor(name='reshape:0', source_op=['reshape'], dest_op=['shape1']),
                            Tensor(name='reshape:0', source_op=['reshape'], dest_op=['equal']),
                            Tensor(name='reshape:0', source_op=['reshape'], dest_op=['where'])]
        reshape_node.construct('reshape', 'Reshape', input_tensors=input_tensors,
                                output_tensors=output_tensors)

        shape_node_1 = OPERATORS['Shape']()
        input_tensors = [Tensor(name='reshape:0', source_op=['reshape'], dest_op=['shape1'])]
        output_tensors = [Tensor(name='shape:1', source_op=['shape1'], dest_op=['constant_of_shape'])]
        shape_node_1.construct('shape1', 'Shape', input_tensors=input_tensors,
                                output_tensors=output_tensors)

        constant_of_shape_node = OPERATORS['ConstantOfShape']()
        input_tensors = [Tensor(name='shape:1', source_op=['shape1'], dest_op=['constant_of_shape'])]
        output_tensors = [Tensor(name='constant_of_shape:0', source_op=['constant_of_shape'], dest_op=['mul']),
                            Tensor(name='constant_of_shape:0', source_op=['constant_of_shape'], dest_op=['where'])]
        constant_of_shape_node.construct('constant_of_shape', 'ConstantOfShape', input_tensors=input_tensors,
                                output_tensors=output_tensors)

        mul_node = OPERATORS['Mul']()
        input_tensors = [Tensor(name='constant_of_shape:0', source_op=['constant_of_shape'], dest_op=['mul']),
                            Tensor(name='mul:0_1', data=np.array(-1), shape=[1])]
        output_tensors = [Tensor(name='mul:0', source_op=['mul'], dest_op=['equal'])]
        mul_node.construct('mul', 'Mul', input_tensors=input_tensors,
                                output_tensors=output_tensors)

        equal_node = OPERATORS['Equal']()
        input_tensors = [Tensor(name='mul:0', source_op=['mul'], dest_op=['equal']),
                            Tensor(name='reshape:0', source_op=['reshape'], dest_op=['equal'])]
        output_tensors = [Tensor(name='equal:0', source_op=['equal'], dest_op=['where'])]
        equal_node.construct('equal', 'Equal', input_tensors=input_tensors,
                                output_tensors=output_tensors)

        where_node = OPERATORS['Where']()
        input_tensors = [Tensor(name='equal:0', source_op=['equal'], dest_op=['where']),
                            Tensor(name='constant_of_shape:0', source_op=['constant_of_shape'], dest_op=['where']),
                            Tensor(name='reshape:0', source_op=['reshape'], dest_op=['where'])]
        output_tensors = [Tensor(name='where:0', source_op=['where'], dest_op=['expand'])]
        where_node.construct('where', 'Where', input_tensors=input_tensors,
                                output_tensors=output_tensors)

        expand_node = OPERATORS['Expand']()
        input_tensors = [Tensor(name='expand:0_0', data=np.array(1), shape=[1, 1, 768]),
                            Tensor(name='where:0', source_op=['where'], dest_op=['expand'])]
        output_tensors = [Tensor(name='expand:0', source_op=['expand'], dest_op=['concat1'])]
        expand_node.construct('expand', 'Expand', input_tensors=input_tensors,
                                output_tensors=output_tensors)

        concat_node_1 = OPERATORS['Concat']()
        input_tensors = [Tensor(name='expand:0', source_op=['expand'], dest_op=['concat1']),
                            Tensor(name='transpose:0', source_op=['transpose'], dest_op=['concat1'])]
        output_tensors = [Tensor(name='concat:1', source_op=['concat1'], dest_op=[])]
        concat_node_1.construct('concat1', 'Concat', input_tensors=input_tensors,
                                output_tensors=output_tensors, attr=OrderedDict({
                                    'axis': 1}))

        graph.insert_nodes(len(graph.nodes), [transpose_node, shape_node, gather_node, unsqueeze_node, concat_node, reshape_node,
                                            shape_node_1, constant_of_shape_node, mul_node, equal_node, where_node, expand_node, concat_node_1])
        graph = AddClsToken()(graph)
        self.assertEqual(2, len(graph.nodes))
        self.assertEqual('0,2,1', graph.nodes[0].attr['dst_perm'])
        self.assertEqual(1, graph.nodes[1].attr['axis'])


    def test_add_cls_token_1(self):
        graph = Graph()
        graph.framework_modeling_config['framework'] = 'onnxruntime'
        input_data_node = OPERATORS['Input']()
        input_tensors = []
        output_tensors = [Tensor(), Tensor(), Tensor()]
        input_data_node.construct('input_data', 'Input', input_tensors=input_tensors,
                                output_tensors=output_tensors)

        transpose_node = OPERATORS['Transpose']()
        input_tensors = [Tensor(data=np.array(1))]
        output_tensors = [Tensor(name='transpose:0', source_op=['transpose'], dest_op=['concat1'])]
        transpose_node.construct('transpose', 'Transpose', input_tensors=input_tensors,
                                    output_tensors=output_tensors, attr=OrderedDict({
                                        'src_perm': '0,1,2', 'dst_perm': '0,2,1'}))

        shape_node = OPERATORS['Shape']()
        input_tensors = [Tensor(data=np.array(1))]
        output_tensors = [Tensor(name='shape:0', source_op=['shape'], dest_op=['gather'])]
        shape_node.construct('shape', 'Shape', input_tensors=input_tensors,
                                output_tensors=output_tensors)

        gather_node = OPERATORS['Gather']()
        input_tensors = [Tensor(name='shape:0', source_op=['shape'], dest_op=['gather']),
                            Tensor(name='gather:0_1', data=np.array(0), shape=[1])]
        output_tensors = [Tensor(name='gather:0', source_op=['gather'], dest_op=['unsqueeze'])]
        gather_node.construct('gather', 'Gather', input_tensors=input_tensors,
                                output_tensors=output_tensors, attr=OrderedDict({
                                    'axis': 0}))

        unsqueeze_node = OPERATORS['Unsqueeze']()
        input_tensors = [Tensor(name='gather:0', source_op=['gather'], dest_op=['unsqueeze']),
                            Tensor(name='unsqueeze:0_1', data=np.array(0), shape=[1])]
        output_tensors = [Tensor(name='unsqueeze:0', source_op=['unsqueeze'], dest_op=['concat'])]
        unsqueeze_node.construct('unsqueeze', 'Unsqueeze', input_tensors=input_tensors,
                                output_tensors=output_tensors)

        concat_node = OPERATORS['Concat']()
        input_tensors = [Tensor(name='unsqueeze:0', source_op=['unsqueeze'], dest_op=['concat']),
                            Tensor(name='concat:0_1', data=np.array(-1), shape=[1]),
                            Tensor(name='concat:0_2', data=np.array(-1), shape=[1])]
        output_tensors = [Tensor(name='concat:0', source_op=['concat'], dest_op=['reshape'])]
        concat_node.construct('concat', 'Concat', input_tensors=input_tensors,
                                output_tensors=output_tensors, attr=OrderedDict({
                                    'axis': 0}))

        reshape_node = OPERATORS['Reshape']()
        input_tensors = [Tensor(name='concat:0', source_op=['concat'], dest_op=['reshape']),
                            Tensor(name='reshape:0_1', data=np.array(-1), shape=[1])]
        output_tensors = [Tensor(name='reshape:0', source_op=['reshape'], dest_op=['shape1']),
                            Tensor(name='reshape:0', source_op=['reshape'], dest_op=['equal']),
                            Tensor(name='reshape:0', source_op=['reshape'], dest_op=['where'])]
        reshape_node.construct('reshape', 'Reshape', input_tensors=input_tensors,
                                output_tensors=output_tensors)

        shape_node_1 = OPERATORS['Shape']()
        input_tensors = [Tensor(name='reshape:0', source_op=['reshape'], dest_op=['shape1'])]
        output_tensors = [Tensor(name='shape:1', source_op=['shape1'], dest_op=['constant_of_shape'])]
        shape_node_1.construct('shape1', 'Shape', input_tensors=input_tensors,
                                output_tensors=output_tensors)

        constant_of_shape_node = OPERATORS['ConstantOfShape']()
        input_tensors = [Tensor(name='shape:1', source_op=['shape1'], dest_op=['constant_of_shape'])]
        output_tensors = [Tensor(name='constant_of_shape:0', source_op=['constant_of_shape'], dest_op=['mul']),
                            Tensor(name='constant_of_shape:0', source_op=['constant_of_shape'], dest_op=['where'])]
        constant_of_shape_node.construct('constant_of_shape', 'ConstantOfShape', input_tensors=input_tensors,
                                output_tensors=output_tensors)

        mul_node = OPERATORS['Mul']()
        input_tensors = [Tensor(name='constant_of_shape:0', source_op=['constant_of_shape'], dest_op=['mul']),
                            Tensor(name='mul:0_1', data=np.array(-1), shape=[1])]
        output_tensors = [Tensor(name='mul:0', source_op=['mul'], dest_op=['equal'])]
        mul_node.construct('mul', 'Mul', input_tensors=input_tensors,
                                output_tensors=output_tensors)

        equal_node = OPERATORS['Equal']()
        input_tensors = [Tensor(name='mul:0', source_op=['mul'], dest_op=['equal']),
                            Tensor(name='reshape:0', source_op=['reshape'], dest_op=['equal'])]
        output_tensors = [Tensor(name='equal:0', source_op=['equal'], dest_op=['where'])]
        equal_node.construct('equal', 'Equal', input_tensors=input_tensors,
                                output_tensors=output_tensors)

        where_node = OPERATORS['Where']()
        input_tensors = [Tensor(name='equal:0', source_op=['equal'], dest_op=['where']),
                            Tensor(name='constant_of_shape:0', source_op=['constant_of_shape'], dest_op=['where']),
                            Tensor(name='reshape:0', source_op=['reshape'], dest_op=['where'])]
        output_tensors = [Tensor(name='where:0', source_op=['where'], dest_op=['expand'])]
        where_node.construct('where', 'Where', input_tensors=input_tensors,
                                output_tensors=output_tensors)

        expand_node = OPERATORS['Expand']()
        input_tensors = [Tensor(name='expand:0_0', data=np.array(1), shape=[1, 1, 768]),
                            Tensor(name='where:0', source_op=['where'], dest_op=['expand'])]
        output_tensors = [Tensor(name='expand:0', source_op=['expand'], dest_op=['concat1'])]
        expand_node.construct('expand', 'Expand', input_tensors=input_tensors,
                                output_tensors=output_tensors)

        concat_node_1 = OPERATORS['Concat']()
        input_tensors = [Tensor(name='expand:0', source_op=['expand'], dest_op=['concat1']),
                            Tensor(name='transpose:0', source_op=['transpose'], dest_op=['concat1'])]
        output_tensors = [Tensor(name='concat:1', source_op=['concat1'], dest_op=[])]
        concat_node_1.construct('concat1', 'Concat', input_tensors=input_tensors,
                                output_tensors=output_tensors, attr=OrderedDict({
                                    'axis': 1}))

        graph.insert_nodes(len(graph.nodes), [transpose_node, shape_node, gather_node, unsqueeze_node, concat_node, reshape_node,
                                            shape_node_1, constant_of_shape_node, mul_node, equal_node, where_node, expand_node, concat_node_1])
        graph = AddClsToken()(graph)
        self.assertEqual(2, len(graph.nodes))
        self.assertEqual('0,2,1', graph.nodes[0].attr['dst_perm'])
        self.assertEqual(1, graph.nodes[1].attr['axis'])


    def test_add_cls_token_2(self):
        graph = Graph()
        graph.framework_modeling_config['framework'] = 'onnxruntime'
        input_data_node = OPERATORS['Input']()
        input_tensors = []
        output_tensors = [Tensor(), Tensor(), Tensor()]
        input_data_node.construct('input_data', 'Input', input_tensors=input_tensors,
                                output_tensors=output_tensors)

        transpose_node = OPERATORS['Transpose']()
        input_tensors = [Tensor(data=np.array(1))]
        output_tensors = [Tensor(name='transpose:0', source_op=['transpose'], dest_op=['concat1'])]
        transpose_node.construct('transpose', 'Transpose', input_tensors=input_tensors,
                                    output_tensors=output_tensors, attr=OrderedDict({
                                        'src_perm': '0,1,2', 'dst_perm': '0,2,1'}))

        shape_node = OPERATORS['Shape']()
        input_tensors = [Tensor(data=np.array(1))]
        output_tensors = [Tensor(name='shape:0', source_op=['shape'], dest_op=['gather'])]
        shape_node.construct('shape', 'Shape', input_tensors=input_tensors,
                                output_tensors=output_tensors)

        gather_node = OPERATORS['Gather']()
        input_tensors = [Tensor(name='shape:0', source_op=['shape'], dest_op=['gather']),
                            Tensor(name='gather:0_1', data=np.array(0), shape=[1])]
        output_tensors = [Tensor(name='gather:0', source_op=['gather'], dest_op=['unsqueeze'])]
        gather_node.construct('gather', 'Gather', input_tensors=input_tensors,
                                output_tensors=output_tensors, attr=OrderedDict({
                                    'axis': 0}))

        unsqueeze_node = OPERATORS['Unsqueeze']()
        input_tensors = [Tensor(name='gather:0', source_op=['gather'], dest_op=['unsqueeze']),
                            Tensor(name='unsqueeze:0_1', data=np.array(0), shape=[1])]
        output_tensors = [Tensor(name='unsqueeze:0', source_op=['unsqueeze'], dest_op=['concat'])]
        unsqueeze_node.construct('unsqueeze', 'Unsqueeze', input_tensors=input_tensors,
                                output_tensors=output_tensors)

        concat_node = OPERATORS['Concat']()
        input_tensors = [Tensor(name='unsqueeze:0', source_op=['unsqueeze'], dest_op=['concat']),
                            Tensor(name='concat:0_1', data=np.array(-1), shape=[1]),
                            Tensor(name='concat:0_2', data=np.array(-1), shape=[1])]
        output_tensors = [Tensor(name='concat:0', source_op=['concat'], dest_op=['reshape'])]
        concat_node.construct('concat', 'Concat', input_tensors=input_tensors,
                                output_tensors=output_tensors, attr=OrderedDict({
                                    'axis': 0}))

        reshape_node = OPERATORS['Reshape']()
        input_tensors = [Tensor(name='concat:0', source_op=['concat'], dest_op=['reshape']),
                            Tensor(name='reshape:0_1', data=np.array(-1), shape=[1])]
        output_tensors = [Tensor(name='reshape:0', source_op=['reshape'], dest_op=['equal']),
                            Tensor(name='reshape:0', source_op=['reshape'], dest_op=['where'])]
        reshape_node.construct('reshape', 'Reshape', input_tensors=input_tensors,
                                output_tensors=output_tensors)

        equal_node = OPERATORS['Equal']()
        input_tensors = [Tensor(name='reshape:0', source_op=['reshape'], dest_op=['equal']),
                            Tensor(name='equal:0_1', data=np.array(-1), shape=[3])]
        output_tensors = [Tensor(name='equal:0', source_op=['equal'], dest_op=['where'])]
        equal_node.construct('equal', 'Equal', input_tensors=input_tensors,
                                output_tensors=output_tensors)

        where_node = OPERATORS['Where']()
        input_tensors = [Tensor(name='equal:0', source_op=['equal'], dest_op=['where']),
                            Tensor(name='where:0_1', data=np.array(1), shape=[3]),
                            Tensor(name='reshape:0', source_op=['reshape'], dest_op=['where'])]
        output_tensors = [Tensor(name='where:0', source_op=['where'], dest_op=['expand'])]
        where_node.construct('where', 'Where', input_tensors=input_tensors,
                                output_tensors=output_tensors)

        expand_node = OPERATORS['Expand']()
        input_tensors = [Tensor(name='expand:0_0', data=np.array(1), shape=[1, 1, 768]),
                            Tensor(name='where:0', source_op=['where'], dest_op=['expand'])]
        output_tensors = [Tensor(name='expand:0', source_op=['expand'], dest_op=['concat1'])]
        expand_node.construct('expand', 'Expand', input_tensors=input_tensors,
                                output_tensors=output_tensors)

        concat_node_1 = OPERATORS['Concat']()
        input_tensors = [Tensor(name='expand:0', source_op=['expand'], dest_op=['concat1']),
                            Tensor(name='transpose:0', source_op=['transpose'], dest_op=['concat1'])]
        output_tensors = [Tensor(name='concat:1', source_op=['concat1'], dest_op=[])]
        concat_node_1.construct('concat1', 'Concat', input_tensors=input_tensors,
                                output_tensors=output_tensors, attr=OrderedDict({
                                    'axis': 1}))

        graph.insert_nodes(len(graph.nodes), [transpose_node, shape_node, gather_node, unsqueeze_node, concat_node, reshape_node,
                                            equal_node, where_node, expand_node, concat_node_1])
        graph = AddClsToken()(graph)
        self.assertEqual(2, len(graph.nodes))
        self.assertEqual('0,2,1', graph.nodes[0].attr['dst_perm'])
        self.assertEqual(1, graph.nodes[1].attr['axis'])


if __name__ == "__main__":
    unittest.main()
