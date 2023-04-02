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
from intel_extension_for_transformers.backends.neural_engine.compile.sub_graph.cast_to import CastTo
import numpy as np


class TestCastTo(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        pass

    @classmethod
    def tearDownClass(self):
        pass

    def test_softmax_cast(self):
        graph = Graph()
        graph.framework_modeling_config['framework'] = 'onnxruntime'
        input_data_node = OPERATORS['Input']()
        input_tensors = []
        output_tensors = [Tensor(), Tensor(), Tensor()]
        input_data_node.construct('input_data', 'Input', input_tensors=input_tensors, 
                                output_tensors=output_tensors)

        softmax_node = OPERATORS['Softmax']()
        input_tensors = [Tensor(name='s_src0')]
        output_tensors = [Tensor(name='softmax:0', source_op=['softmax'], dest_op=['cast'])]
        softmax_node.construct('softmax', 'Softmax', input_tensors=input_tensors, 
                                output_tensors=output_tensors)
        
        cast_node = OPERATORS['Cast']()
        input_tensors = [Tensor(name='softmax:0', source_op=['softmax'], dest_op=['cast'])]
        output_tensors = [Tensor(name='cast:0', source_op=['cast'],
                                dest_op=[])]
        cast_node.construct('cast', 'Cast', input_tensors=input_tensors, 
                                output_tensors=output_tensors, attr=OrderedDict({
                                'DstT': 'fp32'}))

        graph.insert_nodes(len(graph.nodes), [input_data_node, softmax_node, cast_node])
        graph = CastTo()(graph)
        self.assertEqual(2, len(graph.nodes))
        self.assertEqual('fp32', graph.nodes[1].attr['dtype'])

    def test_greater_cast_reducesum(self):
        graph = Graph()
        graph.framework_modeling_config['framework'] = 'onnxruntime'
        input_data_node = OPERATORS['Input']()
        input_tensors = []
        output_tensors = [Tensor(), Tensor(), Tensor()]
        input_data_node.construct('input_data', 'Input', input_tensors=input_tensors, 
                                output_tensors=output_tensors)

        greater_node = OPERATORS['Greater']()
        input_tensors = [Tensor(name='g_src0'), Tensor(name='g_src1',
                         data=np.array([1]).astype("int64"), shape=[1])]
        output_tensors = [Tensor(name='greater:0', source_op=['greater'], dest_op=['cast'])]
        greater_node.construct('greater', 'Greater', input_tensors=input_tensors, 
                                output_tensors=output_tensors)

        cast_node = OPERATORS['Cast']()
        input_tensors = [Tensor(name='greater:0', source_op=['greater'], dest_op=['cast'])]
        output_tensors = [Tensor(name='cast:0', source_op=['cast'], dest_op=['reducesum'])]
        cast_node.construct('cast', 'Cast', input_tensors=input_tensors, 
                                output_tensors=output_tensors, attr=OrderedDict({
                                'DstT': 'int64'}))

        rs_node = OPERATORS['ReduceSum']()
        input_tensors = [Tensor(name='cast:0', source_op=['cast'], dest_op=['reducesum'])]
        output_tensors = [Tensor(name='reducesum:0', source_op=['reducesum'],
                                dest_op=[])]
        rs_node.construct('reducesum', 'ReduceSum', input_tensors=input_tensors, 
                                output_tensors=output_tensors)

        graph.insert_nodes(len(graph.nodes), [input_data_node, greater_node, cast_node, rs_node])
        graph = CastTo()(graph)
        self.assertEqual(3, len(graph.nodes))
        self.assertEqual('s8', graph.nodes[1].attr['dtype'])

    def test_range_cast_less(self):
        graph = Graph()
        graph.framework_modeling_config['framework'] = 'onnxruntime'
        input_data_node = OPERATORS['Input']()
        input_tensors = []
        output_tensors = [Tensor(), Tensor(), Tensor()]
        input_data_node.construct('input_data', 'Input', input_tensors=input_tensors, 
                                output_tensors=output_tensors)

        cast_node = OPERATORS['Cast']()
        input_tensors = [Tensor(name='c_src0')]
        output_tensors = [Tensor(name='cast:0', source_op=['cast'], dest_op=['range'])]
        cast_node.construct('cast', 'Cast', input_tensors=input_tensors, 
                                output_tensors=output_tensors, attr=OrderedDict({
                                'DstT': 'int64'}))

        range_node = OPERATORS['Range']()
        input_tensors = [Tensor(name='cast:0', source_op=['cast'], dest_op=['range'])]
        output_tensors = [Tensor(name='range:0', source_op=['range'], dest_op=['less'])]
        range_node.construct('range', 'Range', input_tensors=input_tensors, 
                                output_tensors=output_tensors)

        less_node = OPERATORS['Less']()
        input_tensors = [Tensor(name='range:0', source_op=['range'], dest_op=['less']),
                         Tensor(name='less_val', data=np.array([3]).astype("int64"))]
        output_tensors = [Tensor(name='less:0', source_op=['less'],
                                dest_op=[])]
        less_node.construct('less', 'Less', input_tensors=input_tensors, 
                                output_tensors=output_tensors)

        graph.insert_nodes(len(graph.nodes), [input_data_node, cast_node, range_node, less_node])
        graph = CastTo()(graph)
        self.assertEqual(3, len(graph.nodes))
        self.assertEqual('int32', graph.nodes[1].attr['src0_dtype'])


if __name__ == "__main__":
    unittest.main()
