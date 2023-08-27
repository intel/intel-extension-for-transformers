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
from intel_extension_for_transformers.backends.neural_engine.compile.sub_graph.conv_reshape import ConvReshape
import numpy as np


class TestConvReshape(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        pass

    @classmethod
    def tearDownClass(self):
        pass

    def test_conv_reshape_0(self):
        graph = Graph()
        graph.framework_modeling_config['framework'] = 'onnxruntime'
        input_data_node = OPERATORS['Input']()
        input_tensors = []
        output_tensors = [Tensor(), Tensor(), Tensor()]
        input_data_node.construct('input_data', 'Input', input_tensors=input_tensors,
                                output_tensors=output_tensors)

        conv_node = OPERATORS['Conv']()
        input_tensors = [Tensor(data=np.array(1), shape=[1,3,224,224]),
                            Tensor(name='conv:0_weight', data=np.array(1), shape=[768,3,16,16]),
                            Tensor(name='conv:0_bias', data=np.array(1),shape=[768])]
        output_tensors = [Tensor(name='conv:0', source_op=['conv'], dest_op=['shape']),
                Tensor(name='conv:0', source_op=['conv'], dest_op=['reshape'])]
        conv_node.construct('conv', 'Conv', input_tensors=input_tensors,
                                output_tensors=output_tensors, attr=OrderedDict({
                                    'dilations': '1,1', 'group': '1', 'kernel_shape': '16,16',
                                    'pads': '0,0,0,0', 'strides': '16,16'}))

        shape_node = OPERATORS['Shape']()
        input_tensors = [Tensor(name='conv:0', source_op=['conv'], dest_op=['shape'])]
        output_tensors = [Tensor(name='shape:0', source_op=['shape'], dest_op=['slice'])]
        shape_node.construct('shape', 'Shape', input_tensors=input_tensors,
                                output_tensors=output_tensors)

        slice_node = OPERATORS['Slice']()
        input_tensors = [Tensor(name='shape:0', source_op=['shape'], dest_op=['slice']),
                            Tensor(name='slice:0_1', data=np.array(0), shape=[1]),
                            Tensor(name='slice:0_2', data=np.array(2), shape=[1]),
                            Tensor(name='slice:0_3', data=np.array(0), shape=[1])]
        output_tensors = [Tensor(name='slice:0', source_op=['slice'], dest_op=['concat'])]
        slice_node.construct('slice', 'Slice', input_tensors=input_tensors,
                                output_tensors=output_tensors)

        concat_node = OPERATORS['Concat']()
        input_tensors = [Tensor(name='slice:0', source_op=['slice'], dest_op=['concat']),
                            Tensor(name='concat:0_1', data=np.array(-1), shape=[1])]
        output_tensors = [Tensor(name='concat:0', source_op=['concat'], dest_op=['reshape'])]
        concat_node.construct('concat', 'Concat', input_tensors=input_tensors,
                                output_tensors=output_tensors)

        reshape_node = OPERATORS['Reshape']()
        input_tensors = [Tensor(name='conv:0', source_op=['conv'], dest_op=['reshape']),
                            Tensor(name='concat:0', source_op=['concat'], dest_op=['reshape'])]
        output_tensors = [Tensor(name='reshape:0', source_op=['reshape'], dest_op=[])]
        reshape_node.construct('reshape', 'Reshape', input_tensors=input_tensors,
                                output_tensors=output_tensors)

        graph.insert_nodes(len(graph.nodes), [input_data_node, conv_node, shape_node, slice_node, concat_node, reshape_node])
        graph = ConvReshape()(graph)
        self.assertEqual(3, len(graph.nodes))
        self.assertEqual('-1,768,-1', graph.nodes[2].attr['dst_shape'])
        self.assertEqual(0, graph.nodes[2].attr['dims'])


if __name__ == "__main__":
    unittest.main()
