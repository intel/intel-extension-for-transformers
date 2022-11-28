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

import os
import unittest
from onnx import helper
from onnx import TensorProto
from intel_extension_for_transformers.backends.neural_engine.compile.onnx_utils \
                                import is_supported_onnx_graph, is_supported_onnx_node

os.environ['GLOG_minloglevel'] = '2'
class TestIsSupportedOnnxGraph(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        pass

    @classmethod
    def tearDownClass(self):
        pass

    def test_is_supported_onnx_graph(self):
        '''test is_supported_onnx_graph'''
        # create onnx matmul + bias graph
        # input and output 
        a = helper.make_tensor_value_info('a', TensorProto.FLOAT, [10, 10])
        x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [10, 10])
        b = helper.make_tensor_value_info('b', TensorProto.FLOAT, [10, 10])
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [10, 10])
        # Mul
        mul = helper.make_node('Mul', ['a', 'x'], ['c'])
        # Add
        add = helper.make_node('Add', ['c', 'b'], ['output'])
        # graph and model
        graph = helper.make_graph([mul, add], 'linear_func', [a, x, b], [output])
        # test API
        is_supported = is_supported_onnx_graph(graph)
        self.assertEqual(is_supported, True)
        
        '''test is_supported_onnx_node'''
        ops_type = ["Add", "Softmax", "Slice", "ReduceMean", "Reshape",
                    "Concat", "Gather", "QuantizeLinear", "Transpose", "MatMul",
                    "Sqrt", "Unsqueeze", "Shape", "Erf", "Pow", "DequantizeLinear",
                    "Cast", "Tanh", "Div", "Mul", "Sub", "Constant", "Relu", "Conv",
                    "Identity", "Split", "TopK"]
        for op_type in ops_type:
            is_supported_node = is_supported_onnx_node(op_type)
            self.assertEqual(is_supported_node, True)

if __name__ == "__main__":
    unittest.main()
