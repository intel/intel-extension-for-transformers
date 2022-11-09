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
from nlp_toolkit.backends.neural_engine.compile.ops.op import OPERATORS
from nlp_toolkit.backends.neural_engine.compile.ops.tensor import Tensor
from nlp_toolkit.backends.neural_engine.compile.onnx_utils import is_supported_onnx_node

os.environ['GLOG_minloglevel'] = '2'
class TestIsSupportedOnnxNode(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        pass

    @classmethod
    def tearDownClass(self):
        pass

    def test_is_supported_onnx_node(self):
        # create matmul node
        matmul_node = OPERATORS['MatMul']()
        input_tensors = []
        output_tensors = [Tensor()]
        matmul_node.construct('matmul', 'MatMul', input_tensors=input_tensors,
                                output_tensors=output_tensors)
        is_supported_matmul = is_supported_onnx_node(matmul_node.op_type)
        # assert
        self.assertEqual(is_supported_matmul, True)

if __name__ == "__main__":
    unittest.main()
