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
import intel_extension_for_transformers.backends.neural_engine.compile as compile
import numpy as np
import onnx
from intel_extension_for_transformers.backends.neural_engine.compile.ops.op import OPERATORS, Operator
from intel_extension_for_transformers.backends.neural_engine.compile.ops.tensor import Tensor


class TestOnnxUtils(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        pass

    @classmethod
    def tearDownClass(self):
        pass

    def test_change_num_name(self):
        out = compile.onnx_utils.change_num_name(1)
        self.assertEqual(1, out)
        
    def test_change_num_namei_same(self):
        out = compile.onnx_utils.change_num_name('1')
        self.assertEqual('1_tensor', out)
    
    def test_bias_to_int32_if1(self):
        fake_input_tensors = [
          Tensor(data=np.array([[1,2],[3,4]], dtype=np.int8)),
          Tensor(data=np.array(0.1, dtype=np.float32)),
          Tensor(data=np.array(0.2, dtype=np.float32)),
          Tensor(data=None),
          Tensor(data=None),
          Tensor(data=None),
        ]
        fake_bias_node = OPERATORS['Add']()
        fake_bias_node.construct('bias_add', 'Add', 
                                    input_tensors=fake_input_tensors)
        out = compile.onnx_utils.bias_to_int32(fake_bias_node, 0.3, 0.4)
        golden_out = np.array([[1,2],[2,3]])
        self.assertSequenceEqual(golden_out.tolist(), out.tolist())
    
    def test_bias_to_int32_else(self):
        fake_input_tensors = [
          Tensor(data=None, source_op=[None]),
          Tensor(data=None, source_op=[None]),
          Tensor(data=None, source_op=[None]),
          Tensor(data=np.array([[1,2],[3,4]], dtype=np.int8)),
          Tensor(data=np.array(0.1, dtype=np.float32)),
          Tensor(data=np.array(0.2, dtype=np.float32)),
        ]
        fake_bias_node = OPERATORS['Add']()
        fake_bias_node.construct('bias_add', 'Add', 
                                    input_tensors=fake_input_tensors)
        out = compile.onnx_utils.bias_to_int32(fake_bias_node, 0.3, 0.4)
        golden_out = np.array([[1,2],[2,3]])
        self.assertSequenceEqual(golden_out.tolist(), out.tolist())
    
    def test_bias_to_int32_if2(self):
        fake_input_tensors = [
          Tensor(data=np.array([[1,2],[3,4]], dtype=np.int64)),
          Tensor(data=np.array(0.1, dtype=np.float32)),
          Tensor(data=np.array(0.2, dtype=np.float32)),
          Tensor(data=None),
          Tensor(data=None),
          Tensor(data=None),
        ]
        fake_bias_node = OPERATORS['Add']()
        fake_bias_node.construct('bias_add', 'Add', 
                                    input_tensors=fake_input_tensors)
        out = compile.onnx_utils.bias_to_int32(fake_bias_node, 0.3, 0.4)
        self.assertEqual(None, out)

    def test_bf16_tensor_to_array(self):
        numpy_array = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float16)
        print(f"Original Numpy array:\n{numpy_array}\n")
        # Convert the Numpy array to a TensorProto    
        tensor = onnx.helper.make_tensor('bf16_tensor', onnx.TensorProto.BFLOAT16, [2, 3], [1.0,2.0,3.0,4.0,5.0,6.0], False)
        print(f"TensorProto:\n{tensor}")
        out = compile.onnx_utils._bf16_tensor_to_array(tensor)
        print(f"Output Numpy array:\n{out}\n")
        self.assertEqual(numpy_array.all(), out.all())

        tensor_2 = onnx.helper.make_tensor('bf16_tensor', onnx.TensorProto.BFLOAT16, [2, 3], numpy_array.tobytes(), True)
        out_2 = compile.onnx_utils._bf16_tensor_to_array(tensor_2)
        print(f"Output_2 Numpy array:\n{out_2}\n")

        self.assertEqual(numpy_array.all(), out_2.all())


if __name__ == "__main__":
    unittest.main()
