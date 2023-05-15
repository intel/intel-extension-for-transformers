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
from intel_extension_for_transformers.backends.neural_engine.compile import compile
import intel_extension_for_transformers.backends.neural_engine.compile.graph_utils as util
import os
import torch
import onnx
import shutil

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.linear = torch.nn.Linear(30, 50)

    def forward(self, x):
        x = self.linear(x)
        return x

class TestFWK(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        pass

    @classmethod
    def tearDownClass(self):
        pass

    def test_fwk(self):
        n = Net()
        example_in = torch.rand(3, 30)
        torch_model = torch.jit.trace(n, example_in)
        torch.onnx.export(n, example_in, 'test_fwk.onnx')
        torch.jit.save(torch_model, 'test_fwk.pt')
        onnx_model = onnx.load('test_fwk.onnx')
        graph = compile('test_fwk.pt')
        graph.save('test_fwk.ir')
        self.assertEqual(util.get_model_fwk_name(torch_model), 'torch')
        self.assertEqual(util.get_model_fwk_name('test_fwk.pt'), 'torch')
        self.assertEqual(util.get_model_fwk_name(onnx_model), 'onnxruntime')
        self.assertEqual(util.get_model_fwk_name('test_fwk.onnx'), 'onnxruntime')
        self.assertEqual(util.get_model_fwk_name('test_fwk.ir'), 'neural engine')
        os.remove('test_fwk.pt')
        os.remove('test_fwk.onnx')
        shutil.rmtree('test_fwk.ir')

if __name__ == "__main__":
    unittest.main()
