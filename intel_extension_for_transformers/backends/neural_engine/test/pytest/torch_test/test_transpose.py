#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Intel Corporation
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
import sys
import torch
import torch.nn as nn
import numpy as np
import os
import shutil
from intel_extension_for_transformers.backends.neural_engine.compile import compile
from intel_extension_for_transformers.backends.neural_engine.compile.graph import Graph

file_name = os.path.splitext(os.path.basename(__file__))[0]

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

    def forward(self, x):
        x = torch.transpose(x, 0, 1)
        return x

class TestTorchOP(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        pass

    @classmethod
    def tearDownClass(self):
        pass

    def test_1(self):
        n = Net()
        example_in = torch.rand(3, 4, 5)
        traced_model = torch.jit.trace(n, example_in)
        
        torch.jit.save(traced_model, '{}.pt'.format(file_name))
        # torch.onnx.export(n, example_in, '{}.onnx'.format(file_name))
        ref_out = traced_model(example_in).detach().numpy()
        print(ref_out.shape)
        
        graph = compile('{}.pt'.format(file_name))
        graph.save(file_name)
        newgraph = Graph()
        newgraph.graph_init(file_name + '/conf.yaml', file_name + '/model.bin')
        out = newgraph.inference([example_in.numpy()])

        np.testing.assert_array_equal(ref_out, [*out.values()][0])
        os.remove('{}.pt'.format(file_name))
        shutil.rmtree(file_name)

    def test_2(self):
        n = Net()
        example_in = torch.rand(3, 4)
        traced_model = torch.jit.trace(n, example_in)
        
        torch.jit.save(traced_model, '{}.pt'.format(file_name))
        # torch.onnx.export(n, example_in, '{}.onnx'.format(file_name))
        ref_out = traced_model(example_in).detach().numpy()
        print(ref_out.shape)
        
        graph = compile('{}.pt'.format(file_name))
        graph.save(file_name)
        newgraph = Graph()
        newgraph.graph_init(file_name + '/conf.yaml', file_name + '/model.bin')
        out = newgraph.inference([example_in.numpy()])

        np.testing.assert_array_equal(ref_out, [*out.values()][0])
        os.remove('{}.pt'.format(file_name))
        shutil.rmtree(file_name)

if __name__ == "__main__":
    unittest.main()
