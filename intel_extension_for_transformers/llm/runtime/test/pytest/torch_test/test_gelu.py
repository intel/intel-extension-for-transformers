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
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.gelu(x)
        return x

class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()

    def forward(self, input):
        return 0.5 * input * (1.0 + torch.tanh(input * 0.7978845608 * (1.0 + 0.044715 * input * input)))

class TestTorchOP(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        pass

    @classmethod
    def tearDownClass(self):
        pass

    def test_1(self):
        n = Net()
        example_in = torch.rand(3, 256)
        traced_model = torch.jit.trace(n, example_in)
        
        torch.jit.save(traced_model, '{}.pt'.format(file_name))
        ref_out = traced_model(example_in).detach().numpy()
        
        graph = compile('{}.pt'.format(file_name))
        graph.save(file_name)
        newgraph = Graph()
        newgraph.graph_init(file_name + '/conf.yaml', file_name + '/model.bin')
        out = newgraph.inference([example_in.numpy()])

        np.testing.assert_almost_equal(ref_out, [*out.values()][0], decimal=5)
        os.remove('{}.pt'.format(file_name))
        # shutil.rmtree(file_name)

    def test_2(self):
        n = Net2()
        example_in = torch.rand(3, 256)
        traced_model = torch.jit.trace(n, example_in)
        
        torch.jit.save(traced_model, '{}.pt'.format(file_name))
        ref_out = traced_model(example_in).detach().numpy()
        
        graph = compile('{}.pt'.format(file_name))
        graph.save(file_name)
        newgraph = Graph()
        newgraph.graph_init(file_name + '/conf.yaml', file_name + '/model.bin')
        out = newgraph.inference([example_in.numpy()])

        np.testing.assert_almost_equal(ref_out, [*out.values()][0], decimal=5) 
        os.remove('{}.pt'.format(file_name))
        shutil.rmtree(file_name)

if __name__ == "__main__":
    unittest.main()
