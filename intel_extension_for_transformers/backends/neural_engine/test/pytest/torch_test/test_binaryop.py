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
    def __init__(self, algo):
        super(Net, self).__init__()
        if algo == 'div':
            self.binaryop = torch.div
        elif algo == 'mul':
            self.binaryop = torch.mul
        elif algo == 'add':
            self.binaryop = torch.add
        elif algo == 'sub':
            self.binaryop = torch.sub
        elif algo == 'gt':
            self.binaryop = torch.gt
        elif algo == 'lt':
            self.binaryop = torch.lt
        elif algo == 'eq':
            self.binaryop = torch.eq
        elif algo == 'ne':
            self.binaryop = torch.ne
    def forward(self, x, y):
        return self.binaryop(x, y)

class TestTorchOP(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        pass

    @classmethod
    def tearDownClass(self):
        pass

    def test_1(self):
        n = Net('div')
        example_in = torch.rand(3, 256)
        example_in2 = torch.rand(256)
        traced_model = torch.jit.trace(n, (example_in, example_in2))
        torch.jit.save(traced_model, '{}.pt'.format(file_name))
        ref_out = traced_model(example_in, example_in2).detach().numpy()
        
        graph = compile('{}.pt'.format(file_name))
        graph.save(file_name)
        newgraph = Graph()
        newgraph.graph_init(file_name + '/conf.yaml', file_name + '/model.bin')
        out = newgraph.inference([example_in.numpy(), example_in2.numpy()])

        np.testing.assert_almost_equal(ref_out, [*out.values()][0], decimal=5)
        os.remove('{}.pt'.format(file_name))
        shutil.rmtree(file_name)

    def test_2(self):
        n = Net('mul')
        example_in = torch.rand(3, 256)
        example_in2 = torch.rand(256)
        traced_model = torch.jit.trace(n, (example_in, example_in2))
        torch.jit.save(traced_model, '{}.pt'.format(file_name))
        ref_out = traced_model(example_in, example_in2).detach().numpy()
        
        graph = compile('{}.pt'.format(file_name))
        graph.save(file_name)
        newgraph = Graph()
        newgraph.graph_init(file_name + '/conf.yaml', file_name + '/model.bin')
        out = newgraph.inference([example_in.numpy(), example_in2.numpy()])

        np.testing.assert_almost_equal(ref_out, [*out.values()][0], decimal=5)
        os.remove('{}.pt'.format(file_name))
        shutil.rmtree(file_name)

    # def test_3(self):
    #     n = Net('add')
    #     example_in = torch.rand(3, 256)
    #     example_in2 = torch.rand(256)
    #     traced_model = torch.jit.trace(n, (example_in, example_in2))
    #     
    #     torch.jit.save(traced_model, '{}.pt'.format(file_name))
    #     ref_out = traced_model(example_in, example_in2).detach().numpy()
        
    #     graph = compile('{}.pt'.format(file_name))
    #     graph.save(file_name)
    #     newgraph = Graph()
    #     newgraph.graph_init(file_name + '/conf.yaml', file_name + '/model.bin')
    #     out = newgraph.inference([example_in.numpy(), example_in2.numpy()])

    #     np.testing.assert_almost_equal(ref_out, [*out.values()][0], decimal=5)
    #     os.remove('{}.pt'.format(file_name))
    #     shutil.rmtree(file_name)

    # def test_4(self):
    #     n = Net('sub')
    #     example_in = torch.rand(3, 256)
    #     example_in2 = torch.rand(256)
    #     traced_model = torch.jit.trace(n, (example_in, example_in2))
    #     
    #     torch.jit.save(traced_model, '{}.pt'.format(file_name))
    #     ref_out = traced_model(example_in, example_in2).detach().numpy()
        
    #     graph = compile('{}.pt'.format(file_name))
    #     graph.save(file_name)
    #     newgraph = Graph()
    #     newgraph.graph_init(file_name + '/conf.yaml', file_name + '/model.bin')
    #     out = newgraph.inference([example_in.numpy(), example_in2.numpy()])

    #     np.testing.assert_almost_equal(ref_out, [*out.values()][0], decimal=5)
    #     os.remove('{}.pt'.format(file_name))
    #     shutil.rmtree(file_name)

    # def test_5(self):
    #     n = Net('gt')
    #     example_in = torch.rand(3, 256)
    #     example_in2 = torch.rand(256)
    #     traced_model = torch.jit.trace(n, (example_in, example_in2))
    #     
    #     torch.jit.save(traced_model, '{}.pt'.format(file_name))
    #     ref_out = traced_model(example_in, example_in2).detach().numpy()
        
    #     graph = compile('{}.pt'.format(file_name))
    #     graph.save(file_name)
    #     newgraph = Graph()
    #     newgraph.graph_init(file_name + '/conf.yaml', file_name + '/model.bin')
    #     out = newgraph.inference([example_in.numpy(), example_in2.numpy()])

    #     np.testing.assert_almost_equal(ref_out, [*out.values()][0], decimal=5)
    #     os.remove('{}.pt'.format(file_name))
    #     shutil.rmtree(file_name)

    # def test_6(self):
    #     n = Net('lt')
    #     example_in = torch.rand(3, 256)
    #     example_in2 = torch.rand(256)
    #     traced_model = torch.jit.trace(n, (example_in, example_in2))
    #     
    #     torch.jit.save(traced_model, '{}.pt'.format(file_name))
    #     ref_out = traced_model(example_in, example_in2).detach().numpy()
        
    #     graph = compile('{}.pt'.format(file_name))
    #     graph.save(file_name)
    #     newgraph = Graph()
    #     newgraph.graph_init(file_name + '/conf.yaml', file_name + '/model.bin')
    #     out = newgraph.inference([example_in.numpy(), example_in2.numpy()])

    #     np.testing.assert_almost_equal(ref_out, [*out.values()][0], decimal=5)
    #     os.remove('{}.pt'.format(file_name))
    #     shutil.rmtree(file_name)

    # def test_7(self):
    #     n = Net('eq')
    #     example_in = torch.rand(3, 256)
    #     example_in2 = torch.rand(256)
    #     traced_model = torch.jit.trace(n, (example_in, example_in2))
    #     
    #     torch.jit.save(traced_model, '{}.pt'.format(file_name))
    #     ref_out = traced_model(example_in, example_in2).detach().numpy()
        
    #     graph = compile('{}.pt'.format(file_name))
    #     graph.save(file_name)
    #     newgraph = Graph()
    #     newgraph.graph_init(file_name + '/conf.yaml', file_name + '/model.bin')
    #     out = newgraph.inference([example_in.numpy(), example_in2.numpy()])

    #     np.testing.assert_almost_equal(ref_out, [*out.values()][0], decimal=5)
    #     os.remove('{}.pt'.format(file_name))
    #     shutil.rmtree(file_name)

    # def test_8(self):
    #     n = Net('ne')
    #     example_in = torch.rand(3, 256)
    #     example_in2 = torch.rand(256)
    #     traced_model = torch.jit.trace(n, (example_in, example_in2))
    #     
    #     torch.jit.save(traced_model, '{}.pt'.format(file_name))
    #     ref_out = traced_model(example_in, example_in2).detach().numpy()
        
    #     graph = compile('{}.pt'.format(file_name))
    #     graph.save(file_name)
    #     newgraph = Graph()
    #     newgraph.graph_init(file_name + '/conf.yaml', file_name + '/model.bin')
    #     out = newgraph.inference([example_in.numpy(), example_in2.numpy()])

    #     np.testing.assert_almost_equal(ref_out, [*out.values()][0], decimal=5)
    #     os.remove('{}.pt'.format(file_name))
    #     shutil.rmtree(file_name)


if __name__ == "__main__":
    unittest.main()
