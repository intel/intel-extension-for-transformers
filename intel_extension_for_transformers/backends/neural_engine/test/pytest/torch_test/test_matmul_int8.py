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
import intel_extension_for_pytorch as ipex
from intel_extension_for_pytorch.quantization import prepare, convert

os.environ["LLGA_DISABLE"] = '1'
file_name = os.path.splitext(os.path.basename(__file__))[0]

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

    def forward(self, x, y):
        return torch.matmul(x, y)

def cmpData(numa, numb):
    if (numa.shape != numb.shape):
        return 1
    totalErr = ((np.abs(numa - numb))**2).sum()
    totalNum = (np.abs(numa)**2).sum()
    return np.sqrt(totalErr/totalNum)

class TestTorchOP(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        pass

    @classmethod
    def tearDownClass(self):
        pass

    def test_1(self):
        from torch.ao.quantization import MinMaxObserver, PerChannelMinMaxObserver, QConfig
        qconfig = QConfig(activation=MinMaxObserver.with_args(qscheme=torch.per_tensor_affine, dtype=torch.qint8),
                        weight=MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric))
        n = Net().eval()
        example_in = torch.rand(3, 256)
        example_in2 = torch.rand(256, 10)
        prepared_model = prepare(n, qconfig, example_inputs=(example_in, example_in2), inplace=False)
        prepared_model(example_in, example_in2)
        convert_model = convert(prepared_model)
        traced_model = torch.jit.trace(convert_model, (example_in, example_in2))
        print(traced_model.inlined_graph)
        
        torch.jit.freeze(traced_model.eval())
        torch.jit.save(traced_model, '{}.pt'.format(file_name))
        ref_out = traced_model(example_in, example_in2).detach().numpy()
        
        graph = compile('{}.pt'.format(file_name))
        graph.save(file_name)
        newgraph = Graph()
        newgraph.graph_init(file_name + '/conf.yaml', file_name + '/model.bin')
        out = newgraph.inference([example_in.numpy(), example_in2.numpy()])

        self.assertTrue(cmpData(ref_out, [*out.values()][0]) < 0.001)
        os.remove('{}.pt'.format(file_name))
        shutil.rmtree(file_name)

if __name__ == "__main__":
    unittest.main()
