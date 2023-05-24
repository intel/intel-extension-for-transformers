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

def is_win():
    return sys.platform.startswith('win')

class TestTorchModel(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        pass

    @classmethod
    def tearDownClass(self):
        pass

    def test_1(self):
        pt_file ='/tf_dataset2/inc-ut/nlptoolkit_ut_model/bert_mini_fp32.pt'
        if is_win():
            pt_file = "D:\\dataset\\nlptoolkit_ut_model\\bert_mini_fp32.pt"
        self.assertTrue(os.path.exists(pt_file),
            'INT8 IR model is not found, please set your own model path!')
        ids = torch.LongTensor([[1, 2, 3]])
        tok = torch.zeros_like(ids)
        att = torch.ones_like(ids)

        traced_model = torch.jit.load(pt_file)
        example_in = torch.rand(8, 128)
        # TODO: enable check accuracy
        ref_out = traced_model(ids, tok, att, ids)[0].detach().numpy()
        
        graph = compile(pt_file)
        graph.save(file_name)
        newgraph = Graph()
        newgraph.graph_init(file_name + '/conf.yaml', file_name + '/model.bin', load_weight=True)
        self.assertTrue(newgraph.nodes[-1].name == 'output_data')
        self.assertTrue(newgraph.nodes[-1].input_tensors[-1].name == '268')
        # out = newgraph.inference([ids.numpy(), tok.numpy(), att.numpy(), ids.numpy()])

        # np.testing.assert_almost_equal(ref_out, [*out.values()][0], decimal=5)
        shutil.rmtree(file_name)

if __name__ == "__main__":
    unittest.main()
