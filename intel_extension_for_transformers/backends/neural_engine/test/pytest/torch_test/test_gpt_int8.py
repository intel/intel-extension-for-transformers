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
import copy
import json
from intel_extension_for_transformers.backends.neural_engine.compile import compile
from intel_extension_for_transformers.backends.neural_engine.compile.graph import Graph

file_name = os.path.splitext(os.path.basename(__file__))[0]


def is_win():
    return sys.platform.startswith('win')

def cmpData(numa, numb):
    numa = numa.flatten()
    numb = numb.flatten()
    if (numa.shape != numb.shape):
        return 1
    totalErr = ((np.abs(numa - numb))**2).sum()
    totalNum = (np.abs(numa)**2).sum()
    return np.sqrt(totalErr/totalNum)

def fp32_to_bf16(fp32_np):
    assert(fp32_np.dtype==np.float32)
    temp = copy.deepcopy(fp32_np)
    int32_np = temp.view(dtype=np.int32)
    int32_np = int32_np >> 16
    bf16_np = int32_np.astype(np.uint16)
    return bf16_np

class TestTorchModel(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        pass

    @classmethod
    def tearDownClass(self):
        pass

    def test_1(self):
        if is_win():
            return
        with open('int8_pattern.conf', 'w')  as f:
            data = {'pattern_switch': {
                            'Int8BF16MixedPrecisionChecker': True,
                            'MultiHeadAttention': True
                            }}
            json.dump(data, f)
        init_input_ids = torch.ones(32).long()
        init_input_ids[0] = 7454
        input_ids = init_input_ids.clone()
        attention_mask = torch.ones(len(input_ids)+1)
        attention_mask[0] = 0
        position_ids = torch.arange(len(input_ids))
        past_key_value_torch = tuple([(torch.zeros([1,16,32,256]), torch.zeros([1,16,32,256])) for i in range(28)])
        past_key_value = tuple([(torch.zeros([1,32,16,256]), torch.zeros([1,32,16,256])) for i in range(28)])
        input_ids = input_ids[0:1].unsqueeze(0)
        attention_mask = attention_mask.unsqueeze(0)
        position_ids = position_ids[0:1].unsqueeze(0)
        pt_file ='/tf_dataset2/models/nlp_toolkit/gpt-j/best_model_bk.pt'
        traced_model = torch.jit.load(pt_file)
        ref_out = traced_model(input_ids, past_key_value_torch, attention_mask, position_ids)

        graph = compile(pt_file, './int8_pattern.conf')
        out = graph.inference([input_ids.numpy(),
                                  past_key_value[0][0].numpy(),
                                  past_key_value[0][0].numpy(),
                                  past_key_value[0][0].numpy(),
                                  past_key_value[0][0].numpy(),
                                  past_key_value[0][0].numpy(),
                                  past_key_value[0][0].numpy(),
                                  past_key_value[0][0].numpy(),
                                  past_key_value[0][0].numpy(),
                                  past_key_value[0][0].numpy(),
                                  past_key_value[0][0].numpy(),

                                  past_key_value[0][0].numpy(),
                                  past_key_value[0][0].numpy(),
                                  past_key_value[0][0].numpy(),
                                  past_key_value[0][0].numpy(),
                                  past_key_value[0][0].numpy(),
                                  past_key_value[0][0].numpy(),
                                  past_key_value[0][0].numpy(),
                                  past_key_value[0][0].numpy(),
                                  past_key_value[0][0].numpy(),
                                  past_key_value[0][0].numpy(),

                                  past_key_value[0][0].numpy(),
                                  past_key_value[0][0].numpy(),
                                  past_key_value[0][0].numpy(),
                                  past_key_value[0][0].numpy(),
                                  past_key_value[0][0].numpy(),
                                  past_key_value[0][0].numpy(),
                                  past_key_value[0][0].numpy(),
                                  past_key_value[0][0].numpy(),
                                  past_key_value[0][0].numpy(),
                                  past_key_value[0][0].numpy(),

                                  past_key_value[0][0].numpy(),
                                  past_key_value[0][0].numpy(),
                                  past_key_value[0][0].numpy(),
                                  past_key_value[0][0].numpy(),
                                  past_key_value[0][0].numpy(),
                                  past_key_value[0][0].numpy(),
                                  past_key_value[0][0].numpy(),
                                  past_key_value[0][0].numpy(),
                                  past_key_value[0][0].numpy(),
                                  past_key_value[0][0].numpy(),
                                  past_key_value[0][0].numpy(),
                                  past_key_value[0][0].numpy(),
                                  past_key_value[0][0].numpy(),
                                  past_key_value[0][0].numpy(),
                                  past_key_value[0][0].numpy(),
                                  past_key_value[0][0].numpy(),
                                  past_key_value[0][0].numpy(),
                                  past_key_value[0][0].numpy(),
                                  past_key_value[0][0].numpy(),
                                  past_key_value[0][0].numpy(),

                                  past_key_value[0][0].numpy(),
                                  past_key_value[0][0].numpy(),
                                  past_key_value[0][0].numpy(),
                                  past_key_value[0][0].numpy(),
                                  past_key_value[0][0].numpy(),
                                  past_key_value[0][0].numpy(),
                                  attention_mask.numpy(),
                                  ])
        diff = cmpData(ref_out[0].detach().numpy(), out['ret2995.1'])
        print(diff)
        self.assertTrue(diff < 0.1)
        os.remove('int8_pattern.conf')

if __name__ == "__main__":
    unittest.main()