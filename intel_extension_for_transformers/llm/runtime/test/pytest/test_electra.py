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
from intel_extension_for_transformers.backends.neural_engine.compile import compile, autocast
from transformers import  AutoTokenizer
import numpy as np
import os
import sys
import torch
import copy


def is_win():
    return sys.platform.startswith('win')


class TestElectra(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        pass

    @classmethod
    def tearDownClass(self):
        pass

    def test_electra(self):
        root_dir = '/tf_dataset2/models/nlp_toolkit/chinese-legal-electra-base-generator/'
        if is_win():
            root_dir = 'D:\\dataset\\chinese-legal-electra-base-generator\\'
        model_dir = root_dir + 'model.pt'
        pattern_config = root_dir + 'bf16_pattern.conf'
        self.assertTrue(os.path.exists(model_dir), 'model is not found, please set your own model path!')

        # inputs
        tokenizer = AutoTokenizer.from_pretrained(root_dir)
        text = "其实了解一个人并不代[MASK]什么，人是会变的，今天他喜欢凤梨，明天他可以喜欢别的"
        inputs = tokenizer(text, return_tensors="pt")

        # fp32 pt model
        jit_model = torch.jit.load(model_dir)
        jit_out = jit_model(inputs.input_ids, inputs.attention_mask, inputs.token_type_ids)[0]

        # fp32 engine model
        fp32_eng = compile(model_dir)
        bs = inputs.input_ids.size()[0]
        seq_len = inputs.input_ids.size()[1]
        fp32_eng_out = copy.deepcopy(list(fp32_eng.inference([inputs.input_ids.detach().numpy(),
                                    inputs.attention_mask.detach().numpy(),
                                    inputs.token_type_ids.detach().numpy()]).values())[0]).reshape(bs, seq_len, -1)

        # bf16 engine model
        with autocast("bf16"):
            bf16_eng = compile(model_dir, pattern_config)
        bf16_eng_out = copy.deepcopy(list(bf16_eng.inference([inputs.input_ids.detach().numpy(),
                                    inputs.attention_mask.detach().numpy(),
                                    inputs.token_type_ids.detach().numpy()]).values())[0]).reshape(bs, seq_len, -1)

        fp32_flag = np.allclose(jit_out.detach().numpy(), fp32_eng_out, atol=1e0)
        bf16_flag = np.allclose(jit_out.detach().numpy(), bf16_eng_out, atol=1e0)
        self.assertTrue(fp32_flag)
        self.assertTrue(bf16_flag)


if __name__ == '__main__':
    unittest.main()
