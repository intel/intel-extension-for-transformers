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
from intel_extension_for_transformers.backends.neural_engine.compile import compile
from intel_extension_for_transformers.backends.neural_engine.compile.graph import Graph
import sys
import os

def is_win():
    return sys.platform.startswith('win')

class TestQKVMerge(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        os.remove('qkv_merge_pattern_config')
        pass

    def test_qkv_merge_1(self):
        model_path = "/tf_dataset2/inc-ut/nlptoolkit_ut_model/onnx_best_acc_distilbert.onnx"
        content = "pattern_switch:\n  'QKVMerge': True\n  'MultiHeadAttention': False"
        pattern_config = "qkv_merge_pattern_config" 
        with open("qkv_merge_pattern_config", "w") as file:
            file.write(content)
        if is_win():
            model_path = "D:\\dataset\\nlptoolkit_ut_model\\onnx_best_acc_distilbert.onnx"
            pattern_config = "D:\\dataset\\nlptoolkit_ut_model\\qkv_merge_pattern_config"
        graph = compile(model_path, config=pattern_config)
        self.assertEqual(100, len(graph.nodes))

if __name__ == "__main__":
    unittest.main()
