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
from intel_extension_for_transformers.backends.neural_engine.compile.graph import Graph
from intel_extension_for_transformers.backends.neural_engine.compile import compile
import numpy as np
import os
import sys

def is_win():
    return sys.platform.startswith('win')

class TestTranspose(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        pass

    @classmethod
    def tearDownClass(self):
        pass

    def test_transpose(self):
        os.environ['GLOG_minloglevel'] = '2'
        root_dir = '/tf_dataset2/inc-ut/nlptoolkit_ut_model/'
        if is_win():
            root_dir = 'D:\\dataset\\nlptoolkit_ut_model\\'
        model_dir = root_dir + 'onnx_best_acc_distilbert.merged.untransposed'
        pattern_config = root_dir + 'pattern_config'
        self.assertTrue(os.path.exists(model_dir),
                        'INT8 IR model is not found, please set your own model path!')
        graph = Graph()
        graph.graph_init(model_dir + '/conf.yaml', model_dir + '/model.bin', load_weight=True)
        graph.save('./mergedQK')
        model = Graph()
        model.graph_init('./mergedQK/conf.yaml', './mergedQK/model.bin', load_weight=True)
        model.transpose_mode_int8()

        model_dir_2 = root_dir + 'Intel.bert-mini-sst2-distilled-sparse-90-1X4-block.mergedQKV.untransposed'
        self.assertTrue(os.path.exists(model_dir_2),
                        'INT8 IR model is not found, please set your own model path!')
        graph_2 = Graph()
        graph_2.graph_init(model_dir_2 + '/conf.yaml', model_dir_2 + '/model.bin', load_weight=True)
        graph_2.save('./mergedQKV')
        model_2 = Graph()
        model_2.graph_init('./mergedQKV/conf.yaml', './mergedQKV/model.bin', load_weight=True)
        model_2.transpose_mode_int8()

        model_dir_3 = root_dir + 'bert_mini_int8_original_IR'
        self.assertTrue(os.path.exists(model_dir),
                        'INT8 IR model is not found, please set your own model path!')
        model_3 = Graph()
        model_3.graph_init(model_dir_3 + '/conf.yaml', model_dir_3 + '/model.bin', load_weight=True)
        model_3.transpose_mode_int8()


if __name__ == '__main__':
    unittest.main()
