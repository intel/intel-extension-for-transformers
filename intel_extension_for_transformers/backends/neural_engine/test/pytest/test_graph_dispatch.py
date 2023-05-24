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

import os
import unittest
import numpy as np
from intel_extension_for_transformers.backends.neural_engine.compile import compile
import sys
import copy
from intel_extension_for_transformers.backends.neural_engine.compile.graph import Graph

def is_win():
    return sys.platform.startswith('win')

class TestGraphDispatch(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        pass

    @classmethod
    def tearDownClass(self):
        pass

    def test_graph_dispatch(self):
        os.environ['GLOG_minloglevel'] = '2'
        # set input data
        shape = [1, 128]
        input_0 = np.random.uniform(low=0, high=128, size=shape).astype('int32')
        input_1 = np.random.uniform(low=0, high=1, size=shape).astype('int32')
        input_2 = np.random.uniform(low=0, high=1, size=shape).astype('int32')
    
        # validate int8 sparse graph tuning
        int8_model_path = "/tf_dataset2/inc-ut/nlptoolkit_ut_model/bert_mini_int8_original_IR"
        if is_win():
            int8_model_path = "D:\\dataset\\nlptoolkit_ut_model\\bert_mini_int8_original_IR"
        self.assertTrue(os.path.exists(int8_model_path),
            'INT8 IR model is not found, please set your own model path!')
        int8_model = Graph()
        int8_model.graph_init(int8_model_path + '/conf.yaml', int8_model_path + '/model.bin', load_weight=True)
        int8_output_dict = int8_model.inference([input_0, input_1, input_2])
        int8_output = copy.deepcopy(list(int8_output_dict.values())[0])
        # sparse graph tuning
        int8_model.graph_dispatch(inputs_shape = [shape, shape, shape])
        int8_dispatch_output_dict = int8_model.inference([input_0, input_1, input_2])
        int8_dispatch_output = copy.deepcopy(list(int8_dispatch_output_dict.values())[0])
        # compare outputs
        self.assertTrue((int8_output == int8_dispatch_output).all())

        # validate onednn graph tuning
        fp32_model_path = "/tf_dataset2/inc-ut/nlptoolkit_ut_model/bert_mini_sst2_1x4_fp32.onnx"
        if is_win():
            fp32_model_path = "D:\\dataset\\nlptoolkit_ut_model\\bert_mini_sst2_1x4_fp32.onnx"
        self.assertTrue(os.path.exists(fp32_model_path),
            'FP32 ONNX model is not found, please set your own model path!')
        fp32_model = compile(fp32_model_path)
        fp32_output_dict = fp32_model.inference([input_0, input_1, input_2])
        fp32_output = copy.deepcopy(list(fp32_output_dict.values())[0])
        # onednn graph tuning
        fp32_model.graph_dispatch(inputs_shape = [shape, shape, shape])
        fp32_dispatch_output_dict = fp32_model.inference([input_0, input_1, input_2])
        fp32_dispatch_output = copy.deepcopy(list(fp32_dispatch_output_dict.values())[0])
        # compare outputs
        self.assertTrue((fp32_output == fp32_dispatch_output).all())
        fp32_model._get_latency()
        fp32_model.dump_tensor()

if __name__ == "__main__":
    unittest.main()
