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
import sys
import unittest
import numpy as np
from intel_extension_for_transformers.backends.neural_engine.compile import compile
import copy

def is_win():
    return sys.platform.startswith('win')
class TestQuantOnnxExecute(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        pass

    @classmethod
    def tearDownClass(self):
        pass

    def test_onnx_qlinear_compile(self):
        # set input data
        input_0 = np.random.uniform(low=1, high=128, size=[1, 128]).astype('int32')
        input_1 = np.random.uniform(low=1, high=1, size=[1, 128]).astype('int32')
        input_2 = np.random.uniform(low=1, high=1, size=[1, 128]).astype('int32')
        # compile and execute qlinear model
        qlinear_model_path = "/tf_dataset2/inc-ut/nlptoolkit_ut_model/qlinear/bert_mini_sst2_qlinear.onnx"
        if is_win():
            qlinear_model_path = "D:\\dataset\\nlptoolkit_ut_model\\qlinear\\bert_mini_sst2_qlinear.onnx"
        os.environ['GLOG_minloglevel'] = '2'
        self.assertTrue(os.path.exists(qlinear_model_path),
            'ONNX QLinear model is not found, please set your own model path!')
        qlinear_model = compile(qlinear_model_path)
        qlinear_output_dict = qlinear_model.inference([input_0, input_1, input_2])
        qlinear_output = copy.deepcopy(list(qlinear_output_dict.values())[0])       
        # compile and execute qdq model
        qdq_model_path = "/tf_dataset2/inc-ut/nlptoolkit_ut_model/qlinear/bert_mini_sst2_qdq.onnx"
        if is_win():
            qdq_model_path = "D:\\dataset\\nlptoolkit_ut_model\\qlinear\\bert_mini_sst2_qdq.onnx"
        self.assertTrue(os.path.exists(qdq_model_path),
            'ONNX QDQ model is not found, please set your own model path!')
        qdq_model = compile(qdq_model_path)
        qdq_output_dict = qdq_model.inference([input_0, input_1, input_2])
        qdq_output = copy.deepcopy(list(qdq_output_dict.values())[0])
        # compare outputs
        self.assertTrue((qlinear_output == qdq_output).all())

if __name__ == "__main__":
    unittest.main()
