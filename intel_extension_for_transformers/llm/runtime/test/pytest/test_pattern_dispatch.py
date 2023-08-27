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
from intel_extension_for_transformers.backends.neural_engine.compile.loaders.loader import Loader
from intel_extension_for_transformers.backends.neural_engine.compile.extractors.extractor import Extractor
from intel_extension_for_transformers.backends.neural_engine.compile.sub_graph.subgraph_matcher import SubGraphMatcher
import sys
import copy

def is_win():
    return sys.platform.startswith('win')

class TestPatternDispatch(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        pass

    @classmethod
    def tearDownClass(self):
        pass

    def test_pattern_dispatch(self):
        # set input data
        shape = [1, 128]
        input_0 = np.random.uniform(low=0, high=128, size=shape).astype('int32')
        input_1 = np.random.uniform(low=0, high=1, size=shape).astype('int32')
        input_2 = np.random.uniform(low=0, high=1, size=shape).astype('int32')
    
        # validate pattern tuning
        fp32_model_path = "/tf_dataset2/inc-ut/nlptoolkit_ut_model/bert_mini_sst2_1x4_fp32.onnx"
        if is_win():
            fp32_model_path = "D:\\dataset\\nlptoolkit_ut_model\\bert_mini_sst2_1x4_fp32.onnx"
        self.assertTrue(os.path.exists(fp32_model_path),
            'FP32 ONNX model is not found, please set your own model path!')
        fp32_model = compile(fp32_model_path)
        fp32_output_dict = copy.deepcopy(fp32_model.inference([input_0, input_1, input_2]))
        fp32_output = list(fp32_output_dict.values())[0]
        # pattern tuning
        load = Loader()
        extract = Extractor()
        subgraph_match = SubGraphMatcher()
        fp32_model_tune = load(fp32_model_path)
        fp32_model_tune = extract(fp32_model_tune)
        fp32_model_tune = subgraph_match(fp32_model_tune, tune = True)
        fp32_tune_output_dict = copy.deepcopy(fp32_model_tune.inference([input_0, input_1, input_2]))
        fp32_tune_output = list(fp32_tune_output_dict.values())[0]
        # compare outputs
        self.assertTrue((fp32_output == fp32_tune_output).all())

if __name__ == "__main__":
    unittest.main()
