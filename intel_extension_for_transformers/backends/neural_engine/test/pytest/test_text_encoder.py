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
import torch
import onnxruntime as ort


def is_win():
    return sys.platform.startswith('win')


class TestTextEncoder(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        pass

    @classmethod
    def tearDownClass(self):
        pass

    def test_text_encoder(self):
        os.environ['GLOG_minloglevel'] = '2'
        root_dir = '/tf_dataset2/models/nlp_toolkit/stable-diffusion/text_encoder/'
        if is_win():
            root_dir = 'D:\\dataset\\nlptoolkit_ut_model\\'
        model_dir = root_dir + 'model.onnx'
        pattern_config = root_dir + 'pattern_config'
        self.assertTrue(os.path.exists(model_dir), 'model is not found, please set your own model path!')

        graph = compile(model_dir, config=pattern_config)
        input_0_path = root_dir + 'input_ids.pt'
        inputs_0 = torch.load(input_0_path)

        output = graph.inference([inputs_0])
        for node_name in output.keys():
            print(node_name, ', shape = ', output[node_name].shape)

        # onnxruntime
        session = ort.InferenceSession(model_dir)
        x = torch.load(input_0_path).numpy().astype(np.int32)

        ortvalue = ort.OrtValue.ortvalue_from_numpy(x)
        ortvalue.device_name()

        outputs = session.run(None, {
            'input_ids': ortvalue,
        })

        for idx, output_ort in enumerate(outputs):
            print('onnxruntime output_shape = ', output_ort.shape)

        flag = np.allclose(output['last_hidden_state:0'], outputs[0], atol=1e-2)
        self.assertTrue(flag)


if __name__ == '__main__':
    unittest.main()
