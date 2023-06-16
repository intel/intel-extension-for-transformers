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
from intel_extension_for_transformers.backends.neural_engine.compile import compile
from transformers import BertForSequenceClassification
import numpy as np
import os
import sys
import torch
import copy
import shutil


def is_win():
    return sys.platform.startswith('win')


class TestSCBuffer(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        # export onnx model
        model_path = "/tf_dataset2/models/nlp_toolkit/bert_mini_mrpc"
        torch_model = BertForSequenceClassification.from_pretrained(model_path)
        with torch.no_grad():
            inputs = {
                'input_ids': torch.ones(1, 128, dtype=torch.int32),
                'attention_mask': torch.ones(1, 128, dtype=torch.int32),
                'token_type_ids': torch.ones(1, 128, dtype=torch.int32)
            }
            outputs = torch_model(**inputs)

            symbolic_names = {0: 'batch_size', 1: 'max_seq_len'}
            torch.onnx.export(
                torch_model,
                (inputs['input_ids'], inputs['attention_mask'], inputs['token_type_ids']),
                "onnx_fp32.onnx",
                opset_version=11,
                do_constant_folding=True,
                input_names=['input_ids', 'input_mask', 'segment_ids'],
                output_names=['output'],
                dynamic_axes={
                    'input_ids': symbolic_names,
                    'input_mask': symbolic_names,
                    'segment_ids': symbolic_names
                })
        graph = compile("onnx_fp32.onnx")
        graph.save()

    @classmethod
    def tearDownClass(self):
        os.remove("onnx_fp32.onnx")
        shutil.rmtree("./ir", ignore_errors=True)
        if os.path.isfile("activation_dag.yaml"):
            os.remove("activation_dag.yaml")

    def test_static_compressed_buffer(self):
        graph_true = compile("ir/")
        inputs = [np.array([100, 101, 104, 90]).astype(np.int32).reshape(1,-1),
                  np.array([1,1,1,1]).astype(np.int32).reshape(1,-1),
                  np.array([1,1,1,1]).astype(np.int32).reshape(1,-1)]
        data_true = copy.deepcopy(list(graph_true.inference(inputs).values())[0])

        graph_test = compile("ir/")
        # set execution options
        options = {'activation_mem_compression': True,
                   'dump_activation_dag': True,
                   'execution_mode': 'inference'}
        graph_test.execution_options = options
        graph_test.max_input_shapes_list = [[[1, 128], [1, 128], [1, 128]],
                                            [[2, 128], [2, 128], [2, 128]]
                                           ]
        out1 = copy.deepcopy(list(graph_test.inference(inputs).values())[0])

        # debug mode, no in-place
        options = {'activation_mem_compression': True,
                   'dump_activation_dag': True,
                   'execution_mode': 'debug'}
        graph_test.execution_options = options
        graph_test.max_input_shapes_list = [[[1, 128], [1, 128], [1, 128]],
                                            [[2, 128], [2, 128], [2, 128]]
                                           ]
        out2 = copy.deepcopy(list(graph_test.inference(inputs).values())[0])

        self.assertTrue(np.allclose(data_true, out1, atol=1e-4))
        self.assertTrue(np.allclose(data_true, out2, atol=1e-4))
        self.assertTrue(os.path.exists("activation_dag.yaml"))

if __name__ == '__main__':
    unittest.main()
