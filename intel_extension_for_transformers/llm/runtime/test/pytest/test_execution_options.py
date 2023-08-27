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
import numpy as np
import os
from intel_extension_for_transformers.backends.neural_engine.compile.ops.op import OPERATORS
from intel_extension_for_transformers.backends.neural_engine.compile.ops.tensor import Tensor
from intel_extension_for_transformers.backends.neural_engine.compile.graph import Graph
import copy


class TestExecutionOptions(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        pass

    @classmethod
    def tearDownClass(self):
        pass

    def test_execution_options(self):
        graph = Graph()
        graph.framework_modeling_config['framework'] = 'onnxruntime'
        input_data_node = OPERATORS['Input']()
        input_tensors = []
        output_tensors = [Tensor(name="activation", shape=[-1, -1], dtype="fp32")]
        input_data_node.construct('input_data', 'Input', input_tensors=input_tensors, 
                                output_tensors=output_tensors)
        ip_node = OPERATORS['InnerProduct']()
        input_tensors = [Tensor(name="activation", shape=[-1, -1], dtype="fp32"),
                        Tensor(name="weight", shape=[256, 256], dtype="fp32",
                        data=np.random.randn(256, 256).astype(np.float32)),
                        Tensor(name="bias", shape=[256], dtype="fp32",
                        data=np.random.randn(256).astype(np.float32))]
        output_tensors = [Tensor(name='ip:0', source_op=['ip'], dest_op=['output_data']),
                          Tensor(name='ip:1', source_op=['ip'], dest_op=['output_data'])]
        ip_node.construct('ip', 'InnerProduct', input_tensors=input_tensors,
                                output_tensors=output_tensors)
        output_node = OPERATORS['Output']()
        input_tensors = [Tensor(name='ip:0', source_op=['ip'], dest_op=['output_data'])]
        output_tensors = []
        output_node.construct('output_data', 'Output', input_tensors=input_tensors,
                                output_tensors=output_tensors)
        graph.insert_nodes(len(graph.nodes), [input_data_node, ip_node, output_node])
        graph.change_node_output_tensors('ip', -1, mode='remove')
        options = {'enable_op_tuning': True,
                   'dispatch_table_file_root': 'dispatch_table.txt',
                   'execution_mode': 'tuning',
                   'warmup_iter': 2
                  }
        data = np.random.randn(128, 256).astype(np.float32)
        graph.execution_options = options
        output_tuning = []
        for i in range(10):
            out = graph.inference([data])
            output_tuning.append(copy.deepcopy(out['ip:0']))
        output_inference = []
        options = {'enable_op_tuning': False,
                   'dispatch_table_file_root': 'dispatch_table.txt',
                   'execution_mode': 'inference'
                   }
        graph.execution_options = options
        for i in range(10):
            out = graph.inference([data])
            output_inference.append(copy.deepcopy(out['ip:0']))
        flag = np.allclose(np.array(output_tuning), np.array(output_inference), atol=1e-1,
                           equal_nan=True)
        self.assertTrue(os.path.exists(graph.execution_options.dispatch_table_file_root))
        self.assertFalse(graph.execution_options.enable_op_tuning)
        self.assertTrue(flag)
        os.remove(graph.execution_options.dispatch_table_file_root)

if __name__ == "__main__":
    unittest.main()
