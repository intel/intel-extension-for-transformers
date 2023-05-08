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
import shutil
from intel_extension_for_transformers.backends.neural_engine.compile.ops.op import OPERATORS
from intel_extension_for_transformers.backends.neural_engine.compile.ops.tensor import Tensor
from intel_extension_for_transformers.backends.neural_engine.compile.graph import Graph
from intel_extension_for_transformers.backends.neural_engine.compile import compile, autocast
import copy


def fp32_to_bf16(fp32_np):
    if fp32_np.dtype == np.int16:
        return fp32_np
    tmp = copy.deepcopy(fp32_np)
    tmp = tmp.view(dtype=np.int32)
    tmp = tmp >> 16
    tmp = tmp.astype(np.int16)
    return tmp

class TestExecutionOptions(unittest.TestCase):
    @classmethod
    def setUpClass(self):

        self.ir_path = 'optimizer_ir'
        graph = Graph()
        input_data_node = OPERATORS['Input']()
        input_tensors = []
        output_tensors = [Tensor(name="activation", shape=[-1, -1], dtype="bf16")]
        input_data_node.construct('input_data', 'Input', input_tensors=input_tensors,
                                output_tensors=output_tensors)
        ip_node = OPERATORS['InnerProduct']()
        input_tensors = [Tensor(name="activation", shape=[-1, -1], dtype="bf16"),
                        Tensor(name="weight", shape=[256, 256], dtype="bf16",
                        data=fp32_to_bf16(np.random.randn(256, 256).astype(np.float32))),
                        Tensor(name="bias", shape=[256], dtype="bf16",
                        data=fp32_to_bf16(np.random.randn(256).astype(np.float32)))]
        output_tensors = [Tensor(name='ip:0', source_op=['ip'], dest_op=['output_data'])]
        ip_node.construct('ip', 'InnerProduct', input_tensors=input_tensors,
                                output_tensors=output_tensors)
        output_node = OPERATORS['Output']()
        input_tensors = [Tensor(name='ip:0', source_op=['ip'], dest_op=['output_data'])]
        output_tensors = []
        output_node.construct('output_data', 'Output', input_tensors=input_tensors,
                                output_tensors=output_tensors)
        graph.insert_nodes(len(graph.nodes), [input_data_node, ip_node, output_node])
        graph.save(self.ir_path)
        del graph

    @classmethod
    def tearDownClass(self):
        shutil.rmtree(self.ir_path)

    def test_fp8_weight_compression(self):
        graph = None
        data = fp32_to_bf16(np.random.randn(128, 256).astype(np.float32))
        graph = compile(self.ir_path)
        g_ret = copy.deepcopy(graph.inference([data])['ip:0'])
        fp8_ret = []
        for w_tag in ['any', 'int8', 'fp8_5e2m', 'fp8_4e3m']:
            with autocast('bf16', weight_dtype= w_tag):
                graph = compile(self.ir_path)
                fp8_ret.append(copy.deepcopy(graph.inference([data])['ip:0']))

        flag = True
        for ret in fp8_ret:
            flag = np.allclose(g_ret, ret, atol=1e0, equal_nan=True)
            if not flag:
                break
        self.assertTrue(flag)

if __name__ == "__main__":
    unittest.main()
