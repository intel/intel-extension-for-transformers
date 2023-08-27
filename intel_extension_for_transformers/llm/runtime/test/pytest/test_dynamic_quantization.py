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
from collections import OrderedDict
from intel_extension_for_transformers.backends.neural_engine.compile.dynamic_quantize import _dynamic_quantization, _quantize2bf16
from intel_extension_for_transformers.backends.neural_engine.compile.ops.op import OPERATORS, Operator
from intel_extension_for_transformers.backends.neural_engine.compile.ops.tensor import Tensor
from intel_extension_for_transformers.backends.neural_engine.compile.graph import Graph
import numpy as np
import os
from copy import deepcopy


class TestDynamicQuantization(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        pass

    @classmethod
    def tearDownClass(self):
        pass

    def test_dynamic_quantization(self):
        np.random.seed(1)
        os.environ['GLOG_minloglevel'] = '2'
        graph = Graph()
        input_data_node = OPERATORS['Input']()
        input_tensors = []
        output_tensors = [
            Tensor(name='input0', shape=[-1], dtype="fp32"),
        ]
        input_data_node.construct('input_data', 'Input', input_tensors=input_tensors, output_tensors=output_tensors)

        Layernorm_node = OPERATORS['Reshape']()
        input_tensors = [
            Tensor(name='input0', dest_op=['reshape'], dtype="fp32"),
        ]
        output_tensors = [
            Tensor(name='layernorm_output', source_op=['reshape'], dest_op=['Q', "K", "V", "Att"], dtype="fp32")
        ]
        Layernorm_node.construct('reshape',
                                 'Reshape',
                                 input_tensors=input_tensors,
                                 output_tensors=output_tensors,
                                 attr=OrderedDict({'dst_shape': "-1,256"}))

        Q_node = OPERATORS['InnerProduct']()
        input_tensors = [
            Tensor(name='layernorm_output', source_op=['layernorm'], dest_op=['Q'], dtype="fp32"),
            Tensor(name='weight_Q',
                   dest_op=['Q'],
                   shape=[256, 256],
                   data=np.array(np.random.random((256, 256)) * 2 - 1, dtype="float32"),
                   dtype="fp32"),
            Tensor(name='bias_Q',
                   dest_op=['Q'],
                   shape=[256],
                   data=np.array(np.random.random((256)), dtype="float32"),
                   dtype="fp32"),
        ]
        output_tensors = [Tensor(name='q', source_op=['Q'], dest_op=['qk_matmul'])]
        Q_node.construct('Q',
                         'InnerProduct',
                         input_tensors=input_tensors,
                         output_tensors=output_tensors,
                         attr=OrderedDict({
                             'src1_perm': '1,0',
                             'reshape': '1,-1,4,64',
                         }))
        K_node = OPERATORS['InnerProduct']()
        input_tensors = [
            Tensor(name='layernorm_output', source_op=['layernorm'], dest_op=['K'], dtype="fp32"),
            Tensor(name='weight_K',
                   dest_op=['K'],
                   shape=[256, 256],
                   data=np.array(np.random.random((256, 256)) * 2 - 1, dtype="float32"),
                   dtype="fp32"),
            Tensor(name='bias_K',
                   dest_op=['K'],
                   shape=[256],
                   data=np.array(np.random.random((256)), dtype="float32"),
                   dtype="fp32"),
        ]
        output_tensors = [Tensor(name='k', source_op=['K'], dest_op=['qk_matmul'], dtype="fp32")]
        K_node.construct('K',
                         'InnerProduct',
                         input_tensors=input_tensors,
                         output_tensors=output_tensors,
                         attr=OrderedDict({
                             'src1_perm': '1,0',
                             'reshape': '1,-1,4,64',
                         }))
        V_node = OPERATORS['InnerProduct']()
        input_tensors = [
            Tensor(name='layernorm_output', source_op=['layernorm'], dest_op=['V'], dtype="fp32"),
            Tensor(name='weight_V',
                   dest_op=['V'],
                   shape=[256, 256],
                   data=np.array(np.random.random((256, 256)) * 2 - 1, dtype="float32"),
                   dtype="fp32"),
            Tensor(name='bias_V',
                   dest_op=['V'],
                   shape=[256],
                   data=np.array(np.random.random((256)), dtype="float32"),
                   dtype="fp32"),
        ]
        output_tensors = [Tensor(name='v', source_op=['V'], dest_op=['av_matmul'], dtype="fp32")]
        V_node.construct('V',
                         'InnerProduct',
                         input_tensors=input_tensors,
                         output_tensors=output_tensors,
                         attr=OrderedDict({
                             'src1_perm': '1,0',
                             'reshape': '1,-1,4,64',
                         }))

        qk_node = OPERATORS['Matmul']()
        input_tensors = [
            Tensor(name='q', source_op=['Q'], dest_op=['qk_matmul'], dtype="fp32"),
            Tensor(name='k', source_op=['K'], dest_op=['qk_matmul'], dtype="fp32"),
            Tensor(name='mask',
                   dest_op=['qk_matmul'],
                   shape=[256],
                   data=np.array(np.random.random((256)), dtype="float32"),
                   dtype="fp32"),
        ]
        output_tensors = [Tensor(name='qk', source_op=['qk_matmul'], dest_op=['softmax'], dtype="fp32")]
        qk_node.construct('qk_matmul',
                          'Matmul',
                          input_tensors=input_tensors,
                          output_tensors=output_tensors,
                          attr=OrderedDict({
                              'src0_perm': '0,2,1,3',
                              'src1_perm': '0,2,3,1',
                              'output_scale': '0.125',
                              'format_any': "false",
                              'append_op': 'binary_add'
                          }))

        softmax_node = OPERATORS['Softmax']()
        input_tensors = [Tensor(name='qk', source_op=['qk_matmul'], dest_op=['softmax'], dtype="fp32")]
        output_tensors = [Tensor(name='a', source_op=['softmax'], dest_op=['av_matmul', 'output_data'], dtype="fp32")]
        softmax_node.construct('softmax', 'Softmax', input_tensors=input_tensors, output_tensors=output_tensors)

        av_node = OPERATORS['Matmul']()
        input_tensors = [
            Tensor(name='a', source_op=['softmax'], dest_op=['av_matmul'], dtype="fp32"),
            Tensor(name='v', source_op=['V'], dest_op=['av_matmul'], dtype="fp32")
        ]
        output_tensors = [Tensor(name='av', source_op=['av_matmul'], dest_op=['Att'], dtype="fp32")]
        av_node.construct('av_matmul',
                          'Matmul',
                          input_tensors=input_tensors,
                          output_tensors=output_tensors,
                          attr=OrderedDict({
                              'src1_perm': '0,2,1,3',
                              'dst_perm': '0,2,1,3',
                              'reshape': "-1,256"
                          }))

        Att_node = OPERATORS['InnerProduct']()
        input_tensors = [
            Tensor(name='av', source_op=['av_matmul'], dest_op=['Att'], dtype="fp32"),
            Tensor(name='weight_Att',
                   dest_op=['Att'],
                   shape=[256, 256],
                   data=np.array(np.random.random((256, 256)) * 2 - 1, dtype="float32"),
                   dtype="fp32"),
            Tensor(name='bias_Att',
                   dest_op=['Att'],
                   shape=[256],
                   data=np.array(np.random.random((256)), dtype="float32"),
                   dtype="fp32"),
            Tensor(name='layernorm_output', source_op=['layernorm'], dest_op=['Att'], dtype="fp32"),
        ]
        output_tensors = [Tensor(name='att', source_op=['Att'], dest_op=['output_data'])]
        Att_node.construct('Att',
                           'InnerProduct',
                           input_tensors=input_tensors,
                           output_tensors=output_tensors,
                           attr=OrderedDict({
                               'src1_perm': '1,0',
                               "append": "sum",
                           }))

        output_node = OPERATORS['Output']()
        input_tensors = [Tensor(name='att', source_op=['Att'], dest_op=['output_data'], dtype="fp32")]
        output_tensors = []
        output_node.construct(
            'output_data',
            'Output',
            input_tensors=input_tensors,
            output_tensors=output_tensors,
        )

        graph.insert_nodes(len(graph.nodes), [
            input_data_node, Layernorm_node, Q_node, K_node, V_node, qk_node, softmax_node, av_node, Att_node,
            output_node
        ])
        int8_model = _dynamic_quantization(deepcopy(graph))
        graph.save("fp32")
        int8_model.save("dq")
        int8_model.get_node_by_name('output_data').input_tensors.append(
            Tensor(name='qk', source_op=['qk_matmul'], dest_op=['output_data'], dtype="fp32"))
        graph.get_node_by_name('output_data').input_tensors.append(
            Tensor(name='qk', source_op=['qk_matmul'], dest_op=['output_data'], dtype="fp32"))

        # int8_model.get_node_by_name('output_data').input_tensors.append(Tensor(name='a_quant', source_op=['softmax'], dest_op=['output_data'], dtype="u8"))
        # graph.get_node_by_name('output_data').input_tensors.append(Tensor(name='a', source_op=['softmax'], dest_op=['output_data'], dtype="fp32"))

        input_data = np.ones((32 * 256), dtype="float32")
        self.assertTrue(True)

        Convolution_node = OPERATORS['Convolution']()
        input_tensors = [
            Tensor(name='input0', dest_op=['av_matmul'], dtype="fp32"),
            Tensor(name='weight',
                   shape=[256],
                   dest_op=['av_matmul'],
                   data=np.array(np.random.random((256)), dtype="float32"),
                   dtype="fp32"),
            Tensor(name='bias', shape=[256], dest_op=['av_matmul'], data=np.zeros((256), dtype="float32"), dtype="fp32")
        ]
        output_tensors = [Tensor(name='av', source_op=['av_matmul'], dest_op=["output"], dtype="fp32")]
        Convolution_node.construct('av_matmul',
                                   'Convolution',
                                   input_tensors=input_tensors,
                                   output_tensors=output_tensors,
                                   attr=OrderedDict({}))
        graph.insert_nodes(3, [input_data_node, Convolution_node, output_node])
        int8_model = _dynamic_quantization(deepcopy(graph))
        bf16_model = _quantize2bf16(int8_model)


if __name__ == "__main__":
    unittest.main()
