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
from intel_extension_for_transformers.backends.neural_engine.compile.ops.tensor import Tensor
from intel_extension_for_transformers.backends.neural_engine.compile.graph import Graph
import intel_extension_for_transformers.backends.neural_engine.compile.graph_utils as util
import numpy as np


class TestBinaryOp(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        pass

    @classmethod
    def tearDownClass(self):
        pass

    def test_binary_op(self):
        model = Graph()
        input_tensors = [Tensor(name="any", source_op=[], dest_op=['anyop'])]
        output_tensors = [Tensor(name="any_out", source_op=['anyop'], dest_op=['mul'])]
        any_node = util.construct_node('anyop', 'AnyOpTest', input_tensors=input_tensors,
                                       output_tensors=output_tensors)
        input_tensors = [Tensor(name="any_out", source_op=['anyop'], dest_op=['mul']),
                         Tensor(name='mul_1', data=np.array(1).astype("int64"), shape=[])]
        output_tensors = [Tensor(name='mul_out', source_op=['mul'], dest_op=['not'])]
        Mul_node = util.construct_node('mul', 'Mul', input_tensors=input_tensors,
                                       output_tensors=output_tensors)
        input_tensors = [Tensor(name='mul_out', source_op=['mul'], dest_op=['not'])]
        output_tensors = [Tensor(name='not_out', source_op=['not'], dest_op=['neg'])]
        Not_node = util.construct_node('not', 'Not', input_tensors=input_tensors,
                                        output_tensors=output_tensors)
        input_tensors = [Tensor(name='not_out', source_op=['not'], dest_op=['neg'])]
        output_tensors = [Tensor(name='neg_out', source_op=['neg'], dest_op=[])]
        Neg_node = util.construct_node('neg', 'Neg', input_tensors=input_tensors,
                                        output_tensors=output_tensors)
        Mul_node.set_attr("onnxruntime", None)
        Not_node.set_attr("onnxruntime", None)
        Neg_node.set_attr("onnxruntime", None)
        model.insert_nodes(0, [any_node, Mul_node, Not_node, Neg_node])
        config = {'architecture': 'Transformers', 'layer': 3}
        model.framework_modeling_config = config
        config_1 = model.framework_modeling_config
        val_0 = model.inquire_config_item('layer')
        val_1 = config_1['layer']
        id_0 = model.get_node_id('not')
        model.rename_node('not', 'not_new')
        id_1 = model.get_node_id('not_new')
        self.assertEqual(any_node.op_type, 'AnyOpTest')
        self.assertEqual(Mul_node.input_tensors[1].data.dtype, np.float32)
        self.assertEqual(len(Not_node.input_tensors), 2)
        self.assertEqual(len(Neg_node.input_tensors), 2)
        self.assertEqual(val_0, val_1)
        self.assertEqual(id_0, id_1)

if __name__ == "__main__":
    unittest.main()