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
from intel_extension_for_transformers.backends.neural_engine.compile.ops.op import OPERATORS, Operator
from intel_extension_for_transformers.backends.neural_engine.compile.ops.tensor import Tensor
from intel_extension_for_transformers.backends.neural_engine.compile.graph import Graph
from intel_extension_for_transformers.backends.neural_engine.compile.sub_graph.add_embeddings import AddEmbeddings
import numpy as np


class TestAddEmbeddings(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        pass

    @classmethod
    def tearDownClass(self):
        pass

    def test_add_embeddings_with_seq_len_first(self):
        graph = Graph()
        graph.framework_modeling_config['framework'] = 'onnxruntime'
        input_data_node = OPERATORS['Input']()
        input_tensors = []
        output_tensors = [Tensor(), Tensor(), Tensor()]
        input_data_node.construct('input_data', 'Input', input_tensors=input_tensors, 
                                output_tensors=output_tensors)

        add_node = OPERATORS['Add']()
        input_tensors = [Tensor(name='add_src0'), Tensor(name='add_src1')]
        output_tensors = [Tensor(name='add:0', source_op=['add'], dest_op=['transpose'])]
        add_node.construct('add', 'Add', input_tensors=input_tensors, 
                                output_tensors=output_tensors)
        
        transpose_node = OPERATORS['Transpose']()
        input_tensors = [Tensor(name='add:0', source_op=['add'], dest_op=['transpose'])]
        output_tensors = [Tensor(name='transpose:0', source_op=['transpose'],
                                dest_op=['layernorm'])]
        transpose_node.construct('transpose', 'Transpose', input_tensors=input_tensors, 
                                output_tensors=output_tensors, attr=OrderedDict({
                                'src_perm': '0,1,2', 'dst_perm': '1,0,2'}))

        ln_node = OPERATORS['LayerNorm']()
        input_tensors = [Tensor(name='transpose:0', source_op=['transpose'],
                                dest_op=['layernorm']),
                         Tensor(name='scale:0', data=np.random.randn(1024).astype('float32'),
                                shape=[1024]),
                         Tensor(name='shift:0', data=np.random.randn(1024).astype("float32"),
                                shape=[1024])]
        output_tensors = [Tensor(name='layernorm:0', source_op=['layernorm'],
                                dest_op=[])]
        ln_node.construct('layernorm', 'LayerNorm', input_tensors=input_tensors, 
                          output_tensors=output_tensors, attr=OrderedDict({'epsilon': 0.009}))

        graph.insert_nodes(len(graph.nodes), [input_data_node, add_node, transpose_node, ln_node])
        graph.add_config_item('hidden_size', 1024)
        graph = AddEmbeddings()(graph)
        self.assertEqual(5, len(graph.nodes))
        self.assertEqual('-1,1024', graph.nodes[3].attr['dst_shape'])
        self.assertEqual('1,0,2', graph.nodes[2].attr['dst_perm'])
        self.assertEqual(0.009, graph.nodes[4].attr['epsilon'])


if __name__ == "__main__":
    unittest.main()
