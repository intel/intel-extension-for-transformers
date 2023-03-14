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
from intel_extension_for_transformers.backends.neural_engine.compile.sub_graph.layer_norm_with_transpose import LayerNormWithTranspose


class TestLayerNormWithTranspose(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        pass

    @classmethod
    def tearDownClass(self):
        pass

    def test_layer_norm_with_transpose(self):
        graph = Graph()
        graph.framework_modeling_config['framework'] = 'onnxruntime'
        input_data_node = OPERATORS['Input']()
        input_tensors = []
        output_tensors = [Tensor(), Tensor(), Tensor()]
        input_data_node.construct('input_data', 'Input', input_tensors=input_tensors, 
                                output_tensors=output_tensors)

        mat_node = OPERATORS['MatMulWithBiasAdd']()
        input_tensors = [Tensor(), Tensor(shape=[768]), Tensor(shape=[768]), Tensor()]
        output_tensors = [Tensor(name='mat:0', source_op=['mat'], dest_op=['layer_norm'])]
        mat_node.construct('mat', 'MatMulWithBiasAdd', input_tensors=input_tensors, 
                                output_tensors=output_tensors, attr=OrderedDict({
                                    'append_op': 'sum'}))

        ln_node = OPERATORS['LayerNorm']()
        input_tensors = [Tensor(name='mat:0', source_op=['mat'], dest_op=['layer_norm']),
                         Tensor(shape=[768]), Tensor(shape=[768])]
        output_tensors = [Tensor(name='layer_norm:0', source_op=['layer_norm'], 
                                 dest_op=['transpose'])]
        ln_node.construct('layer_norm', 'LayerNorm', input_tensors=input_tensors, 
                                output_tensors=output_tensors, attr=OrderedDict({
                                    'epsilon': 0.009}))
        
        transpose_node = OPERATORS['Transpose']()
        input_tensors = [Tensor(name='layer_norm:0', source_op=['layer_norm'], 
                                    dest_op=['transpose'])]
        output_tensors = [Tensor(name='transpose:0', source_op=['transpose'],
                                dest_op=[])]
        transpose_node.construct('transpose', 'Transpose', input_tensors=input_tensors, 
                                output_tensors=output_tensors, attr=OrderedDict(
                                    {'src_perm': '0,1,2', 'dst_perm': '1,0,2'}))

        graph.insert_nodes(len(graph.nodes), [input_data_node, mat_node, ln_node, transpose_node])
        graph.add_config_item('hidden_size', 768)
        graph = LayerNormWithTranspose()(graph)
        self.assertEqual(5, len(graph.nodes))
        self.assertEqual('-1,-1,768', graph.nodes[3].attr['dst_shape'])
        self.assertEqual('reshape_3d_before_transpose', graph.nodes[3].name)
        self.assertEqual('1,0', graph.nodes[3].attr['dims'])


if __name__ == "__main__":
    unittest.main()
