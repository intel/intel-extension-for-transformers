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
from intel_extension_for_transformers.backends.neural_engine.compile.sub_graph.layer_norm_with_reduce_mean import LayerNormWithReduceMean


class TestLayerNormWithReduceMean(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        pass

    @classmethod
    def tearDownClass(self):
        pass
    
    def test_layer_norm_with_reduce_mean_1(self):
        graph = Graph()
        graph.framework_modeling_config['framework'] = 'onnxruntime'
        input_data_node = OPERATORS['Input']()
        input_tensors = []
        output_tensors = [Tensor(), Tensor(), Tensor()]
        input_data_node.construct('input_data', 'Input', input_tensors=input_tensors, 
                                output_tensors=output_tensors)

        ln_node = OPERATORS['LayerNorm']()
        input_tensors = [Tensor(), Tensor(shape=[768]), Tensor(shape=[768])]
        output_tensors = [Tensor(name='layer_norm:0', source_op=['layer_norm'], 
                                    dest_op=['reduce_mean'])]
        ln_node.construct('layer_norm', 'LayerNorm', input_tensors=input_tensors, 
                                output_tensors=output_tensors, attr=OrderedDict({
                                    'epsilon': 0.009}))
        
        reduce_mean_node = OPERATORS['ReduceMean']()
        input_tensors = [Tensor(name='layer_norm:0', source_op=['layer_norm'], 
                                    dest_op=['reduce_mean'])]
        output_tensors = [Tensor(name='reduce_mean:0', source_op=['reduce_mean'],
                                dest_op=[])]
        reduce_mean_node.construct('reduce_mean', 'ReduceMean', input_tensors=input_tensors, 
                                output_tensors=output_tensors, attr=OrderedDict(
                                    {'axis': 1, 'keep_dims': False}))
        
        graph.insert_nodes(len(graph.nodes), [input_data_node, ln_node, reduce_mean_node])
        graph = LayerNormWithReduceMean()(graph)
        self.assertEqual(5, len(graph.nodes))
        self.assertEqual('-1,-1,768', graph.nodes[2].attr['dst_shape'])
        self.assertEqual('reducemean_after_reshape', graph.nodes[3].name)
        self.assertEqual(1, graph.nodes[3].attr['axis'])
        self.assertEqual('-1,768', graph.nodes[4].attr['dst_shape'])


if __name__ == "__main__":
    unittest.main()
