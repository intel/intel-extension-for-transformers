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
from collections import OrderedDict
from intel_extension_for_transformers.backends.neural_engine.compile.ops.op import OPERATORS, Operator
from intel_extension_for_transformers.backends.neural_engine.compile.ops.tensor import Tensor
from intel_extension_for_transformers.backends.neural_engine.compile.graph import Graph
from intel_extension_for_transformers.backends.neural_engine.compile.sub_graph.reshape_before_restore_hidden_states import ReshapeBeforeRestoreHiddenStates


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
        output_tensors = [Tensor(name='input0', shape=[768]), Tensor(name='input1',shape=[768]), Tensor(name='input2',shape=[768])]
        input_data_node.construct('input_data', 'Input', input_tensors=input_tensors, output_tensors=output_tensors)

        ln_node = OPERATORS['LayerNorm']()
        input_tensors = [Tensor(name='a'), Tensor(name='b',shape=[384]), Tensor(name='c',shape=[384])]
        output_tensors = [Tensor(name='layer_norm:0', source_op=['layer_norm'],
                                    dest_op=['scatterelements'])]
        ln_node.construct('layer_norm', 'LayerNorm', input_tensors=input_tensors,
                                output_tensors=output_tensors, attr=OrderedDict({
                                    'epsilon': 0.009}))

        se_node = OPERATORS['ScatterElements']()
        input_tensors = [Tensor(name='data', data=np.array(1),shape=[384]), Tensor(name='indices',data=np.array(1),shape=[384]), Tensor(name = 'layer_norm:0', source_op=['layer_norm'], dest_op=['scatterelements'])]
        output_tensors = [Tensor(name='scatterelements:0', source_op=['scatter_elements'],
                                    dest_op=[])]
        se_node.construct('scatterelements', 'ScatterElements', input_tensors=input_tensors,
                                output_tensors=output_tensors, attr=OrderedDict({
                                    'axis': '1'}))


        graph.insert_nodes(len(graph.nodes), [input_data_node, ln_node, se_node])
        graph = ReshapeBeforeRestoreHiddenStates()(graph)
        graph.save()


        self.assertEqual(4, len(graph.nodes))
        self.assertEqual('-1,-1,384', graph.nodes[2].attr['dst_shape'])
        self.assertEqual('reshape_to_3d_after_layer_norm_in_restoration', graph.nodes[2].name)
        self.assertEqual('1', graph.nodes[3].attr['axis'])


if __name__ == "__main__":
    unittest.main()
