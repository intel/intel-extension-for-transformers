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
from intel_extension_for_transformers.llm.runtime.deprecated.compile.ops.op import OPERATORS, Operator
from intel_extension_for_transformers.llm.runtime.deprecated.compile.ops.tensor import Tensor
from intel_extension_for_transformers.llm.runtime.deprecated.compile.graph import Graph
from intel_extension_for_transformers.llm.runtime.deprecated.compile.sub_graph.last_layer_shape import LastLayerShape


class TestLastLayerShape(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        pass

    @classmethod
    def tearDownClass(self):
        pass
    
    def test_last_layer_shape_1(self):
        graph = Graph()
        graph.framework_modeling_config['framework'] = 'onnxruntime'
        input_data_node = OPERATORS['Input']()
        input_tensors = []
        output_tensors = [Tensor(), Tensor(), Tensor()]
        input_data_node.construct('input_data', 'Input', input_tensors=input_tensors, 
                                output_tensors=output_tensors)

        pack_node = OPERATORS['Pack']()
        input_tensors = [Tensor(), Tensor(), Tensor(data=768)]
        output_tensors = [Tensor(name='pack:0', source_op=['pack'], dest_op=['reshape'])]
        pack_node.construct('pack', 'Pack', input_tensors=input_tensors, 
                                output_tensors=output_tensors)
        
        reshape_node = OPERATORS['Reshape']()
        input_tensors = [Tensor(name='pack:0', source_op=['pack'], dest_op=['reshape'])]
        output_tensors = [Tensor(name='reshape:0', source_op=['reshape'], 
                                    dest_op=['strided_slice'])]
        reshape_node.construct('reshape', 'Reshape', input_tensors=input_tensors, 
                                output_tensors=output_tensors)

        strided_slice_node = OPERATORS['StridedSlice']()
        input_tensors = [Tensor(name='reshape:0', source_op=['reshape'], 
                                    dest_op=['strided_slice'])]
        output_tensors = [Tensor(name='strided_slice:0', source_op=['strided_slice'], 
                                dest_op=['squeeze'])]
        strided_slice_node.construct('strided_slice', 'StridedSlice', input_tensors=input_tensors, 
                                output_tensors=output_tensors, attr=OrderedDict({'test': 1}))
        
        squeeze_node = OPERATORS['Squeeze']()
        input_tensors = [Tensor(name='strided_slice:0', source_op=['strided_slice'], 
                                dest_op=['squeeze'])]
        output_tensors = [Tensor(name='squeeze:0', dest_op=[])]
        squeeze_node.construct('squeeze', 'Squeeze', input_tensors=input_tensors, 
                                output_tensors=output_tensors)
        
        graph.insert_nodes(len(graph.nodes), [input_data_node, pack_node, reshape_node, 
                                                strided_slice_node, squeeze_node])
        
        graph = LastLayerShape()(graph)
        self.assertEqual(4, len(graph.nodes))
        self.assertEqual('-1,-1,768', graph.nodes[1].attr['dst_shape'])
        self.assertEqual(1, graph.nodes[2].attr['test'])
        self.assertEqual('-1,768', graph.nodes[3].attr['dst_shape'])


if __name__ == "__main__":
    unittest.main()
