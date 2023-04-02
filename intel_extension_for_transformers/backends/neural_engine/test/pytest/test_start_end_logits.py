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
from intel_extension_for_transformers.backends.neural_engine.compile.sub_graph.start_end_logits import StartEndLogits


class TestStartEndLogits(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        pass

    @classmethod
    def tearDownClass(self):
        pass
    
    def test_start_end_logits(self):
        graph = Graph()
        graph.framework_modeling_config['framework'] = 'onnxruntime'
        input_data_node = OPERATORS['Input']()
        input_tensors = []
        output_tensors = [Tensor(), Tensor(), Tensor()]
        input_data_node.construct('input_data', 'Input', input_tensors=input_tensors, 
                                output_tensors=output_tensors)

        transpose_node = OPERATORS['Transpose']()
        input_tensors = [Tensor()]
        output_tensors = [Tensor(name='transpose:0', source_op=['transpose'],
                                dest_op=['unpack'])]
        transpose_node.construct('transpose', 'Transpose', input_tensors=input_tensors, 
                                output_tensors=output_tensors)
        
        unpack_node = OPERATORS['Unpack']()
        input_tensors = [Tensor(name='transpose:0', source_op=['transpose'],
                                dest_op=['unpack'])]
        output_tensors = [Tensor(name='unpack:0', source_op=['unpack'],
                                dest_op=['identity'])]
        unpack_node.construct('unpack', 'Unpack', input_tensors=input_tensors, 
                                output_tensors=output_tensors)

        identity_node = OPERATORS['Identity']()
        input_tensors = [Tensor(name='unpack:0', source_op=['unpack'],
                                dest_op=['identity'])]
        output_tensors = [Tensor(name='identity:0', source_op=['identity'], dest_op=[])]
        identity_node.construct('identity', 'Identity', input_tensors=input_tensors, 
                                output_tensors=output_tensors)
        
        graph.insert_nodes(len(graph.nodes), [input_data_node, transpose_node, unpack_node,
                                                    identity_node])
        graph = StartEndLogits()(graph)
        self.assertEqual(1, len(graph.nodes))


if __name__ == "__main__":
    unittest.main()
