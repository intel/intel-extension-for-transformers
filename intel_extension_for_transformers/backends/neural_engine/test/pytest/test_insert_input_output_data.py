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

import os
import unittest
from intel_extension_for_transformers.backends.neural_engine.compile.graph import Graph
from intel_extension_for_transformers.backends.neural_engine.compile.ops.op import OPERATORS
from intel_extension_for_transformers.backends.neural_engine.compile.ops.tensor import Tensor
from intel_extension_for_transformers.backends.neural_engine.compile.sub_graph.input_data import InputData
from intel_extension_for_transformers.backends.neural_engine.compile.sub_graph.output_data import OutputData

os.environ['GLOG_minloglevel'] = '2'
class TestInsertInputOuputData(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        pass

    @classmethod
    def tearDownClass(self):
        pass

    def test_input_output_data(self):
        # construct graph
        graph = Graph()
        graph.framework_modeling_config['framework'] = 'onnxruntime'
        # insert input node
        input_data_node = OPERATORS['ONNXINPUT']()
        input_tensors = []
        output_tensors = [Tensor()]
        input_data_node.construct('input_data', 'ONNXINPUT', input_tensors=input_tensors,
                                output_tensors=output_tensors)
        graph.insert_nodes(0, [input_data_node])
        # compile input and output
        graph = InputData()(graph)
        graph = OutputData()(graph)
        # assert
        self.assertEqual(2, len(graph.nodes))
        self.assertEqual("Input", graph.nodes[0].op_type)
        self.assertEqual("Output", graph.nodes[1].op_type)

if __name__ == "__main__":
    unittest.main()
