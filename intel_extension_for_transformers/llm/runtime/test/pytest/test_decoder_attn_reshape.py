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
from intel_extension_for_transformers.backends.neural_engine.compile.sub_graph.decoder_attn_reshape import DecoderAttnReshape
import numpy as np


class TestDecoderAttnReshape(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        pass

    @classmethod
    def tearDownClass(self):
        pass

    def test_decoder_attn_reshape(self):
        graph = Graph()
        graph.framework_modeling_config['framework'] = 'onnxruntime'
        input_data_node = OPERATORS['Input']()
        input_tensors = []
        output_tensors = [Tensor(), Tensor(), Tensor()]
        input_data_node.construct('input_data', 'Input', input_tensors=input_tensors, 
                                output_tensors=output_tensors)

        unsqueeze1_node = OPERATORS['Unsqueeze']()
        input_tensors = [Tensor(name='unsqueeze1_src0')]
        output_tensors = [Tensor(name='unsqueeze1:0', source_op=['unsqueeze1'],
                                 dest_op=['concat'])]
        unsqueeze1_node.construct('unsqueeze1', 'Unsqueeze', input_tensors=input_tensors, 
                                output_tensors=output_tensors)

        shape1_node = OPERATORS['Shape']()
        input_tensors = [Tensor(name='s1_src0')]
        output_tensors = [Tensor(name='shape1:0', source_op=['shape1'], dest_op=['gather1'])]
        shape1_node.construct('shape1', 'Shape', input_tensors=input_tensors, 
                                output_tensors=output_tensors)

        gather1_node = OPERATORS['Gather']()
        input_tensors = [Tensor(name='shape1:0', source_op=['shape1'], dest_op=['gather1']),
                         Tensor(name='gather1_idx:0', data=np.array([0]).astype("int32"))]
        output_tensors = [Tensor(name='gather1:0', source_op=['gather1'], dest_op=['unsqueeze2'])]
        gather1_node.construct('gather1', 'Gather', input_tensors=input_tensors, 
                                output_tensors=output_tensors)

        unsqueeze2_node = OPERATORS['Unsqueeze']()
        input_tensors = [Tensor(name='gather1:0', source_op=['gather1'], dest_op=['unsqueeze2'])]
        output_tensors = [Tensor(name='unsqueeze2:0', source_op=['unsqueeze2'],
                                 dest_op=['concat'])]
        unsqueeze2_node.construct('unsqueeze2', 'Unsqueeze', input_tensors=input_tensors, 
                                output_tensors=output_tensors)

        shape2_node = OPERATORS['Shape']()
        input_tensors = [Tensor(name='s2_src0')]
        output_tensors = [Tensor(name='shape2:0', source_op=['shape2'], dest_op=['gather2'])]
        shape2_node.construct('shape2', 'Shape', input_tensors=input_tensors, 
                                output_tensors=output_tensors)

        gather2_node = OPERATORS['Gather']()
        input_tensors = [Tensor(name='shape2:0', source_op=['shape2'], dest_op=['gather2']),
                         Tensor(name='gather2_idx:0', data=np.array([0]).astype("int32"))]
        output_tensors = [Tensor(name='gather2:0', source_op=['gather2'], dest_op=['unsqueeze3'])]
        gather2_node.construct('gather2', 'Gather', input_tensors=input_tensors, 
                                output_tensors=output_tensors)

        unsqueeze3_node = OPERATORS['Unsqueeze']()
        input_tensors = [Tensor(name='gather2:0', source_op=['gather2'], dest_op=['unsqueeze3'])]
        output_tensors = [Tensor(name='unsqueeze3:0', source_op=['unsqueeze3'],
                                 dest_op=['concat'])]
        unsqueeze3_node.construct('unsqueeze3', 'Unsqueeze', input_tensors=input_tensors, 
                                output_tensors=output_tensors)

        concat_node = OPERATORS['Concat']()
        input_tensors = [Tensor(name='unsqueeze1:0', source_op=['unsqueeze1'], dest_op=['concat']),
                         Tensor(name='head_num', data=np.array([16]).astype("int32")),
                         Tensor(name='unsqueeze2:0', source_op=['unsqueeze2'], dest_op=['concat']),
                         Tensor(name='unsqueeze3:0', source_op=['unsqueeze3'], dest_op=['concat'])]
        output_tensors = [Tensor(name='concat:0', source_op=['concat'],
                                dest_op=['reshape'])]
        concat_node.construct('concat', 'Concat', input_tensors=input_tensors, 
                                output_tensors=output_tensors)

        reshape_node = OPERATORS['Reshape']()
        input_tensors = [Tensor(name='concat:0', source_op=['concat'], dest_op=['reshape'])]
        output_tensors = [Tensor(name='reshape:0', source_op=['reshape'], dest_op=['gather3'])]
        reshape_node.construct('reshape', 'Reshape', input_tensors=input_tensors, 
                                output_tensors=output_tensors)

        gather3_node = OPERATORS['Gather']()
        input_tensors = [Tensor(name='reshape:0', source_op=['reshape'], dest_op=['gather3']),
                         Tensor(name='gather3_idx:0', data=np.array([0]).astype("int32"))]
        output_tensors = [Tensor(name='gather3:0', source_op=['gather3'], dest_op=['transpose'])]
        gather3_node.construct('gather3', 'Gather', input_tensors=input_tensors, 
                                output_tensors=output_tensors, attr=OrderedDict({'axis': 1}))

        transpose_node = OPERATORS['Transpose']()
        input_tensors = [Tensor(name='gather3:0', source_op=['gather3'], dest_op=['transpose'])]
        output_tensors = [Tensor(name='transpose:0', source_op=['transpose'],
                                dest_op=[])]
        transpose_node.construct('transpose', 'Transpose', input_tensors=input_tensors, 
                          output_tensors=output_tensors)

        graph.insert_nodes(len(graph.nodes), [input_data_node, unsqueeze1_node, shape1_node,
                                              gather1_node, unsqueeze2_node, shape2_node,
                                              gather2_node, unsqueeze3_node, concat_node,
                                              reshape_node, gather3_node, transpose_node])
        graph = DecoderAttnReshape()(graph)
        self.assertEqual(4, len(graph.nodes))
        self.assertEqual('-1,16,-1,-1', graph.nodes[1].attr['dst_shape'])
        self.assertEqual('1,0', graph.nodes[1].attr['dims'])


if __name__ == "__main__":
    unittest.main()
