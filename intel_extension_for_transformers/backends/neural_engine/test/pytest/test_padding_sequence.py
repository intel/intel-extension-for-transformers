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
from intel_extension_for_transformers.backends.neural_engine.compile.sub_graph.padding_sequence import PaddingSequence
import numpy as np

class TestPaddingSequence(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        pass

    @classmethod
    def tearDownClass(self):
        pass
    
    def test_padding_sequence_1(self):
        graph = Graph()
        graph.framework_modeling_config['framework'] = 'onnxruntime'
        input_data_node = OPERATORS['Input']()
        input_tensors = []
        output_tensors = [Tensor(), Tensor(), Tensor()]
        input_data_node.construct('input_data', 'Input', input_tensors=input_tensors, 
                                output_tensors=output_tensors)
        
        shape_node = OPERATORS['Shape']()
        input_tensors = [Tensor()]
        output_tensors = [Tensor(name='shape:0', source_op=['shape'],
                                dest_op=['strided_slice'])]
        shape_node.construct('shape', 'Shape', input_tensors=input_tensors, 
                                output_tensors=output_tensors)
        
        strided_slice_node = OPERATORS['StridedSlice']()
        input_tensors = [Tensor(name='shape:0', source_op=['shape'],
                                dest_op=['strided_slice'])]
        output_tensors = [Tensor(name='strided_slice:0', source_op=['strided_slice'],
                                dest_op=['pack_0', 'pack_1'])]
        strided_slice_node.construct('strided_slice', 'StridedSlice', input_tensors=input_tensors, 
                                output_tensors=output_tensors)
        
        pack_0_node = OPERATORS['Pack']()
        input_tensors = [Tensor(name='strided_slice:0', source_op=['strided_slice'],
                                dest_op=['pack_0'])]
        output_tensors = [Tensor(name='pack_0:0', source_op=['pack_0'],
                                dest_op=['fill'])]
        pack_0_node.construct('pack_0', 'Pack', input_tensors=input_tensors, 
                                output_tensors=output_tensors)
        
        fill_node = OPERATORS['Fill']()
        input_tensors = [Tensor(name='pack_0:0', source_op=['pack_0'],
                                dest_op=['fill'])]
        output_tensors = [Tensor(name='fill:0', source_op=['fill'],
                                dest_op=['mul_1'])]
        fill_node.construct('fill', 'Fill', input_tensors=input_tensors, 
                                output_tensors=output_tensors)
        
        pack_1_node = OPERATORS['Pack']()
        input_tensors = [Tensor(name='strided_slice:0', source_op=['strided_slice'],
                                dest_op=['pack_1'])]
        output_tensors = [Tensor(name='pack_1:0', source_op=['pack_1'],
                                dest_op=['reshape'])]
        pack_1_node.construct('pack_1', 'Pack', input_tensors=input_tensors, 
                                output_tensors=output_tensors)
        
        reshape_node = OPERATORS['Reshape']()
        input_tensors = [Tensor(name='pack_1:0', source_op=['pack_1'],
                                dest_op=['reshape'])]
        output_tensors = [Tensor(name='reshape:0', source_op=['reshape'],
                                dest_op=['cast'])]
        reshape_node.construct('reshape', 'Reshape', input_tensors=input_tensors, 
                                output_tensors=output_tensors)
        
        cast_node = OPERATORS['Cast']()
        input_tensors = [Tensor(name='reshape:0', source_op=['reshape'],
                                dest_op=['cast'])]
        output_tensors = [Tensor(name='cast:0', source_op=['cast'],
                                dest_op=['mul_1'])]
        cast_node.construct('cast', 'Cast', input_tensors=input_tensors, 
                                output_tensors=output_tensors)
        
        mul_1_node = OPERATORS['Mul']()
        input_tensors = [Tensor(name='fill:0', source_op=['fill'],
                                dest_op=['mul_1']), Tensor(name='cast:0', source_op=['cast'],
                                dest_op=['mul_1'])]
        output_tensors = [Tensor(name='mul_1:0', source_op=['mul_1'],
                                dest_op=['expand_dims'])]
        mul_1_node.construct('mul_1', 'Mul', input_tensors=input_tensors, 
                                output_tensors=output_tensors)
        
        expand_dims_node = OPERATORS['ExpandDims']()
        input_tensors = [Tensor(name='mul_1:0', source_op=['mul_1'],
                                dest_op=['expand_dims'])]
        output_tensors = [Tensor(name='expand_dims:0', source_op=['expand_dims'],
                                dest_op=['sub'])]
        expand_dims_node.construct('expand_dims', 'ExpandDims', input_tensors=input_tensors, 
                                output_tensors=output_tensors)

        sub_node = OPERATORS['Sub']()
        input_tensors = [Tensor(name='expand_dims:0', source_op=['expand_dims'],
                                dest_op=['sub'])]
        output_tensors = [Tensor(name='sub:0', source_op=['sub'],
                                dest_op=['mul_2'])]
        sub_node.construct('sub', 'Sub', input_tensors=input_tensors, 
                                output_tensors=output_tensors)

        mul_2_node = OPERATORS['Mul']()
        input_tensors = [Tensor(name='sub:0', source_op=['sub'],
                                dest_op=['mul_2'])]
        output_tensors = [Tensor(name='mul_2:0', source_op=['mul_2'],
                                dest_op=['add'])]
        mul_2_node.construct('mul_2', 'Mul', input_tensors=input_tensors, 
                                output_tensors=output_tensors)                        
        
        add_node = OPERATORS['Add']()
        input_tensors = [Tensor(name='mul_2:0', source_op=['mul_2'],
                                dest_op=['add'])]
        output_tensors = [Tensor(name='add:0', source_op=['add'], dest_op=[])]
        add_node.construct('add', 'Add', input_tensors=input_tensors, 
                                output_tensors=output_tensors)   
        
        graph.insert_nodes(len(graph.nodes), [input_data_node, shape_node, strided_slice_node,
                                                pack_0_node, fill_node, pack_1_node, reshape_node,
                                                cast_node, mul_1_node, expand_dims_node, sub_node,
                                                mul_2_node, add_node])
        graph = PaddingSequence()(graph)
        self.assertEqual(3, len(graph.nodes))
        self.assertEqual('-1,12,0,-1', graph.nodes[1].attr['dst_shape'])
        self.assertEqual('AddV2', graph.nodes[2].op_type)


    def test_padding_sequence_2(self):
        graph = Graph()
        graph.framework_modeling_config['framework'] = 'onnxruntime'
        input_data_node = OPERATORS['Input']()
        input_tensors = []
        output_tensors = [Tensor(), Tensor(), Tensor()]
        input_data_node.construct('input_data', 'Input', input_tensors=input_tensors, 
                                output_tensors=output_tensors)

        unsqueeze_1_node = OPERATORS['Unsqueeze']()
        input_tensors = [Tensor()]
        output_tensors = [Tensor(name='unsqueeze_1:0', source_op=['unsqueeze_1'],
                                dest_op=['unsqueeze_2'])]
        unsqueeze_1_node.construct('unsqueeze_1', 'Unsqueeze', input_tensors=input_tensors, 
                                output_tensors=output_tensors)
        
        unsqueeze_2_node = OPERATORS['Unsqueeze']()
        input_tensors = [Tensor(name='unsqueeze_1:0', source_op=['unsqueeze_1'],
                                dest_op=['unsqueeze_2'])]
        output_tensors = [Tensor(name='unsqueeze_2:0', source_op=['unsqueeze_2'],
                                dest_op=['cast'])]
        unsqueeze_2_node.construct('unsqueeze_2', 'Unsqueeze', input_tensors=input_tensors, 
                                output_tensors=output_tensors)
        
        cast_node = OPERATORS['Cast']()
        input_tensors = [Tensor(name='unsqueeze_2:0', source_op=['unsqueeze_2'],
                                dest_op=['cast'])]
        output_tensors = [Tensor(name='cast:0', source_op=['cast'],
                                dest_op=['sub'])]
        cast_node.construct('cast', 'Cast', input_tensors=input_tensors, 
                                output_tensors=output_tensors)
        
        sub_node = OPERATORS['Sub']()
        input_tensors = [Tensor(name='cast:0', source_op=['cast'],
                                dest_op=['sub'])]
        output_tensors = [Tensor(name='sub:0', source_op=['sub'],
                                dest_op=['mul'])]
        sub_node.construct('sub', 'Sub', input_tensors=input_tensors, 
                                output_tensors=output_tensors)
        
        mul_node = OPERATORS['Mul']()
        input_tensors = [Tensor(name='sub:0', source_op=['sub'],
                                dest_op=['mul'])]
        output_tensors = [Tensor(name='mul:0', source_op=['mul'],
                                dest_op=['add'])]
        mul_node.construct('mul', 'Mul', input_tensors=input_tensors, 
                                output_tensors=output_tensors)
        
        add_node = OPERATORS['Add']()
        input_tensors = [Tensor(name='mul:0', source_op=['mul'],
                                dest_op=['add']), Tensor(data=np.array(1))]
        output_tensors = [Tensor(name='add:0', source_op=['add'],
                                dest_op=[])]
        add_node.construct('add', 'Add', input_tensors=input_tensors, 
                                output_tensors=output_tensors)
        
        mat_node = OPERATORS['MatMul']()
        input_tensors = [Tensor(), 
                        Tensor(name='src:0', dest_op=['matmul'], shape=[768])]
        output_tensors = [Tensor(name='matmul:0', source_op=['matmul'], dest_op=['add_1'])]
        mat_node.construct('matmul', 'MatMul', input_tensors=input_tensors, 
                                output_tensors=output_tensors)
        
        add_1_node = OPERATORS['Add']()
        input_tensors = [Tensor(name='matmul:0', source_op=['matmul'], dest_op=['add_1']),
                        Tensor(data=np.array(1))]
        output_tensors = [Tensor(name='add_1:0', source_op=['add_1'], dest_op=['add_2'])]
        add_1_node.construct('add_1', 'Add', input_tensors=input_tensors, 
                                output_tensors=output_tensors)
        
        add_2_node = OPERATORS['Add']()
        input_tensors = [Tensor(name='add_1:0', source_op=['add_1'], dest_op=['add_2']),
                        Tensor(data=np.array(1))]
        output_tensors = [Tensor(name='add_2:0', source_op=['add_2'], dest_op=['layernorm'])]
        add_2_node.construct('add_2', 'Add', input_tensors=input_tensors, 
                                output_tensors=output_tensors)
        
        layernorm_node = OPERATORS['LayerNorm']()
        input_tensors = [Tensor(name='add_2:0', source_op=['add_2'], dest_op=['layernorm']),
                        Tensor(data=np.array(1), shape=[768, 768]), 
                        Tensor(data=np.array(1), shape=[768])]
        output_tensors = [Tensor(name='layernorm:0', source_op=['layernorm'])]
        layernorm_node.construct('layernorm', 'LayerNorm', input_tensors=input_tensors, 
                                output_tensors=output_tensors)

        graph.insert_nodes(len(graph.nodes), [input_data_node, unsqueeze_1_node, unsqueeze_2_node,
                                                cast_node, sub_node, mul_node, add_node, mat_node,
                                                add_1_node, add_2_node, layernorm_node])
        graph = PaddingSequence()(graph)
        self.assertEqual(7, len(graph.nodes))
        self.assertEqual('-1,12,0,-1', graph.nodes[1].attr['dst_shape'])
    

    def test_padding_sequence_3(self):
        graph = Graph()
        graph.framework_modeling_config['framework'] = 'onnxruntime'
        input_data_node = OPERATORS['Input']()
        input_tensors = []
        output_tensors = [Tensor(), Tensor(), Tensor()]
        input_data_node.construct('input_data', 'Input', input_tensors=input_tensors, 
                                output_tensors=output_tensors)
        
        equal_node = OPERATORS['Equal']()
        input_tensors = [Tensor()]
        output_tensors = [Tensor(name='equal:0', source_op=['equal'],
                                dest_op=['reshape'])]
        equal_node.construct('equal', 'Equal', input_tensors=input_tensors, 
                                output_tensors=output_tensors)

        shape_0_node = OPERATORS['Shape']()
        input_tensors = [Tensor()]
        output_tensors = [Tensor(name='shape_0:0', source_op=['shape_0'],
                                dest_op=['gather'])]
        shape_0_node.construct('shape_0', 'Shape', input_tensors=input_tensors, 
                                output_tensors=output_tensors)
        
        gather_node = OPERATORS['Gather']()
        input_tensors = [Tensor(name='shape_0:0', source_op=['shape_0'],
                                dest_op=['gather'])]
        output_tensors = [Tensor(name='gather:0', source_op=['gather'],
                                dest_op=['unsqueeze_2'])]
        gather_node.construct('gather', 'Gather', input_tensors=input_tensors, 
                                output_tensors=output_tensors)
        
        unsqueeze_1_node = OPERATORS['Unsqueeze']()
        input_tensors = [Tensor()]
        output_tensors = [Tensor(name='unsqueeze_1:0', source_op=['unsqueeze_1'],
                                dest_op=['concat'])]
        unsqueeze_1_node.construct('unsqueeze_1', 'Unsqueeze', input_tensors=input_tensors, 
                                output_tensors=output_tensors)
        
        unsqueeze_2_node = OPERATORS['Unsqueeze']()
        input_tensors = [Tensor(name='gather:0', source_op=['gather'],
                                dest_op=['unsqueeze_2'])]
        output_tensors = [Tensor(name='unsqueeze_2:0', source_op=['unsqueeze_2'],
                                dest_op=['concat'])]
        unsqueeze_2_node.construct('unsqueeze_2', 'Unsqueeze', input_tensors=input_tensors, 
                                output_tensors=output_tensors)

        concat_node = OPERATORS['Concat']()
        input_tensors = [Tensor(name='unsqueeze_1:0', source_op=['unsqueeze_1'],
                                dest_op=['concat']), Tensor(name='unsqueeze_2:0', 
                                source_op=['unsqueeze_2'], dest_op=['concat'])]
        output_tensors = [Tensor(name='concat:0', source_op=['concat'],
                                dest_op=['reshape'])]
        concat_node.construct('concat', 'Concat', input_tensors=input_tensors, 
                                output_tensors=output_tensors)
        
        reshape_node = OPERATORS['Reshape']()
        input_tensors = [Tensor(name='equal:0', source_op=['equal'],
                                dest_op=['reshape']), Tensor(name='concat:0', 
                                source_op=['concat'], dest_op=['reshape'])]
        output_tensors = [Tensor(name='reshape:0', source_op=['reshape'],
                                dest_op=['expand'])]
        reshape_node.construct('reshape', 'Reshape', input_tensors=input_tensors, 
                                output_tensors=output_tensors)
        
        shape_1_node = OPERATORS['Shape']()
        input_tensors = [Tensor()]
        output_tensors = [Tensor(name='shape_1:0', source_op=['shape_1'],
                                dest_op=['expand'])]
        shape_1_node.construct('shape_1', 'Shape', input_tensors=input_tensors, 
                                output_tensors=output_tensors)
        
        expand_node = OPERATORS['Expand']()
        input_tensors = [Tensor(name='reshape:0', source_op=['reshape'],
                                dest_op=['expand']), Tensor(name='shape_1:0', 
                                source_op=['shape_1'], dest_op=['expand'])]
        output_tensors = [Tensor(name='expand:0', source_op=['expand'],
                                dest_op=['cast'])]
        expand_node.construct('expand', 'Expand', input_tensors=input_tensors, 
                                output_tensors=output_tensors)
        
        cast_node = OPERATORS['Cast']()
        input_tensors = [Tensor(name='expand:0', source_op=['expand'],
                                dest_op=['cast'])]
        output_tensors = [Tensor(name='cast:0', source_op=['cast'],
                                dest_op=['where'])]
        cast_node.construct('cast', 'Cast', input_tensors=input_tensors, 
                                output_tensors=output_tensors)

        where_node = OPERATORS['Where']()
        input_tensors = [Tensor(name='cast:0', source_op=['cast'],
                                dest_op=['where'])]
        output_tensors = [Tensor(name='where:0', source_op=['where'],
                                dest_op=[])]
        where_node.construct('where', 'Where', input_tensors=input_tensors, 
                                output_tensors=output_tensors)
        
        graph.insert_nodes(len(graph.nodes), [input_data_node, shape_0_node, gather_node, 
                                                equal_node, unsqueeze_1_node, unsqueeze_2_node, 
                                                concat_node, reshape_node, shape_1_node, 
                                                expand_node, cast_node, where_node])
        graph = PaddingSequence()(graph)
        self.assertEqual(3, len(graph.nodes))
        self.assertEqual('-1,12,0,-1', graph.nodes[1].attr['dst_shape'])
        self.assertEqual('AddV2', graph.nodes[2].op_type)

    def test_padding_sequence_4(self):
        graph = Graph()
        graph.framework_modeling_config['framework'] = 'onnxruntime'
        input_data_node = OPERATORS['Input']()
        input_tensors = []
        output_tensors = [Tensor(name='src', source_op=['input_data'],
                                 dest_op=['reducemax', 'unsqueeze1']), Tensor(), Tensor()]
        input_data_node.construct('input_data', 'Input', input_tensors=input_tensors, 
                                output_tensors=output_tensors)

        reducemax_node = OPERATORS['OpAny']()
        input_tensors = [Tensor(name='src', source_op=['input_data'], dest_op=['reducemax'])]
        output_tensors = [Tensor(name='reducemax:0', source_op=['reducemax'], dest_op=['cast1'])]
        reducemax_node.construct('reducemax', 'ReduceMax', input_tensors=input_tensors, 
                                output_tensors=output_tensors)

        cast1_node = OPERATORS['Cast']()
        input_tensors = [Tensor(name='reducemax:0', source_op=['reducemax'], dest_op=['cast1'])]
        output_tensors = [Tensor(name='cast1:0', source_op=['cast1'], dest_op=['range'])]
        cast1_node.construct('cast1', 'Cast', input_tensors=input_tensors, 
                                output_tensors=output_tensors)

        range_node = OPERATORS['Range']()
        input_tensors = [Tensor(name='cast1:0', source_op=['cast1'], dest_op=['range'])]
        output_tensors = [Tensor(name='range:0', source_op=['range'], dest_op=['expand'])]
        range_node.construct('range', 'Range', input_tensors=input_tensors, 
                                output_tensors=output_tensors)

        conshape_node = OPERATORS['ConstantOfShape']()
        input_tensors = [Tensor(name='cons_src0:0')]
        output_tensors = [Tensor(name='conshape:0', source_op=['conshape'], dest_op=['expand'])]
        conshape_node.construct('conshape', 'ConstantOfShape', input_tensors=input_tensors, 
                                output_tensors=output_tensors)

        expand_node = OPERATORS['Expand']()
        input_tensors = [Tensor(name='range:0', source_op=['range'], dest_op=['expand']),
                         Tensor(name='conshape:0', source_op=['conshape'], dest_op=['expand'])]
        output_tensors = [Tensor(name='expand:0', source_op=['expand'], dest_op=['tile'])]
        expand_node.construct('expand', 'Expand', input_tensors=input_tensors, 
                                output_tensors=output_tensors)

        tile_node = OPERATORS['Tile']()
        input_tensors = [Tensor(name='expand:0', source_op=['expand'], dest_op=['tile'])]
        output_tensors = [Tensor(name='tile:0', source_op=['tile'], dest_op=['less'])]
        tile_node.construct('tile', 'Tile', input_tensors=input_tensors, 
                                output_tensors=output_tensors)

        unsqueeze1_node = OPERATORS['Unsqueeze']()
        input_tensors = [Tensor(name='src', source_op=['input_data'], dest_op=['unsqueeze1'])]
        output_tensors = [Tensor(name='unsqueeze1:0', source_op=['unsqueeze1'], dest_op=['less'])]
        unsqueeze1_node.construct('unsqueeze1', 'Unsqueeze', input_tensors=input_tensors, 
                                output_tensors=output_tensors)

        less_node = OPERATORS['Less']()
        input_tensors = [Tensor(name='tile:0', source_op=['tile'], dest_op=['less']),
                         Tensor(name='unsqueeze1:0', source_op=['unsqueeze1'], dest_op=['less'])]
        output_tensors = [Tensor(name='less:0', source_op=['less'], dest_op=['unsqueeze2'])]
        less_node.construct('less', 'Less', input_tensors=input_tensors, 
                                output_tensors=output_tensors)

        unsqueeze2_node = OPERATORS['Unsqueeze']()
        input_tensors = [Tensor(name='less:0', source_op=['less'], dest_op=['unsqueeze2'])]
        output_tensors = [Tensor(name='unsqueeze2:0', source_op=['unsqueeze2'], dest_op=['not'])]
        unsqueeze2_node.construct('unsqueeze2', 'Unsqueeze', input_tensors=input_tensors, 
                                output_tensors=output_tensors)

        not_node = OPERATORS['Not']()
        input_tensors = [Tensor(name='unsqueeze2:0', source_op=['unsqueeze2'], dest_op=['not'])]
        output_tensors = [Tensor(name='not:0', source_op=['not'], dest_op=['unsqueeze3'])]
        not_node.construct('not', 'Not', input_tensors=input_tensors, 
                            output_tensors=output_tensors)
        
        unsqueeze3_node = OPERATORS['Unsqueeze']()
        input_tensors = [Tensor(name='not:0', source_op=['not'], dest_op=['unsqueeze2'])]
        output_tensors = [Tensor(name='unsqueeze3:0', source_op=['unsqueeze3'], dest_op=['cast2'])]
        unsqueeze3_node.construct('unsqueeze3', 'Unsqueeze', input_tensors=input_tensors, 
                                output_tensors=output_tensors)

        cast2_node = OPERATORS['Cast']()
        input_tensors = [Tensor(name='unsqueeze3:0', source_op=['unsqueeze3'], dest_op=['cast2'])]
        output_tensors = [Tensor(name='cast2:0', source_op=['cast2'], dest_op=['where'])]
        cast2_node.construct('cast2', 'Cast', input_tensors=input_tensors, 
                                output_tensors=output_tensors)

        where_node = OPERATORS['Where']()
        input_tensors = [Tensor(name='cast2:0', source_op=['cast2'], dest_op=['where']),
                         Tensor(name='where_src1'), Tensor(name='where_src2')]
        output_tensors = [Tensor(name='where:0', source_op=['where'], dest_op=[])]
        where_node.construct('where', 'Where', input_tensors=input_tensors, 
                                output_tensors=output_tensors)

        graph.insert_nodes(len(graph.nodes), [input_data_node, reducemax_node, cast1_node,
                                              range_node, conshape_node, expand_node, tile_node,
                                              unsqueeze1_node, less_node, unsqueeze2_node,
                                              not_node, unsqueeze3_node, cast2_node, where_node])
        graph = PaddingSequence()(graph)
        self.assertEqual(3, len(graph.nodes))
        self.assertEqual(True, graph.nodes[1].attr['seq_len_first'])
        self.assertEqual('AddV2', graph.nodes[2].op_type)


if __name__ == "__main__":
    unittest.main()
