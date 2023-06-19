#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Intel Corporation
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
import os
import numpy as np
from intel_extension_for_transformers.backends.neural_engine.compile.graph import Graph
from intel_extension_for_transformers.backends.neural_engine.compile.sub_graph.pattern import PATTERNS
import intel_extension_for_transformers.backends.neural_engine.compile.graph_utils as util
util.autocast_init()
util.set_autocast('cast_type','bf16')
util.quant_info_init()

file_name = os.path.splitext(os.path.basename(__file__))[0]


class TestTorchOP(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        pass

    @classmethod
    def tearDownClass(self):
        os.remove('conf.yaml')
        pass

    def test_1(self):
        text = '''
model:
  name: model
  operator:
    input_data:
      type: Input
      output:
        input_ids.1:
          dtype: s32
          shape: [-1, -1]
        pastkv.1:
          dtype: fp32
          shape: [-1, -1]
        mask.1:
          dtype: s32
          shape: [-1, -1]
        self.model.gpt_neox.embed_in.weight:
          dtype: fp32
          shape: [50280, 2560]
          location: [0, 514867200]
        self.model.gpt_neox.layers.0.input_layernorm.weight:
          dtype: fp32
          shape: [2560]
          location: [514867200, 10240]
        self.model.gpt_neox.layers.0.input_layernorm.bias:
          dtype: fp32
          shape: [2560]
          location: [514877440, 10240]
        self.model.gpt_neox.layers.0.attention.query_key_value.weight:
          dtype: fp32
          shape: [7680, 2560]
          location: [514887680, 78643200]
        self.model.gpt_neox.layers.0.attention.query_key_value.bias:
          dtype: fp32
          shape: [7680]
          location: [593530880, 30720]
        '13':
          dtype: fp32
          shape: [1, 1, 2048, 20]
          location: [593561600, 163840]
        '12':
          dtype: fp32
          shape: [1, 1, 2048, 20]
          location: [593725440, 163840]
        '19':
          dtype: s32
          shape: [1]
          location: [593889280, 4]
        aten::neg_133_mul_val:
          dtype: fp32
          shape: [1]
          location: [593889284, 4]
        aten::neg_135_mul_val:
          dtype: fp32
          shape: [1]
          location: [593889288, 4]
        '6':
          shape: [1, 1, 2048, 2048]
          location: [593889292, 4194304]
        '4':
          dtype: fp32
          shape: [1]
          location: [598083596, 4]
        self.model.gpt_neox.layers.0.attention.dense.weight:
          dtype: fp32
          shape: [2560, 2560]
          location: [598083600, 26214400]
        self.model.gpt_neox.layers.0.attention.dense.bias:
          dtype: fp32
          shape: [2560]
          location: [624298000, 10240]
        self.model.gpt_neox.layers.0.post_attention_layernorm.weight:
          dtype: fp32
          shape: [2560]
          location: [624308240, 10240]
        self.model.gpt_neox.layers.0.post_attention_layernorm.bias:
          dtype: fp32
          shape: [2560]
          location: [624318480, 10240]
        self.model.gpt_neox.layers.0.mlp.dense_h_to_4h.weight:
          dtype: fp32
          shape: [10240, 2560]
          location: [624328720, 104857600]
        self.model.gpt_neox.layers.0.mlp.dense_h_to_4h.bias:
          dtype: fp32
          shape: [10240]
          location: [729186320, 40960]
        self.model.gpt_neox.layers.0.mlp.dense_4h_to_h.weight:
          dtype: fp32
          shape: [2560, 10240]
          location: [729227280, 104857600]
        self.model.gpt_neox.layers.0.mlp.dense_4h_to_h.bias:
          dtype: fp32
          shape: [2560]
          location: [834084880, 10240]
        self.model.gpt_neox.final_layer_norm.weight:
          dtype: fp32
          shape: [2560]
          location: [834095120, 10240]
        self.model.gpt_neox.final_layer_norm.bias:
          dtype: fp32
          shape: [2560]
          location: [834105360, 10240]
        self.model.embed_out.weight:
          dtype: fp32
          shape: [50280, 2560]
          location: [834115600, 514867200]
    prim::padding_sequence_3:
      type: PaddingSequence
      input:
        mask.1: {}
      output:
        '197': {}
      attr:
        dst_shape: -1,1,1,-1
        dims: 1
        padding_value: -3.4028234663852886e+38
    prim::TupleUnpack_2:
      type: TupleUnpack
      input:
        pastkv.1: {}
      output:
        '51': {}
    prim::TupleUnpack_119:
      type: TupleUnpack
      input:
        '51': {}
      output:
        past_key.1: {}
        past_value.1: {}
    prim::gather_indices_1:
      type: prim::gather_indices
      input:
        input_ids.1: {}
      output:
        '202': {}
    aten::embedding_0:
      type: Gather
      input:
        self.model.gpt_neox.embed_in.weight: {}
        input_ids.1: {}
      output:
        '74': {}
      attr:
        batch_dims: 0
        axis: 1
        embedding: true
    aten::layer_norm_17:
      type: LayerNorm
      input:
        '74': {}
        self.model.gpt_neox.layers.0.input_layernorm.weight: {}
        self.model.gpt_neox.layers.0.input_layernorm.bias: {}
      output:
        input.4: {}
      attr:
        epsilon: 1.0e-05
        normalized_shape: ''
    aten::linear_16:
      type: InnerProduct
      input:
        input.4: {}
        self.model.gpt_neox.layers.0.attention.query_key_value.weight: {}
        self.model.gpt_neox.layers.0.attention.query_key_value.bias: {}
      output:
        qkv.1: {}
    aten::size_94:
      type: Shape
      input:
        qkv.1: {}
      output:
        '77': {}
      attr:
        start: 0
        end: 1
    aten::size_80:
      type: Shape
      input:
        qkv.1: {}
      output:
        '78': {}
      attr:
        start: 1
        end: 2
    prim::ListConstruct_45:
      type: ListConstruct
      input:
        '77': {}
        '78': {}
      output:
        '79': {}
    aten::view_123:
      type: View
      input:
        qkv.1: {}
        '79': {}
      output:
        qkv.2: {}
      attr:
        shape: -1,-1,32,240
    aten::slice_43:
      type: Slice
      input:
        qkv.2: {}
      output:
        '81': {}
      attr:
        axes: 3
        starts: 0
        ends: 80
        steps: 1
    aten::permute_23:
      type: Reorder
      input:
        '81': {}
      output:
        query.1: {}
      attr:
        src_perm: 0,1,2,3
        dst_perm: 0,2,1,3
    aten::slice_40:
      type: Slice
      input:
        qkv.2: {}
      output:
        '83': {}
      attr:
        axes: 3
        starts: 80
        ends: 160
        steps: 1
    aten::permute_24:
      type: Reorder
      input:
        '83': {}
      output:
        key.1: {}
      attr:
        src_perm: 0,1,2,3
        dst_perm: 0,2,1,3
    aten::slice_41:
      type: Slice
      input:
        qkv.2: {}
      output:
        '85': {}
      attr:
        axes: 3
        starts: 160
        ends: 9223372036854775807
        steps: 1
    aten::permute_25:
      type: Reorder
      input:
        '85': {}
      output:
        value.1: {}
      attr:
        src_perm: 0,1,2,3
        dst_perm: 0,2,1,3
    aten::slice_35:
      type: Slice
      input:
        query.1: {}
      output:
        q.1: {}
      attr:
        axes: 3
        starts: 0
        ends: 20
        steps: 1
    aten::slice_36:
      type: Slice
      input:
        query.1: {}
      output:
        query_pass.1: {}
      attr:
        axes: 3
        starts: 20
        ends: 9223372036854775807
        steps: 1
    aten::slice_37:
      type: Slice
      input:
        key.1: {}
      output:
        k.1: {}
      attr:
        axes: 3
        starts: 0
        ends: 20
        steps: 1
    aten::slice_38:
      type: Slice
      input:
        key.1: {}
      output:
        key_pass.1: {}
      attr:
        axes: 3
        starts: 20
        ends: 9223372036854775807
        steps: 1
    aten::size_63:
      type: Shape
      input:
        key.1: {}
      output:
        '91': {}
      attr:
        start: 2
        end: 3
    aten::size_64:
      type: Shape
      input:
        past_key.1: {}
      output:
        '93': {}
      attr:
        start: 2
        end: 3
    aten::add_92:
      type: Add
      input:
        '91': {}
        '93': {}
      output:
        seq_len0.1: {}
    aten::slice_7:
      type: Slice
      input:
        '13': {}
        seq_len0.1: {}
      output:
        '98': {}
      attr:
        axes: 0
        starts: 0
        ends: null
        steps: 1
    aten::slice_6:
      type: Slice
      input:
        '12': {}
        seq_len0.1: {}
      output:
        '100': {}
      attr:
        axes: 0
        starts: 0
        ends: null
        steps: 1
    aten::size_81:
      type: Shape
      input:
        '98': {}
      output:
        '106': {}
      attr:
        start: 1
        end: 2
    aten::size_54:
      type: Shape
      input:
        '98': {}
      output:
        '107': {}
      attr:
        start: 3
        end: 4
    prim::ListConstruct_82:
      type: ListConstruct
      input:
        '106': {}
        '107': {}
      output:
        '108': {}
    aten::repeat_122:
      type: Repeat
      input:
        '202': {}
        '108': {}
      output:
        gather_indices0.1: {}
    aten::size_95:
      type: Shape
      input:
        gather_indices0.1: {}
      output:
        '110': {}
      attr:
        start: 0
        end: 1
    prim::ListConstruct_83:
      type: ListConstruct
      input:
        '110': {}
      output:
        '111': {}
    aten::repeat_128:
      type: Repeat
      input:
        '98': {}
        '111': {}
      output:
        '112': {}
    aten::gather_65:
      type: Gather
      input:
        '112': {}
        gather_indices0.1: {}
      output:
        cos.2: {}
    aten::repeat_129:
      type: Repeat
      input:
        '100': {}
        '111': {}
      output:
        '114': {}
    aten::gather_66:
      type: Gather
      input:
        '114': {}
        gather_indices0.1: {}
      output:
        sin.2: {}
    aten::mul_124:
      type: Mul
      input:
        q.1: {}
        cos.2: {}
      output:
        '116': {}
      attr:
        algorithm: mul
    aten::size_55:
      type: Shape
      input:
        q.1: {}
      output:
        '117': {}
      attr:
        start: 3
        end: 4
    aten::floor_divide_8:
      type: Div
      input:
        '117': {}
        '19': {}
      output:
        '119': {}
      attr:
        algorithm: div
    aten::slice_56:
      type: Slice
      input:
        q.1: {}
        '119': {}
      output:
        x1.1: {}
      attr:
        axes: 3
        starts: 0
        ends: null
        steps: 1
    aten::slice_57:
      type: Slice
      input:
        q.1: {}
        '119': {}
      output:
        x2.1: {}
      attr:
        axes: 3
        starts: null
        ends: 9223372036854775807
        steps: 1
    aten::neg_133:
      type: Neg
      input:
        x2.1: {}
        aten::neg_133_mul_val: {}
      output:
        '123': {}
      attr:
        algorithm: mul
    prim::ListConstruct_132:
      type: ListConstruct
      input:
        '123': {}
        x1.1: {}
      output:
        '124': {}
    aten::cat_70:
      type: Concat
      input:
        '124': {}
      output:
        '125': {}
      attr:
        axis: -1
    aten::mul_130:
      type: Mul
      input:
        '125': {}
        sin.2: {}
      output:
        '126': {}
      attr:
        algorithm: mul
    aten::add_84:
      type: Add
      input:
        '116': {}
        '126': {}
      output:
        query0.1: {}
    aten::mul_126:
      type: Mul
      input:
        k.1: {}
        cos.2: {}
      output:
        '128': {}
      attr:
        algorithm: mul
    aten::size_58:
      type: Shape
      input:
        k.1: {}
      output:
        '129': {}
      attr:
        start: 3
        end: 4
    aten::floor_divide_9:
      type: Div
      input:
        '129': {}
        '19': {}
      output:
        '131': {}
      attr:
        algorithm: div
    aten::slice_59:
      type: Slice
      input:
        k.1: {}
        '131': {}
      output:
        x10.1: {}
      attr:
        axes: 3
        starts: 0
        ends: null
        steps: 1
    aten::slice_60:
      type: Slice
      input:
        k.1: {}
        '131': {}
      output:
        x20.1: {}
      attr:
        axes: 3
        starts: null
        ends: 9223372036854775807
        steps: 1
    aten::neg_135:
      type: Neg
      input:
        x20.1: {}
        aten::neg_135_mul_val: {}
      output:
        '135': {}
      attr:
        algorithm: mul
    prim::ListConstruct_134:
      type: ListConstruct
      input:
        '135': {}
        x10.1: {}
      output:
        '136': {}
    aten::cat_71:
      type: Concat
      input:
        '136': {}
      output:
        '137': {}
      attr:
        axis: -1
    aten::mul_131:
      type: Mul
      input:
        '137': {}
        sin.2: {}
      output:
        '138': {}
      attr:
        algorithm: mul
    aten::add_85:
      type: Add
      input:
        '128': {}
        '138': {}
      output:
        key0.1: {}
    prim::ListConstruct_125:
      type: ListConstruct
      input:
        query0.1: {}
        query_pass.1: {}
      output:
        '140': {}
    aten::cat_72:
      type: Concat
      input:
        '140': {}
      output:
        query1.1: {}
      attr:
        axis: -1
    prim::ListConstruct_127:
      type: ListConstruct
      input:
        key0.1: {}
        key_pass.1: {}
      output:
        '142': {}
    aten::cat_73:
      type: Concat
      input:
        '142': {}
      output:
        key1.1: {}
      attr:
        axis: -1
    prim::ListConstruct_120:
      type: ListConstruct
      input:
        past_key.1: {}
        key1.1: {}
      output:
        '144': {}
    aten::cat_76:
      type: Concat
      input:
        '144': {}
      output:
        key2.1: {}
      attr:
        axis: -2
    prim::ListConstruct_121:
      type: ListConstruct
      input:
        past_value.1: {}
        value.1: {}
      output:
        '146': {}
    aten::cat_77:
      type: Concat
      input:
        '146': {}
      output:
        value0.1: {}
      attr:
        axis: -2
    aten::size_96:
      type: Shape
      input:
        query1.1: {}
      output:
        '148': {}
      attr:
        start: 0
        end: 1
    aten::size_86:
      type: Shape
      input:
        query1.1: {}
      output:
        '150': {}
      attr:
        start: 1
        end: 2
    aten::size_67:
      type: Shape
      input:
        query1.1: {}
      output:
        '152': {}
      attr:
        start: 2
        end: 3
    aten::size_78:
      type: Shape
      input:
        key2.1: {}
      output:
        '155': {}
      attr:
        start: -2
        end: -1
    aten::sub_87:
      type: Sub
      input:
        '155': {}
        '152': {}
      output:
        '157': {}
      attr:
        algorithm: sub
    aten::slice_5:
      type: Slice
      input:
        '6': {}
        '157': {}
        '155': {}
      output:
        '159': {}
      attr:
        axes: 2
        starts: null
        ends: null
        steps: 1
    aten::slice_61:
      type: Slice
      input:
        '159': {}
        '155': {}
      output:
        causal_mask.1: {}
      attr:
        axes: 3
        starts: 0
        ends: null
        steps: 1
    aten::mul_139:
      type: Mul
      input:
        '148': {}
        '150': {}
      output:
        '161': {}
      attr:
        algorithm: mul
    prim::ListConstruct_140:
      type: ListConstruct
      input:
        '161': {}
        '152': {}
        '155': {}
      output:
        '167': {}
    aten::zeros_52:
      type: Zeros
      input:
        '167': {}
      output:
        attn_scores.1: {}
    aten::transpose_137:
      type: Reorder
      input:
        key2.1: {}
      output:
        '200': {}
      attr:
        transpose_dims: 2,3
    prim::mybaddbmm_20:
      type: Baddbmm
      input:
        attn_scores.1: {}
        query1.1: {}
        '200': {}
      output:
        '201': {}
      attr:
        beta: 1.0
        alpha: 0.11180339753627777
    aten::where_4:
      type: Where
      input:
        causal_mask.1: {}
        '201': {}
        '4': {}
      output:
        attn_scores2.1: {}
      attr:
        mask_value: -3.4028234663852886e+38
    aten::add_88:
      type: Add
      input:
        attn_scores2.1: {}
        '197': {}
      output:
        input.6: {}
    aten::softmax_74:
      type: Softmax
      input:
        input.6: {}
      output:
        attn_weights.1: {}
      attr:
        axis: -1
    aten::matmul_138:
      type: Matmul
      input:
        attn_weights.1: {}
        value0.1: {}
      output:
        tensor.1: {}
    aten::permute_26:
      type: Reorder
      input:
        tensor.1: {}
      output:
        '178': {}
      attr:
        src_perm: 0,1,2,3
        dst_perm: 0,2,1,3
    aten::size_97:
      type: Shape
      input:
        '178': {}
      output:
        '180': {}
      attr:
        start: 0
        end: 1
    aten::size_89:
      type: Shape
      input:
        '178': {}
      output:
        '181': {}
      attr:
        start: 1
        end: 2
    prim::ListConstruct_29:
      type: ListConstruct
      input:
        '180': {}
        '181': {}
      output:
        '182': {}
    aten::view_143:
      type: View
      input:
        '178': {}
        '182': {}
      output:
        input0.1: {}
      attr:
        shape: -1,-1,2560
    aten::linear_15:
      type: InnerProduct
      input:
        input0.1: {}
        self.model.gpt_neox.layers.0.attention.dense.weight: {}
        self.model.gpt_neox.layers.0.attention.dense.bias: {}
      output:
        '184': {}
    aten::layer_norm_14:
      type: LayerNorm
      input:
        '74': {}
        self.model.gpt_neox.layers.0.post_attention_layernorm.weight: {}
        self.model.gpt_neox.layers.0.post_attention_layernorm.bias: {}
      output:
        input.8: {}
      attr:
        epsilon: 1.0e-05
        normalized_shape: ''
    aten::gelu_49:
      type: MatMulWithBiasGelu
      input:
        input.8: {}
        self.model.gpt_neox.layers.0.mlp.dense_h_to_4h.weight: {}
        self.model.gpt_neox.layers.0.mlp.dense_h_to_4h.bias: {}
      output:
        '187': {}
      attr:
        append_op: gelu_tanh
    aten::add_90:
      type: MatMulWithBiasAdd
      input:
        '187': {}
        self.model.gpt_neox.layers.0.mlp.dense_4h_to_h.weight: {}
        self.model.gpt_neox.layers.0.mlp.dense_4h_to_h.bias: {}
        '184': {}
      output:
        '189': {}
      attr:
        append_op: sum
    aten::add_91:
      type: Add
      input:
        '189': {}
        '74': {}
      output:
        input.2: {}
    aten::layer_norm_11:
      type: LayerNorm
      input:
        input.2: {}
        self.model.gpt_neox.final_layer_norm.weight: {}
        self.model.gpt_neox.final_layer_norm.bias: {}
      output:
        input.1: {}
      attr:
        epsilon: 1.0e-05
        normalized_shape: ''
    aten::linear_10:
      type: InnerProduct
      input:
        input.1: {}
        self.model.embed_out.weight: {}
      output:
        '192': {}
    prim::TupleConstruct_136:
      type: TupleConstruct
      input:
        key2.1: {}
        value0.1: {}
      output:
        '193': {}
    prim::TupleConstruct_145:
      type: TupleConstruct
      input:
        '193': {}
      output:
        '194': {}
    prim::TupleConstruct_144:
      type: TupleConstruct
      input:
        '192': {}
        '194': {}
      output:
        '195': {}
'''
        file = open('conf.yaml', 'w')
        file.write(text)
        file.close()
        dollygraph = Graph()
        dollygraph.graph_init('./conf.yaml')
        dollygraph.framework_modeling_config['framework'] = 'torch'
        for dest_op_name in dollygraph.nodes[0].output_tensors[7].dest_op:
            dest_node = dollygraph.get_node_by_name(dest_op_name)
            dest_node.input_tensors[1].data = np.zeros([7680,2560], dtype=np.float32)
            dest_node.input_tensors[2].data = np.zeros([7680,2560], dtype=np.float32)
        for dest_op_name in dollygraph.nodes[0].output_tensors[15].dest_op:
            dest_node = dollygraph.get_node_by_name(dest_op_name)
            dest_node.input_tensors[1].data = np.zeros([2560,2560], dtype=np.float32)
            dest_node.input_tensors[2].data = np.zeros([2560], dtype=np.float32)
        for dest_op_name in dollygraph.nodes[0].output_tensors[20].dest_op:
            dest_node = dollygraph.get_node_by_name(dest_op_name)
            dest_node.input_tensors[1].data = np.zeros([10240,2560], dtype=np.float32)
            dest_node.input_tensors[2].data = np.zeros([10240], dtype=np.float32)
        for dest_op_name in dollygraph.nodes[0].output_tensors[21].dest_op:
            dest_node = dollygraph.get_node_by_name(dest_op_name)
            dest_node.input_tensors[1].data = np.zeros([2560,10240], dtype=np.float32)
            dest_node.input_tensors[2].data = np.zeros([2560], dtype=np.float32)
        for dest_op_name in dollygraph.nodes[0].output_tensors[25].dest_op:
            dest_node = dollygraph.get_node_by_name(dest_op_name)
            dest_node.input_tensors[1].data = np.zeros([50280,2560], dtype=np.float32)
        for dest_op_name in dollygraph.nodes[0].output_tensors[8].dest_op:
            dest_node = dollygraph.get_node_by_name(dest_op_name)
            dest_node.input_tensors[0].data = np.zeros([1,1,2048,20], dtype=np.float32)
        for dest_op_name in dollygraph.nodes[0].output_tensors[9].dest_op:
            dest_node = dollygraph.get_node_by_name(dest_op_name)
            dest_node.input_tensors[0].data = np.zeros([1,1,2048,20], dtype=np.float32)
        for dest_op_name in dollygraph.nodes[0].output_tensors[13].dest_op:
            dest_node = dollygraph.get_node_by_name(dest_op_name)
            dest_node.input_tensors[0].data = np.zeros([1,1,2048,2048], dtype=np.float32)
        oldlen = len(dollygraph.nodes)
        p_fusion = PATTERNS['TorchUnpackBaddbmm']()
        dollygraph = p_fusion(dollygraph)
        newlen = len(dollygraph.nodes)
        self.assertTrue(oldlen != newlen)
        oldlen = len(dollygraph.nodes)
        p_fusion = PATTERNS['RemoveZeros']()
        dollygraph = p_fusion(dollygraph)
        newlen = len(dollygraph.nodes)
        self.assertTrue(oldlen != newlen)
        oldlen = len(dollygraph.nodes)
        p_fusion = PATTERNS['LowerAllTuples']()
        dollygraph = p_fusion(dollygraph)
        newlen = len(dollygraph.nodes)
        self.assertTrue(oldlen != newlen)
        p_fusion = PATTERNS['TorchEmbedding']()
        dollygraph = p_fusion(dollygraph)
        p_fusion = PATTERNS['RmsNorm']()
        dollygraph = p_fusion(dollygraph)
        p_fusion = PATTERNS['InnerproductReshapeFusion']()
        dollygraph = p_fusion(dollygraph)
        p_fusion = PATTERNS['InnerproductWithBiasGelu']()
        dollygraph = p_fusion(dollygraph)
        p_fusion = PATTERNS['InnerproductWithSwish']()
        dollygraph = p_fusion(dollygraph)
        p_fusion = PATTERNS['SliceMask']()
        dollygraph = p_fusion(dollygraph)
        p_fusion = PATTERNS['ArangewithReciprocal']()
        dollygraph = p_fusion(dollygraph)
        p_fusion = PATTERNS['InnerproductwithSlice']()
        dollygraph = p_fusion(dollygraph)
        p_fusion = PATTERNS['RoraryPosEmb']()
        dollygraph = p_fusion(dollygraph)
        p_fusion = PATTERNS['EinsumwithArange']()
        dollygraph = p_fusion(dollygraph)
        p_fusion = PATTERNS['RemoveSlice']()
        dollygraph = p_fusion(dollygraph)
        p_fusion = PATTERNS['RemoveRange']()
        dollygraph = p_fusion(dollygraph)
        p_fusion = PATTERNS['MatMulWithTransposeScaleAdd']()
        dollygraph = p_fusion(dollygraph)
        oldlen = len(dollygraph.nodes)
        p_fusion = PATTERNS['NeoxReorderChange']()
        dollygraph = p_fusion(dollygraph)
        newlen = len(dollygraph.nodes)
        self.assertTrue(oldlen != newlen)
        oldlen = len(dollygraph.nodes)
        p_fusion = PATTERNS['NeoxRoraryPosEmb']()
        dollygraph = p_fusion(dollygraph)
        newlen = len(dollygraph.nodes)
        self.assertTrue(oldlen != newlen)
        p_fusion = PATTERNS['InsertQuantNode']()
        dollygraph = p_fusion(dollygraph)
        p_fusion = PATTERNS['InsertBF16Node']()
        dollygraph = p_fusion(dollygraph)
        p_fusion = PATTERNS['TorchInsertBF16Node']()
        dollygraph = p_fusion(dollygraph)
        p_fusion = PATTERNS['QunatizeFusion']()
        dollygraph = p_fusion(dollygraph)
        p_fusion = PATTERNS['QKVMerge']()
        dollygraph = p_fusion(dollygraph)
        p_fusion = PATTERNS['ReshapeFusion']()
        dollygraph = p_fusion(dollygraph)
        p_fusion = PATTERNS['OperatorAdaptor']()
        dollygraph = p_fusion(dollygraph)
        p_fusion = PATTERNS['EmbeddingsTo2DBeforeInnerProduct']()
        dollygraph = p_fusion(dollygraph)
        p_fusion = PATTERNS['QuantGatherToBF16']()
        dollygraph = p_fusion(dollygraph)
        p_fusion = PATTERNS['MultiHeadAttention']()
        dollygraph = p_fusion(dollygraph)

if __name__ == "__main__":
    unittest.main()