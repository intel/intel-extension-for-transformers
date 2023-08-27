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
        past_key.1:
          dtype: bf16
          shape: [-1, -1, -1, -1]
        past_value.1:
          dtype: bf16
          shape: [-1, -1, -1, -1]
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
        '20':
          dtype: s32
          shape: [1]
          location: [593889280, 4]
        aten::neg_128_mul_val:
          dtype: fp32
          shape: [1]
          location: [593889284, 4]
        aten::neg_130_mul_val:
          dtype: fp32
          shape: [1]
          location: [593889288, 4]
        prim::mybaddbmm_19_mm_other:
          dtype: fp32
          shape: [1]
          location: [593889292, 4]
        '6':
          dtype: fp32
          shape: [1, 1, 2048, 2048]
          location: [593889296, 16777216]
        self.model.gpt_neox.layers.0.attention.dense.weight:
          dtype: fp32
          shape: [2560, 2560]
          location: [610666512, 26214400]
        self.model.gpt_neox.layers.0.attention.dense.bias:
          dtype: fp32
          shape: [2560]
          location: [636880912, 10240]
        self.model.gpt_neox.layers.0.post_attention_layernorm.weight:
          dtype: fp32
          shape: [2560]
          location: [636891152, 10240]
        self.model.gpt_neox.layers.0.post_attention_layernorm.bias:
          dtype: fp32
          shape: [2560]
          location: [636901392, 10240]
        self.model.gpt_neox.layers.0.mlp.dense_h_to_4h.weight:
          dtype: fp32
          shape: [10240, 2560]
          location: [636911632, 104857600]
        self.model.gpt_neox.layers.0.mlp.dense_h_to_4h.bias:
          dtype: fp32
          shape: [10240]
          location: [741769232, 40960]
        self.model.gpt_neox.layers.0.mlp.dense_4h_to_h.weight:
          dtype: fp32
          shape: [2560, 10240]
          location: [741810192, 104857600]
        self.model.gpt_neox.layers.0.mlp.dense_4h_to_h.bias:
          dtype: fp32
          shape: [2560]
          location: [846667792, 10240]
        self.model.gpt_neox.final_layer_norm.weight:
          dtype: fp32
          shape: [2560]
          location: [846678032, 10240]
        self.model.gpt_neox.final_layer_norm.bias:
          dtype: fp32
          shape: [2560]
          location: [846688272, 10240]
        self.model.embed_out.weight:
          dtype: fp32
          shape: [50280, 2560]
          location: [846698512, 514867200]
    prim::padding_sequence_2:
      type: PaddingSequence
      input:
        mask.1: {}
      output:
        '180': {}
      attr:
        dst_shape: -1,1,1,-1
        dims: 1
        padding_value: -3.4028234663852886e+38
    aten::embedding_0:
      type: Gather
      input:
        self.model.gpt_neox.embed_in.weight: {}
        input_ids.1: {}
      output:
        aten::embedding_0_out_ts: {}
      attr:
        batch_dims: 0
        axis: 1
        embedding: true
    aten::embedding_0_reshape:
      type: Reshape
      input:
        aten::embedding_0_out_ts: {}
      output:
        '63': {}
      attr:
        dst_shape: -1,-1,-1
        dims: 0,1
        mul: 0,1
    aten::layer_norm_16:
      type: LayerNorm
      input:
        '63': {}
        self.model.gpt_neox.layers.0.input_layernorm.weight: {}
        self.model.gpt_neox.layers.0.input_layernorm.bias: {}
      output:
        input.4: {}
      attr:
        epsilon: 1.0e-05
        normalized_shape: ''
    aten::linear_15:
      type: InnerProduct
      input:
        input.4: {}
        self.model.gpt_neox.layers.0.attention.query_key_value.weight: {}
        self.model.gpt_neox.layers.0.attention.query_key_value.bias: {}
        input_ids.1: {}
      output:
        qkv.2: {}
      attr:
        reshape: -1,-1,32,240
        reshape_dims: '0'
    aten::slice_46:
      type: Slice
      input:
        qkv.2: {}
      output:
        query.1: {}
      attr:
        axes: 3
        starts: 0
        ends: 80
        steps: 1
    aten::slice_43:
      type: Slice
      input:
        qkv.2: {}
      output:
        key.1: {}
      attr:
        axes: 3
        starts: 80
        ends: 160
        steps: 1
    aten::slice_44:
      type: Slice
      input:
        qkv.2: {}
      output:
        value.1: {}
      attr:
        axes: 3
        starts: 160
        ends: 9223372036854775807
        steps: 1
    aten::slice_38:
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
    aten::slice_39:
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
    aten::slice_40:
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
    aten::slice_41:
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
    aten::size_68:
      type: Shape
      input:
        key.1: {}
      output:
        '80': {}
      attr:
        start: 2
        end: 3
    aten::size_69:
      type: Shape
      input:
        past_key.1: {}
      output:
        '82': {}
      attr:
        start: 2
        end: 3
    aten::add_85:
      type: Add
      input:
        '80': {}
        '82': {}
      output:
        seq_len0.1: {}
    aten::slice_6:
      type: Slice
      input:
        '13': {}
        seq_len0.1: {}
      output:
        '87': {}
      attr:
        axes: 0
        starts: 0
        ends: null
        steps: 1
    aten::slice_5:
      type: Slice
      input:
        '12': {}
        seq_len0.1: {}
      output:
        '89': {}
      attr:
        axes: 0
        starts: 0
        ends: null
        steps: 1
    aten::size_70:
      type: Shape
      input:
        q.1: {}
      output:
        '91': {}
      attr:
        start: 2
        end: 3
    aten::add_76:
      type: Add
      input:
        '91': {}
        '82': {}
      output:
        '93': {}
    aten::slice_71:
      type: Slice
      input:
        '87': {}
        '82': {}
        '93': {}
      output:
        '95': {}
      attr:
        axes: 2
        starts: null
        ends: null
        steps: 1
    aten::slice_58:
      type: Slice
      input:
        '95': {}
      output:
        cos.2: {}
      attr:
        axes: 3
        starts: 0
        ends: 9223372036854775807
        steps: 1
    aten::slice_72:
      type: Slice
      input:
        '89': {}
        '82': {}
        '93': {}
      output:
        '97': {}
      attr:
        axes: 2
        starts: null
        ends: null
        steps: 1
    aten::slice_59:
      type: Slice
      input:
        '97': {}
      output:
        sin.2: {}
      attr:
        axes: 3
        starts: 0
        ends: 9223372036854775807
        steps: 1
    aten::mul_121:
      type: Mul
      input:
        q.1: {}
        cos.2: {}
      output:
        '99': {}
      attr:
        algorithm: mul
    aten::size_60:
      type: Shape
      input:
        q.1: {}
      output:
        '100': {}
      attr:
        start: 3
        end: 4
    aten::floor_divide_7:
      type: Div
      input:
        '100': {}
        '20': {}
      output:
        '102': {}
      attr:
        algorithm: div
    aten::slice_61:
      type: Slice
      input:
        q.1: {}
        '102': {}
      output:
        x1.1: {}
      attr:
        axes: 3
        starts: 0
        ends: null
        steps: 1
    aten::slice_62:
      type: Slice
      input:
        q.1: {}
        '102': {}
      output:
        x2.1: {}
      attr:
        axes: 3
        starts: null
        ends: 9223372036854775807
        steps: 1
    aten::neg_128:
      type: Neg
      input:
        x2.1: {}
        aten::neg_128_mul_val: {}
      output:
        '106': {}
      attr:
        algorithm: mul
    aten::cat_88:
      type: Concat
      input:
        '106': {}
        x1.1: {}
      output:
        '108': {}
      attr:
        axis: -1
    aten::mul_125:
      type: Mul
      input:
        '108': {}
        sin.2: {}
      output:
        '109': {}
      attr:
        algorithm: mul
    aten::add_77:
      type: Add
      input:
        '99': {}
        '109': {}
      output:
        query0.1: {}
    aten::mul_123:
      type: Mul
      input:
        k.1: {}
        cos.2: {}
      output:
        '111': {}
      attr:
        algorithm: mul
    aten::size_63:
      type: Shape
      input:
        k.1: {}
      output:
        '112': {}
      attr:
        start: 3
        end: 4
    aten::floor_divide_8:
      type: Div
      input:
        '112': {}
        '20': {}
      output:
        '114': {}
      attr:
        algorithm: div
    aten::slice_64:
      type: Slice
      input:
        k.1: {}
        '114': {}
      output:
        x10.1: {}
      attr:
        axes: 3
        starts: 0
        ends: null
        steps: 1
    aten::slice_65:
      type: Slice
      input:
        k.1: {}
        '114': {}
      output:
        x20.1: {}
      attr:
        axes: 3
        starts: null
        ends: 9223372036854775807
        steps: 1
    aten::neg_130:
      type: Neg
      input:
        x20.1: {}
        aten::neg_130_mul_val: {}
      output:
        '118': {}
      attr:
        algorithm: mul
    aten::cat_89:
      type: Concat
      input:
        '118': {}
        x10.1: {}
      output:
        '120': {}
      attr:
        axis: -1
    aten::mul_126:
      type: Mul
      input:
        '120': {}
        sin.2: {}
      output:
        '121': {}
      attr:
        algorithm: mul
    aten::add_78:
      type: Add
      input:
        '111': {}
        '121': {}
      output:
        key0.1: {}
    aten::cat_90:
      type: Concat
      input:
        query0.1: {}
        query_pass.1: {}
      output:
        query1.1: {}
      attr:
        axis: -1
    aten::cat_91:
      type: Concat
      input:
        key0.1: {}
        key_pass.1: {}
      output:
        key1.1: {}
      attr:
        axis: -1
    aten::cat_34:
      type: Concat
      input:
        past_key.1: {}
        key1.1: {}
      output:
        key2.1: {}
      attr:
        axis: 1
    aten::cat_35:
      type: Concat
      input:
        past_value.1: {}
        value.1: {}
      output:
        value0.1: {}
      attr:
        axis: 1
    prim::mybaddbmm_19:
      type: Matmul
      input:
        query1.1: {}
        key2.1: {}
      output:
        prim::mybaddbmm_19:0: {}
      attr:
        src0_perm: 0,2,1,3
        src1_perm: 0,2,3,1
    Baddbmm_div:
      type: BinaryOp
      input:
        prim::mybaddbmm_19:0: {}
        prim::mybaddbmm_19_mm_other: {}
      output:
        Baddbmm_div:0: {}
      attr:
        algorithm: div
    aten::slice_66:
      type: SliceMask
      input:
        '6': {}
        input_ids.1: {}
        past_value.1: {}
      output:
        aten::slice_66:0: {}
      attr:
        starts: 0
        ends_with_tensor: 1
        ends_with_tensor_alg: sub
        axes: 2, 3
        steps: 1
    aten::where_3:
      type: BinaryAdd
      input:
        Baddbmm_div:0: {}
        aten::slice_66:0: {}
      output:
        attn_scores2.1: {}
    aten::add_81:
      type: Add
      input:
        attn_scores2.1: {}
        '180': {}
      output:
        input.6: {}
    aten::softmax_92:
      type: Softmax
      input:
        input.6: {}
      output:
        attn_weights.1: {}
      attr:
        axis: -1
    aten::matmul_133:
      type: Matmul
      input:
        attn_weights.1: {}
        value0.1: {}
      output:
        input0.1: {}
      attr:
        src1_perm: 0,2,1,3
        dst_perm: 0,2,1,3
        reshape: -1,2560
    aten::linear_14:
      type: InnerProduct
      input:
        input0.1: {}
        self.model.gpt_neox.layers.0.attention.dense.weight: {}
        self.model.gpt_neox.layers.0.attention.dense.bias: {}
      output:
        '167': {}
    aten::layer_norm_13:
      type: LayerNorm
      input:
        '63': {}
        self.model.gpt_neox.layers.0.post_attention_layernorm.weight: {}
        self.model.gpt_neox.layers.0.post_attention_layernorm.bias: {}
      output:
        input.8: {}
      attr:
        epsilon: 1.0e-05
        normalized_shape: ''
    aten::gelu_54:
      type: MatMulWithBiasGelu
      input:
        input.8: {}
        self.model.gpt_neox.layers.0.mlp.dense_h_to_4h.weight: {}
        self.model.gpt_neox.layers.0.mlp.dense_h_to_4h.bias: {}
      output:
        '170': {}
      attr:
        append_op: gelu_tanh
    aten::add_83:
      type: MatMulWithBiasAdd
      input:
        '170': {}
        self.model.gpt_neox.layers.0.mlp.dense_4h_to_h.weight: {}
        self.model.gpt_neox.layers.0.mlp.dense_4h_to_h.bias: {}
        '167': {}
      output:
        '172': {}
      attr:
        append_op: sum
    aten::add_84:
      type: Add
      input:
        '172': {}
        '63': {}
      output:
        input.2: {}
    aten::layer_norm_10:
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
    aten::linear_9:
      type: InnerProduct
      input:
        input.1: {}
        self.model.embed_out.weight: {}
      output:
        '175': {}
'''
        file = open('conf.yaml', 'w')
        file.write(text)
        file.close()
        dollygraph = Graph()
        dollygraph.graph_init('./conf.yaml')
        dollygraph.framework_modeling_config['framework'] = 'torch'
        for dest_op_name in dollygraph.nodes[0].output_tensors[9].dest_op:
            dest_node = dollygraph.get_node_by_name(dest_op_name)
            dest_node.input_tensors[0].data = np.zeros([1,1,2048,20], dtype=np.float32)
        for dest_op_name in dollygraph.nodes[0].output_tensors[10].dest_op:
            dest_node = dollygraph.get_node_by_name(dest_op_name)
            dest_node.input_tensors[0].data = np.zeros([1,1,2048,20], dtype=np.float32)
        oldlen = len(dollygraph.nodes)
        p_fusion = PATTERNS['NeoxRoraryPosEmb']()
        dollygraph = p_fusion(dollygraph)
        newlen = len(dollygraph.nodes)
        self.assertTrue(oldlen != newlen)

if __name__ == "__main__":
    unittest.main()