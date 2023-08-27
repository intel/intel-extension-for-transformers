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
from intel_extension_for_transformers.backends.neural_engine.compile.graph import Graph
from intel_extension_for_transformers.backends.neural_engine.compile import compile, autocast
import numpy as np
import os
import sys
import torch
import copy


def is_win():
    return sys.platform.startswith('win')


unet_pattern_config = {
    'pattern_switch': {
        # General Pattern
        'PaddingSequence': False,
        'AttentionReshape': False,
        'QKVReshape': False,
        'ReshapeFusion': False,
        'InsertBF16Node': False,
        'OperatorAdaptor': False,

        # transpose_int8
        'QKVMerge': False,

        # 'TextEncoder
        'TextEncoder_WordEmbedding': False,
        'TextEncoder_QReshape': False,
        'TextEncoder_KVReshape': False,
        'TextEncoder_AttentionMaskAddReshape': False,
        'TextEncoder_SoftmaxReshape': False,
        'TextEncoder_MulReshape': False,
        'TextEncoder_AttentionReshape': False,
        'TextEncoder_CasualAttentionMask': False,

        # for unet and vae decoder
        'GroupNorm': True,

        # vae deocder & Transformer2Dmodel
        'AttentionBlock_QKVPreReshape': True,
        'AttentionBlock_AttentionMaskAddReshape': True,
        'AttentionBlock_ConstantOfShapeWithMul': True,
        'Transformer2Dmodel_GetSampleBatch': True,
        'Transformer2Dmodel_SampleSlice': True,
        'Transformer2Dmodel_EncoderHiddenStatesReshape': True,
        'Transformer2Dmodel_ConstantOfShapeWithMul': True,
        'Transformer2Dmodel_QKVPreReshape': True,
        'Transformer2Dmodel_QKVReshape': True,
        'AttentionBlock_QKVReshape': False,
        'Transformer2Dmodel_QKVReshapeTo4D': True,
        'Transformer2Dmodel_AttentionMaskAddReshape': True,
        'Transformer2Dmodel_FFNInputSlice': True,
        'Transformer2Dmodel_FFNInputSlice_1': True,

        # for all stable diffusion models
        'StableDiffusion_bf16Convert': True,
        'StableDiffusion_ReshapeFusion': True,

        # MHA
        'TorchInsertBF16Node': False,
        'StableDiffusion_MHAReshape': True,
        'StableDiffusion_MHA': True,
        'ExplicitNHWCTransposeForConv': True,

        # Channel_last
        'ConvReshape': False
    }
}


class TestUnetBF16(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        pass

    @classmethod
    def tearDownClass(self):
        pass

    def test_unet_bf16(self):
        os.environ['GLOG_minloglevel'] = '2'
        root_dir = '/tf_dataset2/models/nlp_toolkit/stable-diffusion/unet_bf16/'
        if is_win():
            root_dir = 'D:\\dataset\\nlptoolkit_ut_model\\'
        model_dir = root_dir + 'model.onnx'
        self.assertTrue(os.path.exists(model_dir), 'model is not found, please set your own model path!')

        with autocast('bf16'):
            graph = compile(model_dir, config=unet_pattern_config)
        input_0_path = root_dir + 'sample.pt'
        inputs_0 = torch.load(input_0_path)
        inputs_1 = torch.tensor([301], dtype=torch.float32)
        input_2_path = root_dir + 'encoder_hidden_states.pt'
        inputs_2 = torch.load(input_2_path)

        output = graph.inference([inputs_0, inputs_1, inputs_2])
        for node_name in output.keys():
            print(node_name, ', shape = ', output[node_name].shape)

        output_bf16 = copy.deepcopy(output['out_sample:0'])

        # fp32 unet
        root_dir = '/tf_dataset2/models/nlp_toolkit/stable-diffusion/unet_fp32/'
        unet_fp32_output_dir = root_dir + 'unet_fp32_output.pt'
        unet_fp32_output = torch.load(unet_fp32_output_dir)

        flag = np.allclose(unet_fp32_output, output_bf16, atol=1e-1)
        self.assertTrue(flag)


if __name__ == '__main__':
    unittest.main()
