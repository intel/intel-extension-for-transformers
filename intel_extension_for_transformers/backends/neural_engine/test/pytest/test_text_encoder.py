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
from intel_extension_for_transformers.backends.neural_engine.compile import compile
import numpy as np
import os
import sys
import torch
import onnxruntime as ort

text_encoder_pattern_config = {
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
        'TextEncoder_WordEmbedding': True,
        'TextEncoder_QReshape': True,
        'TextEncoder_KVReshape': True,
        'TextEncoder_AttentionMaskAddReshape': True,
        'TextEncoder_SoftmaxReshape': True,
        'TextEncoder_MulReshape': True,
        'TextEncoder_AttentionReshape': True,
        'TextEncoder_CasualAttentionMask': True,

        # for unet and vae decoder
        'GroupNorm': False,

        # vae deocder & Transformer2Dmodel
        'AttentionBlock_QKVPreReshape': False,
        'AttentionBlock_AttentionMaskAddReshape': False,
        'AttentionBlock_ConstantOfShapeWithMul': False,
        'Transformer2Dmodel_GetSampleBatch': False,
        'Transformer2Dmodel_SampleSlice': False,
        'Transformer2Dmodel_EncoderHiddenStatesReshape': False,
        'Transformer2Dmodel_ConstantOfShapeWithMul': False,
        'Transformer2Dmodel_QKVPreReshape': False,
        'Transformer2Dmodel_QKVReshape': False,
        'AttentionBlock_QKVReshape': False,
        'Transformer2Dmodel_QKVReshapeTo4D': False,
        'Transformer2Dmodel_AttentionMaskAddReshape': False,
        'Transformer2Dmodel_FFNInputSlice': False,
        'Transformer2Dmodel_FFNInputSlice_1': False,

        # for all stable diffusion models
        'StableDiffusion_bf16Convert': True,
        'StableDiffusion_ReshapeFusion': True,

        # MHA
        'TorchInsertBF16Node': False,
        'StableDiffusion_MHAReshape': True,
        'StableDiffusion_MHA': False,
        'ExplicitNHWCTransposeForConv': True,

        # Channel_last
        'ConvReshape': False
    }
}


def is_win():
    return sys.platform.startswith('win')


class TestTextEncoder(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        pass

    @classmethod
    def tearDownClass(self):
        pass

    def test_text_encoder(self):
        os.environ['GLOG_minloglevel'] = '2'
        root_dir = '/tf_dataset2/models/nlp_toolkit/stable-diffusion/text_encoder_fp32/'
        if is_win():
            root_dir = 'D:\\dataset\\nlptoolkit_ut_model\\'
        model_dir = root_dir + 'model.onnx'
        self.assertTrue(os.path.exists(model_dir), 'model is not found, please set your own model path!')

        graph = compile(model_dir, config=text_encoder_pattern_config)
        input_0_path = root_dir + 'input_ids.pt'
        inputs_0 = torch.load(input_0_path)

        output = graph.inference([inputs_0])
        for node_name in output.keys():
            print(node_name, ', shape = ', output[node_name].shape)

        # onnxruntime
        session = ort.InferenceSession(model_dir)
        x = torch.load(input_0_path).numpy().astype(np.int32)

        ortvalue = ort.OrtValue.ortvalue_from_numpy(x)
        ortvalue.device_name()

        outputs = session.run(None, {
            'input_ids': ortvalue,
        })

        for idx, output_ort in enumerate(outputs):
            print('onnxruntime output_shape = ', output_ort.shape)

        flag = np.allclose(output['last_hidden_state:0'], outputs[0], atol=1e-2)
        self.assertTrue(flag)


if __name__ == '__main__':
    unittest.main()
