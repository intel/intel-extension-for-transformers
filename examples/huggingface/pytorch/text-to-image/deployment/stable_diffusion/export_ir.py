#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2023 Intel Corporation
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

import argparse
from intel_extension_for_transformers.backends.neural_engine.compile import compile, autocast

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
        'AttentionBlock_Resize2Gather': False,
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
        'Transformer2DModel_UpBlockResize': False,

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
        'AttentionBlock_Resize2Gather': True,
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
        'Transformer2DModel_UpBlockResize': True,

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

vae_decoder_pattern_config = {
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
        'AttentionBlock_Resize2Gather': True,
        'AttentionBlock_QKVPreReshape': True,
        'AttentionBlock_AttentionMaskAddReshape': True,
        'AttentionBlock_ConstantOfShapeWithMul': True,

        'Transformer2Dmodel_GetSampleBatch': True,
        'Transformer2Dmodel_SampleSlice': True,
        'Transformer2Dmodel_EncoderHiddenStatesReshape': True,
        'Transformer2Dmodel_ConstantOfShapeWithMul': True,
        'Transformer2Dmodel_QKVPreReshape': True,
        'Transformer2Dmodel_QKVReshape': True,
        'AttentionBlock_QKVReshape': True,
        'Transformer2Dmodel_QKVReshapeTo4D': False,
        'Transformer2Dmodel_AttentionMaskAddReshape': True,
        'Transformer2Dmodel_FFNInputSlice': True,
        'Transformer2Dmodel_FFNInputSlice_1': True,
        'Transformer2DModel_UpBlockResize': True,

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx_model", default="./model",
                        type=str, help="onnx model path.")
    parser.add_argument("--pattern_config", default="./",
                        type=str, help="pattern graph path.")
    parser.add_argument("--output_path", default="./ir",
                        type=str, help="pattern graph path.")
    parser.add_argument("--dtype", default="fp32", type=str)
    args = parser.parse_args()

    if args.pattern_config == 'text_encoder':
        args.pattern_config = text_encoder_pattern_config
    if args.pattern_config == 'unet':
        args.pattern_config = unet_pattern_config
    if args.pattern_config == 'vae_decoder':
        args.pattern_config = vae_decoder_pattern_config

    if args.dtype == "bf16":
        args.pattern_config['pattern_switch']['StableDiffusion_MHA'] = True
        with autocast(args.dtype):
            graph = compile(args.onnx_model, args.pattern_config)
            graph.save(args.output_path)
    elif args.dtype == "dynamic_int8":
        with autocast(args.dtype):
            graph = compile(args.onnx_model, args.pattern_config)
            graph.save(args.output_path)
    else:
        graph = compile(args.onnx_model, args.pattern_config)
        graph.save(args.output_path)
