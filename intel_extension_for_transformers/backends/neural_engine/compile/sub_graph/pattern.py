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

"""The supported pattern file."""

from abc import abstractmethod

# for complicated pattern, use a several lists and tuple to represent the main computation flow
# and sub-graphs
# for example, layer_norm pattern in bert_large:
# pattern: [ [(0, 'Mean'), (1, 'SquaredDifference'), (2, 'Mean'), (3, 'AddV2'), (4, 'Rsqrt'),
#             (5, 'Mul'),
#             (7 ,'Mul'), (8, 'Sub'), (9, 'AddV2')], [(5, 'Mul'), (6, 'Mul'), (9, 'AddV2')] ]
# the pattern has order, but need connections check cause of the staggered edges
# for detailed info, please see search_pattern func in compile/graph_utils.py
# supported patterns should have order
supported_patterns = [
    'InputFile',
    'InputData',
    'RemoveConstantOP',
    'CollectQuantInfo',
    'TorchInnerProductInsertBias',
    'LayerNorm',
    'WordEmbeddings',
    'MergedEmbeddingbag',
    'EmbeddingBag',
    'TokenTypeEmbeddings',
    'TokenTypeEmbeddingsV1',
    'PositionEmbeddings',
    'PositionEmbeddingsV1',
    'GenerateSequence',
    'PaddingSequence',
    'AttentionMaskLengthAdaptiveExpandIndices',
    'MatMulWithBias',
    'LastLayerShape',
    'InteractFeatures',
    'AttentionReshape',

    # Text Encoder
    'TextEncoder_WordEmbedding',
    'TextEncoder_QReshape',
    'TextEncoder_KVReshape',
    'TextEncoder_AttentionMaskAddReshape',
    'TextEncoder_SoftmaxReshape',
    'TextEncoder_MulReshape',
    'TextEncoder_AttentionReshape',
    'TextEncoder_CasualAttentionMask',

    #Vae decoder & Transformer2Dmodel
    'GroupNorm',
    'GroupNormSwish',
    'AttentionBlock_QKVPreReshape',
    'AttentionBlock_AttentionMaskAddReshape',
    'AttentionBlock_ConstantOfShapeWithMul',

    'Transformer2Dmodel_GetSampleBatch',
    'Transformer2Dmodel_SampleSlice',
    'Transformer2Dmodel_EncoderHiddenStatesReshape',
    'Transformer2Dmodel_ConstantOfShapeWithMul',
    'Transformer2Dmodel_QKVPreReshape',
    'Transformer2Dmodel_QKVReshape',
    'AttentionBlock_QKVReshape',
    'Transformer2Dmodel_QKVReshapeTo4D',
    'Transformer2Dmodel_AttentionMaskAddReshape',
    'Transformer2Dmodel_FFNInputSlice',
    'Transformer2Dmodel_FFNInputSlice_1',

    # General
    'QKVReshape',
    'DecoderAttnReshape',
    'ConvReshape',
    'AddClsToken',
    'TransposeBatchMatMul',
    'Gelu',

    'MatMulWithBiasGelu',
    'MatMulWithBiasAdd',
    'AddEmbeddings',
    'MatMulWithBiasTanh',
    'MatMulWithBiasRelu',
    'MatMulWithBiasSigmoid',
    'MatMulWithBiasUnsqueeze',

    "RestoreHiddenStatesInLengthAdaptiveUpdateIndices",
    "AttentionOutputLayerNormLengthAdaptiveExpandIndices",
    "ReshapeBeforeAndAfterAttentionOutLayerNormGatherElements",
    "ReshapeBeforeRestoreHiddenStates",
    "ReshapeAfterRestoreHiddenStates",
    'LayerNormWithReduceMean',
    'LayerNormWithTranspose',
    'StartEndLogits',
    'CastTo',

    'TorchUnpackBaddbmm',
    'RemoveZeros',
    'LowerAllTuples',
    'TorchEmbedding',
    'RmsNorm',
    'LlamaEmbeddings',
    'InnerproductReshapeFusion',
    'MatMulWithTranspose',
    'LlamaMatMulWithTranspose',
    'InnerproductWithBiasGelu',
    'InnerproductWithSwish',
    'SliceMask',
    'ArangewithReciprocal',
    'InnerproductwithSlice',
    'RoraryPosEmb',
    'EinsumwithArange',
    'RemoveSlice',
    'RemoveRange',
    'RemoveLastView',

    'MatMulWithTransposeScaleAdd',
    'NeoxReorderChange',
    'NeoxRoraryPosEmb',

    'InsertQuantNode',
    'InsertBF16Node',
    'TorchInsertBF16Node',
    'QunatizeFusion',
    'LlamaRoraryPosEmb',
    'QKVMerge',
    'ReshapeFusion',
    'StableDiffusion_bf16Convert',
    'StableDiffusion_ReshapeFusion',
    'StableDiffusion_MHAReshape',
    'StableDiffusion_MHA',

    'OperatorAdaptor',
    'EmbeddingsTo2DBeforeInnerProduct',
    'QuantGatherToBF16',
    'MultiHeadAttention',
    'OutputData',
    'QuantizedGraphDtypeRefactor',
    'ExplicitNHWCTransposeForConv',

    'Int8BF16MixedPrecisionChecker',
    'LlamaPostprocess',
    'RemoveUnusedOperator',
]

# for superbert, superbert patterns are huge patterns based on supported patterns
superbert_patterns = []

PATTERNS = {}


def pattern_registry(pattern_type):
    """The class decorator used to register all Algorithm subclasses.

    Args:
        cls (class): The class of register.
        pattern_type (str): The pattern registration name

    Returns:
        cls: The class of register.
    """
    def decorator_pattern(cls):
        """The pattern decorator."""
        if pattern_type in PATTERNS:
            raise ValueError('Cannot have two patterns with the same name')
        PATTERNS[pattern_type] = cls
        return cls

    return decorator_pattern


class Pattern(object):
    """The bass pattern class."""
    @abstractmethod
    def __call__(self, model, *args, **kwargs):
        """The __call__ function of the pattern class."""
        raise NotImplementedError
