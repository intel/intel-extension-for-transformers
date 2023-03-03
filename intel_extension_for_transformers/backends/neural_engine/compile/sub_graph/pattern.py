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
    'CollectQuantInfo',
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
    'QKVReshape',
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
    "RestoreHiddenStatesInLengthAdaptiveUpdateIndices",
    "AttentionOutputLayerNormLengthAdaptiveExpandIndices",
    "ReshapeBeforeAndAfterAttentionOutLayerNormGatherElements",
    "ReshapeBeforeRestoreHiddenStates",
    "ReshapeAfterRestoreHiddenStates",
    'LayerNormWithReduceMean',
    'StartEndLogits',
    'InsertQuantNode',
    'InsertBF16Node',
    'QunatizeFusion',
    'QKVMerge',
    'ReshapeFusion',
    'OutputData',
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
