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

"""The neural engine subgraph matcher file."""

import copy
from tqdm import tqdm
from collections import namedtuple, OrderedDict
from .pattern import supported_patterns, superbert_patterns, PATTERNS
from .. import logger

EXECUTOR_TYPE = {
    "InnerProduct": "InnerProduct",
    "MatMulWithBias": "InnerProduct",
    "MatMulWithBiasAdd": "InnerProduct",
    "MatMulWithBiasGelu": "InnerProduct",
    "MatMulWithBiasTanh": "InnerProduct",
    "MatMulWithBiasRelu": "InnerProduct",
    "MatMulWithBiasSigmoid": "InnerProduct",
    "MatMulWithBiasSwish": "InnerProduct",
    "Matmul": "Matmul",
    "Einsum": "Matmul",
    "MatMul": "InnerProduct",
    "Conv": "Convolution",
    "QuantizedMatMulWithBiasAndDequantize": "InnerProduct",
    "TransposeBatchMatMul": "Matmul",
    "MatmulwithTranspose" : "Matmul", 
    "BatchMatMul": "Matmul",
    "BatchMatMulV2": "Matmul",
    "Add": "BinaryAdd",
    "AddV2": "BinaryAdd",
    "AddWithAdd": "BinaryAdd",
    "QLinearAdd": "BinaryAdd",
    "Transpose": "Reorder",
    "GatherV2": "Gather",
    "ExpandDimsToReshape": "Reshape",
    "QuantizeV2": "Quantize",
    "QuantizeLinear": "Quantize",
    "OneHot": "Onehot",
    "LayerNormalization": "LayerNorm",
    "FusedGemm": "InnerProduct",
    "_QuantizedFusedMatMulAndDequantize": "InnerProduct",
    "_FusedMatMul": "InnerProduct",
    "_MklLayerNorm": "LayerNorm",
    "Div": "BinaryOp",
    "Sub": "BinaryOp",
    "Mul": "BinaryOp",
    "Equal": "BinaryOp",
    "Less": "BinaryOp",
    "Greater": "BinaryOp",
    "LessEqual": "BinaryOp",
    "LessOrEqual": "BinaryOp",
    "GreaterOrEqual": "BinaryOp",
    "NonZero": "BinaryOp",
    "NotEqual": "BinaryOp",
    'Not': "BinaryOp",
    'Neg': "BinaryOp",
    "Sin": "CosSin",
    "Cos": "CosSin",
    "Resize": "Resampling",
}

pattern_default_setting = {
    # General Pattern
    'PaddingSequence': True,
    'AttentionReshape': True,
    'QKVReshape': True,
    'ReshapeFusion': True,
    'InsertBF16Node': True,
    'OperatorAdaptor': True,
    'ConvReshape': True,

    'GroupNorm': True,

    # transpose_mode_int8
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
    'StableDiffusion_bf16Convert': False,
    'StableDiffusion_ReshapeFusion': False,

    # MHA for the stable diffusion
    'StableDiffusion_MHAReshape': False,
    'StableDiffusion_MHA': False,
    'ExplicitNHWCTransposeForConv': False,
    
    #GPT-J
    'TorchEmbedding': True,
    'InnerproductReshapeFusion': True,
    'MatMulWithTranspose': True,
    'InnerproductWithBiasGelu': True,
    'SliceMask': True,
    'ArangewithReciprocal': True,
    'InnerproductwithSlice': True,
    'RoraryPosEmb': True,
    'EinsumwithArange': True,
    'RemoveSlice': True,
    'RemoveRange': True,
    'RemoveLastView': True,
    
    'MatMulWithTransposeScaleAdd': True,
    'EmbeddingsTo2DBeforeInnerProduct': True,
    'QuantGatherToBF16': False,
    'TorchInsertBF16Node': True,
    'MultiHeadAttention': True,
    'Int8BF16MixedPrecisionChecker': False,
    'QuantizedGraphDtypeRefactor': True,
    
    #LLAMA
    'LlamaEmbeddings': False,
    'LlamaMatMulWithTranspose': False,
    'LlamaRoraryPosEmb': False,
    'LlamaPostprocess': False,
    'RemoveUnusedOperator': True,
    #GPT-NEOX
    'NeoxReorderChange': False,
    'NeoxRoraryPosEmb': False,
}

class SubGraphMatcher(object):
    """The SubGraphMatcher class."""
    def __call__(self, model, tune = False, pattern_config = None):
        """The __call__ function of SubGraphMatcher class."""
        logger.info('Start to implement Sub-Graph matching and replacing...') 
        if tune:
            model = self._tune_patterns(model)
        else:
            model = self._fuse_patterns(model, pattern_config=pattern_config)
        logger.info('Sub-Graph match and replace done...')
        return model

    def _fuse_patterns(self, model, supported_patterns=supported_patterns, pattern_mask=None, pattern_config=None):
        pattern_mask = [True for _ in range(len(supported_patterns))] \
                if pattern_mask == None else pattern_mask
        
        for index in range(len(supported_patterns)):
            pattern_name = supported_patterns[index]
            if pattern_name in pattern_default_setting:
                pattern_mask[index] = pattern_default_setting[pattern_name]

        # modify the pattern mask according to pattern_config
        if pattern_config != None:
            for index in range(len(supported_patterns)):
                pattern_name = supported_patterns[index]
                if pattern_name in pattern_config['pattern_switch']:
                    status = pattern_config['pattern_switch'][pattern_name]
                    pattern_mask[index] = status

        for pattern_id, pattern in enumerate(supported_patterns):
            if pattern in PATTERNS and pattern_mask[pattern_id]:
                p_fusion = PATTERNS[pattern]()
                model = p_fusion(model)
        model = self._remove_identity(model)
        return model

    def _tune_patterns(self, model, iterations = 10, warm_up = 5):
        # pattern tuning strategy(for superbert): 
        #    1. only one pattern off/on each time (pruning)
        #    2. check accuracy with framework
        #    3. and only save min latency config
        logger.info('Start tuning pattern...')
        all_patterns = supported_patterns + superbert_patterns
        pattern_mask = [True for i in range(len(all_patterns))]
        min_latency = float("inf")
        # skip tuning input node fusion and output node fusion
        for idx in tqdm(range(len(supported_patterns), len(all_patterns))):
            # pattern on
            on_latency = float("inf")
            try:
                on_model = copy.deepcopy(model)
                self._fuse_patterns(on_model, all_patterns, pattern_mask)
                on_result, on_latency = on_model._get_latency([], iterations, warm_up)
            except:
                logger.warning("Graph can not be inferenced, please check the graph!")
            # pattern off
            off_latency = float("inf")
            try:
                off_pattern_mask = copy.deepcopy(pattern_mask)
                off_pattern_mask[idx] = False
                off_model = copy.deepcopy(model)
                self._fuse_patterns(off_model, all_patterns, off_pattern_mask)
                off_result, off_latency = off_model._get_latency([], iterations, warm_up)
            except:
                logger.warning("Graph can not be inferenced, please check the graph!")
            # update min latency and pattern mask
            if off_latency < on_latency and off_latency < min_latency:
                min_latency = off_latency
                pattern_mask = off_pattern_mask
        
        # generate model according pattern mask 
        self._fuse_patterns(model, all_patterns, pattern_mask)
        logger.info('End tuning pattern...')
        return model

    def _remove_identity(self, model):
        rm_node_names = []
        rm_op_type = ['Identity']
        for i in range(len(model.nodes)):
            node = model.nodes[i]
            if node.op_type in rm_op_type:
                rm_node_names.append(node.name)
            else:
                if node.op_type in EXECUTOR_TYPE.keys():
                    if node.op_type == "Cos":
                        node.attr = OrderedDict({'algorithm': 'cos'})
                    if node.op_type == "Sin":
                        node.attr = OrderedDict({'algorithm': 'sin'})                           
                    op_type = EXECUTOR_TYPE[node.op_type]
                    model.nodes[i].op_type = op_type
        model.remove_nodes(rm_node_names)
        return model

