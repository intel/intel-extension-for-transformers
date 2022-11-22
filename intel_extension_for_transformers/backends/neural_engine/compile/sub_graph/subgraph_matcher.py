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

import time
import copy
import numpy as np
from tqdm import tqdm
from .pattern import supported_patterns, superbert_patterns, PATTERNS
from .. import logger

EXECUTOR_TYPE = {
    "MatMulWithBias": "InnerProduct",
    "MatMulWithBiasAdd": "InnerProduct",
    "MatMulWithBiasGelu": "InnerProduct",
    "MatMulWithBiasTanh": "InnerProduct",
    "MatMulWithBiasRelu": "InnerProduct",
    "MatMulWithBiasSigmoid": "InnerProduct",
    "MatMul": "InnerProduct",
    "QuantizedMatMulWithBiasAndDequantize": "InnerProduct",
    "TransposeBatchMatMul": "Matmul",
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
}

class SubGraphMatcher(object):
    def __call__(self, model, tune = False):
        logger.info('Start to implement Sub-Graph matching and replacing...') 
        if tune:
            self._tune_patterns(model)
        else:
            self._fuse_patterns(model)
        logger.info('Sub-Graph match and replace done...')
        return model

    def _fuse_patterns(self, model, supported_patterns=supported_patterns, pattern_mask=None):
        pattern_mask = [True for _ in range(len(supported_patterns))] \
                        if pattern_mask == None else pattern_mask
        for pattern_id, pattern in enumerate(supported_patterns):
            if pattern in PATTERNS and pattern_mask[pattern_id]:
                p_fusion = PATTERNS[pattern]()
                model = p_fusion(model)
        self._remove_identity(model) 
         
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
                    op_type = EXECUTOR_TYPE[node.op_type]
                    model.nodes[i].op_type = op_type
        model.remove_nodes(rm_node_names)

