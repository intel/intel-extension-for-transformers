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

"""The QuantizedGraphDtypeRefactor Pattern."""

from .pattern import Pattern, pattern_registry
from collections import namedtuple, OrderedDict
from ..ops import Tensor
from .. import graph_utils as util
from .subgraph_matcher import EXECUTOR_TYPE
from .. import logger
import copy

@pattern_registry(pattern_type='Int8BF16MixedPrecisionChecker')
class Int8BF16MixedPrecisionChecker(Pattern):
    """The Int8BF16MixedPrecisionChecker pattern.

    Check if the model can be inferenced under in8/bf16 mixed precision. And modify graph if need.
    Quant -> InnerProduct / Matmul -> other Op
    u8 / s8 -> fp32 -> fp32 -----> u8 / s8 -> bf16 -> bf16
    """
    def __call__(self, model):
        """The __call__ function of this pattern class."""
        if model.framework_modeling_config['framework'] != 'torch':
            return model
        quant_info = util.get_quant_info()
        if not quant_info:
            return model
        def _get_dst_ops(node, model):
            ret = []
            output_name = node.output_tensors[0].name
            for node in model.nodes:
                for input_tensor in node.input_tensors:
                    if output_name == input_tensor.name:
                        ret.append(node)
            return ret

        def _all_dst_ops(node, model, checker, output_dtype):
            dst_ops = _get_dst_ops(node, model)
            for op in dst_ops:
                valid_connections = False if model.get_tensor_idx(op.name,
                                    node.output_tensors[0].name, from_output=False) == -1 else True
                o_type = EXECUTOR_TYPE.get(op.op_type, op.op_type)
                if not valid_connections or o_type in ['Shape', 'Quantize', 'Output', 'Softmax']:
                    continue
                if o_type not in checker[output_dtype] or \
                   op.attr.get('output_dtype', 'fp32') != output_dtype:
                    return False
            return True
        
        def _insert_bf16_quant_node(pre_node, model):
            output_tensor = copy.deepcopy(pre_node.output_tensors[0])
            output_tensor.dtype = 'bf16'
            pre_node.output_tensors[0].name = pre_node.output_tensors[0].name + '_before_quant'
            pre_node.output_tensors[0].dest_op = [pre_node.name + '_quant']
            input_tensor = copy.deepcopy(pre_node.output_tensors[0])
            quantize_op = util.construct_node(
                node_name=pre_node.name + "_quant",
                op_type='Quantize',
                input_tensors=[input_tensor],
                output_tensors=[output_tensor],
                attr=OrderedDict({'output_dtype': 'bf16'}))
            insert_idx = model.get_node_id(pre_node.name)
            model.insert_nodes(insert_idx + 1, [quantize_op])
            return model

        def _check_dst_op(start_node, model, checker, output_dtype):
            dst_ops = _get_dst_ops(start_node, model)
            if len(dst_ops) == 0:
                return
            for op in dst_ops:
                if EXECUTOR_TYPE.get(op.op_type, op.op_type) in checker[output_dtype]:
                    if op.attr.get('output_dtype', 'fp32') == 'fp32':
                        op.attr['output_dtype'] = output_dtype
                        _check_dst_op(op, model, checker, output_dtype)
                else:
                    continue

        def _scan_nodes_graph_dtype(model, checker, output_dtype):
            for node in model.nodes:
                o_type = EXECUTOR_TYPE.get(node.op_type, node.op_type)
                # int8 matrix multiply can output bf16 dtype but fp32 cannot
                if (o_type in ['InnerProduct', 'Matmul', 'Convolution',] and \
                    len(node.input_tensors) > 3):
                    if node.attr.get('output_dtype', 'fp32') == 'fp32':
                        node.attr['output_dtype'] = output_dtype
                        _check_dst_op(node, model, checker, output_dtype)

        def _fall_back_to_fp32(model, checker):
            for node in model.nodes:
                fall_back_node = None
                if EXECUTOR_TYPE.get(node.op_type, node.op_type) in ['Softmax'] and \
                   node.attr.get('output_dtype', 'fp32') in ['u8', 's8']:
                    pre_node = None
                    try:
                        pre_node = model.get_node_by_name(node.input_tensors[0].source_op[0])
                    except:
                        pre_node = None
                    if pre_node and pre_node.attr.get('output_dtype', 'fp32') == 'bf16':
                        fall_back_node = pre_node
                else:
                    if node.attr and node.attr.get('output_dtype', 'fp32') == 'bf16':
                        if not _all_dst_ops(node, model, checker, 'bf16'):
                            fall_back_node = node
                if fall_back_node:
                    if EXECUTOR_TYPE.get(fall_back_node.op_type, fall_back_node.op_type) in \
                       checker['bf16_fp32_mixed']:
                        fall_back_node.attr['output_dtype'] = 'fp32'
                    else:
                        logger.warning("{} can not fall back into fp32 dtype".format(
                                        fall_back_node.name))
                        return False
            return True

        def _revert_logits_output_dtype(model, output_dtype):
            for t in model.nodes[-1].input_tensors:
                pre_node = model.get_node_by_name(t.source_op[0])
                if pre_node and EXECUTOR_TYPE.get(pre_node.op_type, pre_node.op_type) in \
                   ['InnerProduct', 'Softmax', 'LogSoftmax']:
                    if pre_node.attr.get('output_dtype', "fp32") == output_dtype:
                        pre_node.attr['output_dtype'] = 'fp32'
        model_cpy = copy.deepcopy(model)
        # all in and out tensors support the dtype
        bf16_op = ['InnerProduct', 'Slice', 'Matmul', 'Reshape', 'BinaryOp',
                   'BinaryAdd', 'Reorder', 'Concat', 'Softmax', 'LayerNorm',
                   'LogSoftmax', 'Convolution', 'Gather']
        int8_op = ['InnerProduct', 'Matmul']
        bf16_fp32_mixed_op = ['InnerProduct', 'Matmul', 'BinaryOp', 'BinaryAdd', 'LayerNorm',
                              'Convolution', 'Softmax', 'Reorder', 'LogSoftmax']
        checker = {'bf16': bf16_op, 'int8': int8_op, 'bf16_fp32_mixed': bf16_fp32_mixed_op}
        # quant cos and sin nodes to bf16
        non_quantized_patterns = [[(0, 'Range'), (1, ['Div', 'BinaryOp']), (2, 'Pow'),
                                  (3, ['Div', 'BinaryOp']), (4, 'Reshape'),
                                  (7, ['Matmul', 'Einsum', 'BatchMatmul'])],
                                [(), (5, 'Range'), (6, 'Reshape'), 
                                 (7, ['Matmul', 'Einsum', 'BatchMatMul'])]]
        match_ret = util.search_pattern(non_quantized_patterns, model)
        for ret in match_ret:
            mat_node = model.get_node_by_name(ret[-2])
            dst_ops = _get_dst_ops(mat_node, model)
            dst_ops_type = [EXECUTOR_TYPE.get(n.op_type, n.op_type) for n in dst_ops]
            if dst_ops_type == ['CosSin', 'CosSin'] or dst_ops_type == ['Sin', 'Cos']:
                for n in dst_ops:
                    model = _insert_bf16_quant_node(n, model)

        for node in model.nodes:
            if EXECUTOR_TYPE.get(node.op_type, node.op_type) == 'Quantize' and \
               node.attr['output_dtype'] in ['u8', 's8', 'bf16']:
                dst_ops = _get_dst_ops(node, model)
                for op in dst_ops:
                    if op.attr.get('output_dtype', 'fp32') == 'fp32' and \
                        EXECUTOR_TYPE.get(op.op_type, op.op_type) in checker['bf16']:
                        op.attr['output_dtype'] = 'bf16'
                        _check_dst_op(op, model, checker, 'bf16')
        _scan_nodes_graph_dtype(model, checker, 'bf16')
        status = _fall_back_to_fp32(model, checker)
        _revert_logits_output_dtype(model, "bf16")
        for t in model.nodes[0].output_tensors:
            if t.location and len(t.location) == 2:
                break
            else:
                if t.dtype == "fp32":
                    t.dtype = "bf16"
        if status:
            del model_cpy
            return model
        else:
            del model
            logger.warning("The model cannot use int8/bf16 mixed precision...")
            return model_cpy
