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
import copy


@pattern_registry(pattern_type='QuantizedGraphDtypeRefactor')
class QuantizedGraphDtypeCheck(Pattern):
    """The QuantizedGraphDtypeRefactor pattern.
    Fuse the original sub-graph into the custom acceleration 'QuantizedGraphDtypeRefactor' graph.
    The search strategy is based on the following pattern mapping configs for different models.
    """

    def __call__(self, model):
        """The __call__ function of this pattern class."""

        def _quantized_dtype(model):
            dtype = 'fp32'
            for node in model.nodes:
                if node.op_type == 'Quantize':
                    dtype = node.attr['output_dtype']
                    break
            return dtype

        def _get_dst_ops(node, model):
            ret = []
            output_name = node.output_tensors[0].name
            for node in model.nodes:
                for input_tensor in node.input_tensors:
                    if output_name == input_tensor.name:
                        ret.append(node)
            return ret

        def _insert_quant_node(pre_node, model, graph_dtype):
            output_tensor = copy.deepcopy(pre_node.output_tensors[0])
            output_tensor.dtype = graph_dtype
            pre_node.output_tensors[0] = copy.deepcopy(pre_node.output_tensors[0])
            pre_node.output_tensors[0].name = pre_node.output_tensors[0].name + '_before_quant'
            pre_node.output_tensors[0].dest_op = [pre_node.name + '_quant']
            input_tensor = copy.deepcopy(pre_node.output_tensors[0])
            quantize_op = util.construct_node(node_name=pre_node.name + "_quant",
                                              op_type='Quantize',
                                              input_tensors=[input_tensor],
                                              output_tensors=[output_tensor],
                                              attr=OrderedDict({'output_dtype': graph_dtype}))
            insert_idx = model.get_node_id(pre_node.name)
            model.insert_nodes(insert_idx + 1, [quantize_op])

        def _check_dst_op(start_node, model, checker, graph_dtype):
            dst_ops = _get_dst_ops(start_node, model)
            if len(dst_ops) == 0:
                return
            for op in dst_ops:
                if EXECUTOR_TYPE.get(op.op_type, op.op_type) in checker[graph_dtype]:
                    if op.attr.get('output_dtype', None) != graph_dtype:
                        op.attr['output_dtype'] = graph_dtype
                        _check_dst_op(op, model, checker, graph_dtype)
                else:
                    continue

        def _scan_nodes_graph_dtype(model, checker, graph_dtype):
            for node in model.nodes:
                if node.attr and node.attr.get('output_dtype', 'fp32') == graph_dtype and node.name != 'input_data':
                    dst_ops = _get_dst_ops(node, model)
                    for op in dst_ops:
                        if EXECUTOR_TYPE.get(op.op_type, op.op_type) in checker[graph_dtype] and \
                            op.attr.get('output_dtype', 'fp32') != graph_dtype:
                            op.attr['output_dtype'] = graph_dtype

        def _remove_redundant_quant_node(model, checker, graph_dtype):
            remove_node_names = []
            for node in model.nodes:
                if node.op_type == 'Quantize':
                    pre_node = None
                    try:
                        pre_node = model.get_node_by_name(node.input_tensors[0].source_op[0])
                    except:
                        pre_node = None
                    if pre_node and pre_node.attr and pre_node.attr.get('output_dtype', None) \
                       == graph_dtype:
                        dst_ops = _get_dst_ops(pre_node, model)
                        if len(dst_ops) == 1 and dst_ops[0].op_type == "Quantize":
                            pre_node.output_tensors = copy.deepcopy(node.output_tensors)
                            remove_node_names.append(node.name)
                        else:
                            if len(dst_ops) >= 2:
                                dst_ops_type = [EXECUTOR_TYPE.get(o.op_type, o.op_type)
                                                for o in dst_ops]
                                valid = True
                                for ot in dst_ops_type:
                                    if ot in checker[graph_dtype] or ot in ['Quantize', 'Output']:
                                        continue
                                    else:
                                        valid = False
                                        break
                                if valid:
                                    quant_dst_ops = _get_dst_ops(node, model)
                                    qot_name = node.output_tensors[0].name
                                    for qdo in quant_dst_ops:
                                        r_tensor_idx = model.get_tensor_idx(qdo.name, qot_name,
                                                                            from_output=False)
                                        if r_tensor_idx != -1:
                                            pre_node.output_tensors[0].dest_op.append(qdo.name)
                                            qdo.input_tensors[r_tensor_idx] = copy.deepcopy(
                                                pre_node.output_tensors[0])
                                    remove_node_names.append(node.name)
            model.remove_nodes(remove_node_names)

        def _check_append_sum_nodes(model, checker, graph_dtype):
            name_list = []
            for node in model.nodes:
                if EXECUTOR_TYPE.get(node.op_type, node.op_type) in \
                   ['InnerProduct', 'Convolution'] and node.attr and node.attr.get('append_op') \
                   == 'sum' and graph_dtype == 'bf16':
                    post_n = model.get_node_by_name(node.input_tensors[-1].source_op[0])
                    if post_n.attr and post_n.attr.get('output_dtype', 'fp32') != 'bf16':
                        if post_n.name not in name_list:
                            name_list.append(post_n.name)

            if name_list:
                for n in name_list:
                    _insert_quant_node(model.get_node_by_name(n), model, graph_dtype)
                _scan_nodes_graph_dtype(model, checker, graph_dtype)

        def _revert_logits_output_dtype(model, graph_dtype):
            for t in model.nodes[-1].input_tensors:
                pre_node = model.get_node_by_name(t.source_op[0])
                if pre_node and EXECUTOR_TYPE.get(pre_node.op_type, pre_node.op_type) in \
                   ['InnerProduct', 'Softmax', 'LogSoftmax', 'Convolution']:
                    if pre_node.attr.get('output_dtype', "fp32") == graph_dtype:
                        pre_node.attr['output_dtype'] = 'fp32'

        if util.get_autocast_info()['cast_type'] != "bf16":
            return model
        graph_dtype = _quantized_dtype(model)
        # all in and out tensors support the dtype
        bf16_op = [
            'InnerProduct', 'Slice', 'Matmul', 'Reshape', 'BinaryOp', 'BinaryAdd', 'Reorder',
            'Concat', 'Softmax', 'LayerNorm', 'LogSoftmax', 'Convolution', 'Gather', 'GroupNorm',
            'Sigmoid', 'Gelu', 'MultiHeadAttention', 'Resampling','StridedSlice'
        ]
        s8_op = ['InnerProduct', 'Reshape', 'Shape', 'BinaryOp']
        checker = {'bf16': bf16_op, 's8': s8_op}

        non_quantized_patterns = [[(0, 'Range'), (1, ['Div', 'BinaryOp']), (2, 'Pow'),
                                   (3, ['Div', 'BinaryOp']), (4, 'Reshape'),
                                   (7, ['Matmul', 'BatchMatmul'])],
                                  [(), (5, 'Range'), (6, 'Reshape'), (7, ['Matmul',
                                                                          'BatchMatMul'])]]
        match_ret = util.search_pattern(non_quantized_patterns, model)
        for ret in match_ret:
            mat_node = model.get_node_by_name(ret[-2])
            dst_ops = _get_dst_ops(mat_node, model)
            dst_ops_type = [n.op_type for n in dst_ops]
            if dst_ops_type == ['Sin', 'Cos'] or dst_ops_type == ['CosSin', 'CosSin']:
                for n in dst_ops:
                    _insert_quant_node(n, model, graph_dtype)

        if graph_dtype in ['bf16']:
            for node in model.nodes:
                if EXECUTOR_TYPE.get(node.op_type, node.op_type) in ['Quantize'] and \
                   node.attr['output_dtype'] == "bf16":
                    dst_ops = _get_dst_ops(node, model)
                    for op in dst_ops:
                        if op.attr and op.attr.get('output_dtype', 'fp32') != graph_dtype and \
                            EXECUTOR_TYPE.get(op.op_type, op.op_type) in checker[graph_dtype]:
                            op.attr['output_dtype'] = graph_dtype
                            _check_dst_op(op, model, checker, graph_dtype)
            _scan_nodes_graph_dtype(model, checker, graph_dtype)
            _check_append_sum_nodes(model, checker, graph_dtype)
            _remove_redundant_quant_node(model, checker, graph_dtype)
            _revert_logits_output_dtype(model, graph_dtype)
            for t in model.nodes[0].output_tensors:
                if t.location and len(t.location) == 2:
                    break
                else:
                    if (match_ret or \
                        model._framework_modeling_config.get('architecture', 'None') == \
                        'decoder_only') and t.dtype == "fp32":
                        t.dtype = "bf16"

        return model
