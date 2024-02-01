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
"""The ExplicitNHWCTransposeForConvQAT Pattern."""

from .pattern import Pattern, pattern_registry
from collections import namedtuple, OrderedDict
from .. import graph_utils as util
from .. import logger
from .subgraph_matcher import EXECUTOR_TYPE
import numpy as np
import copy


@pattern_registry(pattern_type='ExplicitNHWCTransposeForConvQAT')
class ExplicitNHWCTransposeForConvQAT(Pattern):
    """The ExplicitNHWCTransposeForConvQAT pattern.

    Fuse the original sub-graph into the custom acceleration 'ExplicitNHWCTransposeForConvQAT' graph.
    The search strategy is based on the following pattern mapping configs for different models.
    """

    def __call__(self, model):
        """The __call__ function of this pattern class."""
        pattern_mapping_config = {
            'ExplicitNHWCTransposeForConvQAT': [
                {
                    'patterns': {
                        'in': [[(0, 'Conv')]],
                        'out': [[(0, 'Transpose'), (1, 'Conv'), (2, 'Transpose')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 'reorder_pre_for_conv',
                        1: 0,
                        2: 'reorder_post_for_conv',
                    },
                    'input_tensors': {
                        0: [[{
                            0: [0]
                        }], [[0], 1]],
                        1: [[{
                            0: [1]
                        }, {
                            0: [2]
                        },{
                            0: [3]
                        },{
                            0: [4]
                        },{
                            0: [5]
                        },{
                            0: [6]
                        },{
                            0: [7]
                        },{
                            0: [8]
                        }], [[1, 2, 3, 4, 5, 6, 7, 8], 9]],
                        2: [[], [[], 1]]
                    },
                    'output_tensors': {
                        0: [[], [[], 1]],
                        1: [[], [[], 1]],
                        2: [[{
                            0: [0]
                        }], [[0], 1]]
                    },
                    'returns': [0]
                },
            ]
        }

        pattern_dict = pattern_mapping_config['ExplicitNHWCTransposeForConvQAT'][0]
        model, new_node_names, ret_old_nodes = \
            util.pattern_mapping("ExplicitNHWCTransposeForConvQAT", pattern_dict, model)
        if len(new_node_names) != 0:
            logger.info('ExplicitNHWCTransposeForConvQAT matched...')
            logger.debug('ExplicitNHWCTransposeForConvQAT = {}'.format(new_node_names))
            for i in range(len(new_node_names)):
                conv_attr = OrderedDict()
                conv_node_idx = model.get_node_id(new_node_names[i][1])
                conv_attr = ret_old_nodes[i][0].attr
                conv_node = model.nodes[conv_node_idx]
                conv_node.attr = conv_attr
                conv_node.attr['src_perm'] = '0,3,1,2'
                conv_node.attr['dst_perm'] = '0,2,3,1'

                # the first node
                attr = OrderedDict()
                reorder_pre_node = model.get_node_by_name(new_node_names[i][0])
                attr['src_perm'] = '0,1,2,3'
                attr['dst_perm'] = '0,2,3,1'
                reorder_pre_node.attr = attr

                # the third node
                attr_2 = OrderedDict()
                reorder_post_node = model.get_node_by_name(new_node_names[i][2])
                attr_2['src_perm'] = '0,1,2,3'
                attr_2['dst_perm'] = '0,3,1,2'
                reorder_post_node.attr = attr_2

                if 'output_dtype' in conv_attr:
                    if 'fp32' in conv_attr['output_dtype']:
                        reorder_pre_node.attr['output_dtype'] = 'bf16'
                        conv_node.attr['output_dtype'] = 'bf16'
                        reorder_post_node.attr['output_dtype'] = 'fp32'
                    else:
                        reorder_post_node.attr['output_dtype'] = conv_attr['output_dtype']
                        reorder_pre_node.attr['output_dtype'] = conv_attr['output_dtype']
                else:
                    reorder_pre_node.attr['output_dtype'] = 'u8'
                    conv_node.attr['output_dtype'] = 'bf16'
                    reorder_post_node.attr['output_dtype'] = 'bf16'

        if len(new_node_names) != 0:
            remove_node_name = []
            for transpose_node in model.nodes:
                if transpose_node.op_type == 'Transpose':
                    node_id = model.get_node_id(transpose_node.name)
                    second_transpose_node = model.nodes[node_id + 1]
                    if second_transpose_node.op_type == 'Transpose' \
                        and second_transpose_node.input_tensors[0].name == transpose_node.output_tensors[0].name:
                        if transpose_node.attr['dst_perm'] == '0,3,1,2' \
                            and second_transpose_node.attr['dst_perm'] == '0,2,3,1':
                            remove_node_name.append(transpose_node.name)
                            remove_node_name.append(second_transpose_node.name)

                            pre_node = model.nodes[node_id - 1]
                            target_node = model.nodes[node_id + 2]
                            pre_node.output_tensors[0] = transpose_node.output_tensors[0]
                            target_node.input_tensors[0] = transpose_node.output_tensors[0]

            model.remove_nodes(remove_node_name)

            #Convert Innerproduct and Conv bias for ONEDNN 3.x
            for node in model.nodes:
                if node.op_type in EXECUTOR_TYPE and \
                    EXECUTOR_TYPE[node.op_type] == 'InnerProduct':
                    # convert s32 bias to fp32 bias due to ONEDNN 3.x required
                    weight_s8 = node.input_tensors[1].data
                    bias_s32 = node.input_tensors[2].data
                    offset = 1 if node.attr.get("append_op", "") == "sum" else 0
                    activation_min = node.input_tensors[3 + offset].data
                    activation_max = node.input_tensors[4 + offset].data
                    weight_min = node.input_tensors[5 + offset].data
                    weight_max = node.input_tensors[6 + offset].data
                    activation_scale = ((activation_max - activation_min) / 255).astype(float)
                    weight_scale = (np.maximum(abs(weight_max), abs(weight_min)) /
                                    127).astype(float)
                    bias_fp32 = (bias_s32 * activation_scale * weight_scale).astype(np.float32)
                    if node.attr.get("src1_perm", "0,1") == "0,1":
                        compensation = activation_min * weight_scale * weight_s8.sum(
                            -1).astype(np.float32)
                    else:
                        compensation = activation_min * weight_scale * weight_s8.sum(
                            0).astype(np.float32)

                    node.input_tensors[2].data = copy.deepcopy((bias_fp32 + compensation).astype(np.float32))

                if node.op_type in EXECUTOR_TYPE and \
                    EXECUTOR_TYPE[node.op_type] == 'Convolution':
                    # convert s32 bias to fp32 bias due to ONEDNN 3.x required
                    weight_s8 = node.input_tensors[1].data
                    bias_s32 = node.input_tensors[2].data
                    offset = 1 if node.attr.get("append_op", "") == "sum" else 0
                    activation_min = node.input_tensors[3 + offset].data
                    activation_max = node.input_tensors[4 + offset].data
                    weight_min = node.input_tensors[5 + offset].data
                    weight_max = node.input_tensors[6 + offset].data
                    activation_scale = ((activation_max - activation_min) / 255).astype(float)
                    weight_scale = (np.maximum(abs(weight_max), abs(weight_min)) /
                                    128).astype(float)
                    bias_fp32 = (bias_s32 * activation_scale * weight_scale).astype(np.float32)
                    compensation = 0
                    node.input_tensors[2].data = copy.deepcopy((bias_fp32 + compensation).astype(np.float32))

        # modify output name and remove useless outputs
        for tensor in model.nodes[-1].input_tensors:
            if  model.nodes[-2].output_tensors[0].name == tensor.name:
                model.nodes[-2].output_tensors[0].name = 'out_sample:0'
                tensor.name = model.nodes[-2].output_tensors[0].name
                model.nodes[-2].attr['output_dtype'] = 'fp32'

        if len(model.nodes[-1].input_tensors) != 1:
            del model.nodes[-1].input_tensors[:-1]

        def _mixed_bf16_precision(model):
            def _get_dst_ops(node, model):
                ret = []
                output_name = node.output_tensors[0].name
                for node in model.nodes:
                    for input_tensor in node.input_tensors:
                        if output_name == input_tensor.name:
                            ret.append(node)
                return ret

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

            def _revert_logits_output_dtype(model, output_dtype):
                for t in model.nodes[-1].input_tensors:
                    pre_node = model.get_node_by_name(t.source_op[0])
                    if pre_node and EXECUTOR_TYPE.get(pre_node.op_type, pre_node.op_type) in \
                    ['InnerProduct', 'Softmax', 'LogSoftmax']:
                        if pre_node.attr.get('output_dtype', "fp32") == output_dtype:
                            pre_node.attr['output_dtype'] = 'fp32'
            # all in and out tensors support the dtype
            bf16_op = ['InnerProduct', 'Slice', 'Matmul', 'Reshape', 'BinaryOp',
                    'BinaryAdd', 'Reorder', 'Concat', 'Softmax', 'LayerNorm',
                    'LogSoftmax', 'Convolution', 'Gather']
            int8_op = ['InnerProduct', 'Matmul']
            checker = {'bf16': bf16_op, 'int8': int8_op}
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
            _revert_logits_output_dtype(model, "bf16")

            for node in model.nodes:
                if  node.op_type in EXECUTOR_TYPE and \
                    (EXECUTOR_TYPE[node.op_type] == 'BinaryAdd' or EXECUTOR_TYPE[node.op_type] == 'BinaryOp'):
                    # for the first BinaryOp /time_proj/Mul
                    if model.get_node_by_name(node.output_tensors[0].dest_op[0]).op_type == 'Sin':
                        continue
                    node.attr['output_dtype'] = 'bf16'
            return model

        _mixed_bf16_precision(model)

        return model
