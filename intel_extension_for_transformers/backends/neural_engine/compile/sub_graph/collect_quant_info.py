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

"""The CollectQuantInfo Pattern."""

from .pattern import Pattern, pattern_registry
from collections import namedtuple, OrderedDict
from .. import graph_utils as util
from .subgraph_matcher import EXECUTOR_TYPE
import numpy as np
import copy

@pattern_registry(pattern_type='CollectQuantInfo')
class CollectQuantInfo(Pattern):
    """The CollectQuantInfo pattern.

    Fuse the original sub-graph into the custom acceleration 'CollectQuantInfo' graph.
    The search strategy is based on the following pattern mapping configs for different models.
    """

    def __call__(self, model):
        """The __call__ function of this pattern class."""
        pattern_mapping_config = {
            'CollectQDQInfo': [
                {
                    'patterns': {
                        'in': [[(0, 'Quantize'), (1, ['DequantizeLinear'])]],
                    },
                },
                {
                    'patterns': {
                        'in': [[(0, 'DequantizeLinear')]],
                    },
                },
            ],
            'CollectTorchInfo': [
                {
                    'patterns': {
                        'in': [[(0, 'Quantize'), (1, ['Dequantize'])]],
                    },
                    
                },
                {
                    'patterns': {
                        'in': [[(0, 'Dequantize')]],
                    },
                },
            ],
            'RemoveQuantDequant': [
                {
                    'patterns': {
                        'in': [[(0, 'Quantize')]]
                    }
                },
                {
                    'patterns': {
                        'in': [[(0, 'DequantizeLinear')]]
                    }
                }
            ],
            'QLinearMatMul': [
                {
                    'patterns': {
                        'in': [[(0, 'QLinearMatMul')]],
                        'out': [[(0, 'MatMul')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 0
                    },
                    'input_tensors': {
                        0: [[{
                            0: [0]
                        }, {
                            0: [3]
                        }], [[0, 1], 2]]
                    },
                    'output_tensors': {
                        0: [[{
                            0: [0]
                        }], [[0], 1]]
                    },
                    'returns': [0]
                }
            ]
        }

        def CollectQDQInfo(model):
            # Collect the activation quant info
            pattern = pattern_mapping_config['CollectQDQInfo'][0]['patterns']['in']
            patterns_nodes_name = util.search_pattern(pattern, model)
            util.quant_info_init()
            new_dict = {}
            for pattern_nodes_name in patterns_nodes_name:
                quant_node = model.get_node_by_name(pattern_nodes_name[0])
                dquant_node = model.get_node_by_name(pattern_nodes_name[1])
                dquant_output = dquant_node.output_tensors[0]
                dtype = "s8" if dquant_node.input_tensors[2].data.dtype == 'int8' else "u8"
                max_range = 127 if dtype == "s8" else 255
                quant_max = (max_range -
                             dquant_node.input_tensors[2].data) * dquant_node.input_tensors[1].data
                quant_min = quant_max - 255 * dquant_node.input_tensors[1].data

                dtype = dtype + "_insert"
                util.insert_quant_info(quant_node.input_tensors[0].name, [quant_min, quant_max, dtype])
                for dst_op in dquant_output.dest_op:
                    dst_node = model.get_node_by_name(dst_op)
                    for idx, input_tensor in enumerate(dst_node.input_tensors):
                        if input_tensor.name == dquant_output.name:
                            for pre_quant_name in quant_node.input_tensors[0].source_op:
                                pre_quant_node = model.get_node_by_name(pre_quant_name)
                                if pre_quant_node.op_type in EXECUTOR_TYPE and \
                                    (EXECUTOR_TYPE[pre_quant_node.op_type] == "InnerProduct" or \
                                    EXECUTOR_TYPE[pre_quant_node.op_type] == "Matmul"):
                                    quant_info = util.get_quant_info()
                                    output_dtype = "output_" + dtype
                                    if model.get_node_by_name(pre_quant_node.input_tensors[0].
                                                              source_op[0]).op_type == "Transpose":
                                        input_tensor_name = model.get_node_by_name(
                                            pre_quant_node.input_tensors[0].source_op[0]
                                        ).input_tensors[0].name
                                        origin_quant_min = quant_info[
                                            pre_quant_node.input_tensors[0].name][0]
                                        origin_quant_max = quant_info[
                                            pre_quant_node.input_tensors[0].name][1]
                                        util.insert_quant_info(input_tensor_name, [
                                            origin_quant_min, origin_quant_max, output_dtype,
                                            quant_min, quant_max
                                        ])
                                    elif model.get_node_by_name(pre_quant_node.input_tensors[0].
                                                                source_op[0]).op_type == "Softmax":
                                        input_tensor_name = pre_quant_node.input_tensors[0].name
                                        origin_quant_min = quant_info[
                                            pre_quant_node.input_tensors[0].name][0]
                                        origin_quant_max = quant_info[
                                            pre_quant_node.input_tensors[0].name][1]
                                        util.insert_quant_info(input_tensor_name, [
                                            origin_quant_min, origin_quant_max, output_dtype,
                                            quant_min, quant_max
                                        ])
                                    else:
                                        input_tensor_name = pre_quant_node.input_tensors[1].name
                                        new_dict[input_tensor_name] = [quant_min, quant_max]

                                if pre_quant_node.op_type == "Transpose":
                                    util.insert_quant_info(pre_quant_node.input_tensors[0].name,
                                                           [quant_min, quant_max, dtype])
                                pre_quant_node.output_tensors[0].dest_op.append(dst_node.name)

                                dst_node.input_tensors[idx] = pre_quant_node.output_tensors[0]

                model.remove_nodes([pattern_nodes_name[0], pattern_nodes_name[1]])

            # Collect the weight quant info
            pattern = pattern_mapping_config['CollectQDQInfo'][1]['patterns']['in']
            patterns_nodes_name = util.search_pattern(pattern, model)
            for pattern_nodes_name in patterns_nodes_name:
                dquant_node = model.get_node_by_name(pattern_nodes_name[0])
                dquant_output = dquant_node.output_tensors[0]
                dtype = "s8" if dquant_node.input_tensors[2].data.dtype == 'int8' else "u8"
                dtype = dtype + "_weight"
                max_range = 127 if "s8" in dtype else 255
                quant_max = (max_range -
                             dquant_node.input_tensors[2].data) * dquant_node.input_tensors[1].data
                quant_min = quant_max - 255 * dquant_node.input_tensors[1].data
                util.insert_quant_info(dquant_node.input_tensors[0].name,
                                       [quant_min, quant_max, dtype])
                if dquant_output.name in new_dict:
                    old_quant_min = new_dict[dquant_output.name][0]
                    old_quant_max = new_dict[dquant_output.name][1]
                    dtype = "output_" + dtype
                    util.insert_quant_info(dquant_node.input_tensors[0].name,
                                           [quant_min, quant_max, dtype, old_quant_min, old_quant_max])
                for dst_op in dquant_output.dest_op:
                    matmul_node = model.get_node_by_name(dst_op)
                    # fall back int8 weight matmul op_type
                    if matmul_node.op_type == "BatchMatMul":
                        matmul_node.op_type = "MatMul"
                    for idx, input_tensor in enumerate(matmul_node.input_tensors):
                        if input_tensor.name == dquant_output.name:
                            matmul_node.input_tensors[idx] = dquant_node.input_tensors[0]
                model.remove_nodes([pattern_nodes_name[0]])

        def RemoveQuantDequant(model):
            # Remove quant/dequant
            remove_nodes_list = []
            for qpattern_info in pattern_mapping_config['RemoveQuantDequant']:
                qpattern = qpattern_info['patterns']['in']
                qnodes_name = util.search_pattern(qpattern, model)
                for qnode_name in qnodes_name:
                    qnode = model.get_node_by_name(qnode_name[0])
                    for qnode_input in qnode.input_tensors:
                        if not isinstance(qnode_input.data, np.ndarray):
                            # connect quant/dequant pre node with next node
                            qpre_node_name = qnode_input.source_op[0]
                            qpre_node = model.get_node_by_name(qpre_node_name)
                            qpre_node_out_dest_op = copy.deepcopy(qpre_node.output_tensors[0].dest_op)
                            # 1. the dest of previous node's output tensor connects to next node
                            for qpre_node_dst_op in qpre_node_out_dest_op:
                                if qpre_node_dst_op == qnode.name:
                                    qpre_node.output_tensors[0].dest_op.remove(qnode.name)
                                    qnode_output = qnode.output_tensors[0]
                                    for qnext_node in qnode_output.dest_op:
                                        qpre_node.output_tensors[0].dest_op.append(qnext_node)
                            # 2. the input of next node is current node's input
                            # 3. the source node of next node's input is previous node
                            qnode_output = qnode.output_tensors[0]
                            for qnext_node_name in qnode_output.dest_op:
                                qnext_node = model.get_node_by_name(qnext_node_name)
                                for idx in range(len(qnext_node.input_tensors)):
                                    if qnext_node.input_tensors[idx].name == qnode_output.name:
                                        qnext_node.input_tensors[idx] = qnode_input
                                        qnext_node.input_tensors[idx].source_op = [qpre_node.name]
                    remove_nodes_list.append(qnode_name[0])
            model.remove_nodes(remove_nodes_list)

        def QLinearMatMul(model):
            def scale2minmax(dtype, scale, zp):
                maxq = 127 if dtype == "s8" else 255
                quant_max = (maxq - zp) * scale
                quant_min = quant_max - 255 * scale
                return quant_min, quant_max

            def set_attr(new_node_names, model):
                for i in range(len(new_node_names)):
                    mat_node_idx = model.get_node_id(new_node_names[i][0])
                    attr = OrderedDict()
                    # see OneDNN InnerProduct related requirements
                    attr['transpose_a'] = False
                    attr['transpose_b'] = False
                    model.nodes[mat_node_idx].attr = attr
            # Collect the quant info
            pattern_dict = pattern_mapping_config['QLinearMatMul'][0]
            model, matmul_nodes_name, qmatmul_nodes = \
                        util.pattern_mapping("QLinearMatMul", pattern_dict, model)
            util.quant_info_init()
            for qmatmul_node in qmatmul_nodes:
                qmatmul_node = qmatmul_node[0]
                # input
                input = qmatmul_node.input_tensors[0]
                input_scale = qmatmul_node.input_tensors[1]
                input_zp = qmatmul_node.input_tensors[2]
                # input min max info
                input_dtype = "s8" if input_zp.data.dtype == "int8" else "u8"
                input_qdtype = input_dtype + "_weight" \
                            if type(input.data) == np.ndarray else input_dtype + "_insert"
                input_min, input_max = scale2minmax(input_dtype, input_scale.data, input_zp.data)
                util.insert_quant_info(input.name, [input_min, input_max, input_qdtype])
                # weight
                weight = qmatmul_node.input_tensors[3]
                weight_scale = qmatmul_node.input_tensors[4]
                weight_zp = qmatmul_node.input_tensors[5]
                # weight min max info
                weight_dtype = "s8" if weight_zp.data.dtype == "int8" else "u8"
                weight_qdtype = weight_dtype + "_weight" \
                            if type(weight.data) == np.ndarray else weight_dtype + "_insert"
                weight_min, weight_max = scale2minmax(weight_dtype, weight_scale.data, weight_zp.data)
                # util.insert_quant_info(weight.name, [weight_min, weight_max, weight_qdtype])
                # weight_s8 -> weight_fp32
                if type(weight.data) == np.ndarray :
                    weight.data = (weight.data-weight_zp.data) * weight_scale.data
                # output
                output = qmatmul_node.output_tensors[0]
                output_scale = qmatmul_node.input_tensors[6]
                output_zp = qmatmul_node.input_tensors[7]
                # output min max info
                output_dtype = "s8" if output_zp.data.dtype == "int8" else "u8"
                output_qdtype = "output_" + weight_qdtype # trick for insert_quant
                output_min, output_max = scale2minmax(output_dtype, output_scale.data, output_zp.data)
                # trick: for insert_quant_node.py
                util.insert_quant_info(weight.name, \
                        [weight_min, weight_max, output_qdtype, output_min, output_max])
                # trick: for insert_quant_node.py
                for id in range(len(qmatmul_node.input_tensors)):
                    for pre_op_name in qmatmul_node.input_tensors[id].source_op:
                        pre_op = model.get_node_by_name(pre_op_name)
                        pre_out_tensor_name = pre_op.output_tensors[0].name
                        qmatmul_in_tensor_name = qmatmul_node.input_tensors[id].name
                        if pre_out_tensor_name == qmatmul_in_tensor_name:
                            # trick: collect softmax output min/max(fp32 in, int8 out in neural engine)
                            if pre_op.op_type == "Softmax":
                                origin_scale = qmatmul_node.input_tensors[id+1]
                                origin_zp = qmatmul_node.input_tensors[id+2]
                                origin_min, origin_max = scale2minmax(origin_zp.data.dtype,
                                                            origin_scale.data, origin_zp.data)
                                output_scale = qmatmul_node.input_tensors[-2]
                                output_zp = qmatmul_node.input_tensors[-1]
                                output_min, output_max = scale2minmax(output_zp.data.dtype,
                                                            output_scale.data, output_zp.data)
                                output_zp_dtype = "s8" if output_zp.data.dtype == 'int8' else "u8"
                                output_dtype = "output_" + output_zp_dtype + "_insert"
                                util.insert_quant_info(pre_out_tensor_name,
                                    [origin_min, origin_max, output_dtype, output_min, output_max])
                            # trick: pre transpose has the same min/max as matmul,
                            elif pre_op.op_type == "Transpose":
                                qmatmul_in_scale = qmatmul_node.input_tensors[id+1]
                                qmatmul_in_zp = qmatmul_node.input_tensors[id+2]
                                qmatmul_in_min, qmatmul_in_max = scale2minmax(qmatmul_in_zp.data.dtype,
                                                            qmatmul_in_scale.data, qmatmul_in_zp.data)
                                trans_in_tensor_name = pre_op.input_tensors[0].name
                                qmatmul_in_zp_dtype = "s8" if qmatmul_in_zp.data.dtype == 'int8' else "u8"
                                qmatmul_in_dtype = qmatmul_in_zp_dtype + "_insert"
                                util.insert_quant_info(trans_in_tensor_name,
                                    [qmatmul_in_min, qmatmul_in_max, qmatmul_in_dtype])
                # matmul has tranpose attr, but qlinear matmul has not
                set_attr(matmul_nodes_name, model)

        def CollectTorchInfo(model):
            # Collect the activation quant info
            pattern = pattern_mapping_config['CollectTorchInfo'][0]['patterns']['in']
            patterns_nodes_name = util.search_pattern(pattern, model)
            new_dict = {}
            for pattern_nodes_name in patterns_nodes_name:
                quant_node = model.get_node_by_name(pattern_nodes_name[0])
                dquant_node = model.get_node_by_name(pattern_nodes_name[1])
                scale = quant_node.attr['scale']
                zp = quant_node.attr['zero_point']
                dquant_output = dquant_node.output_tensors[0]
                dtype = "u8" if quant_node.attr['dtype'] == 13 else "s8"
                max_range = 127 if dtype == "s8" else 255
                quant_max = np.array((max_range - zp) * scale)
                quant_min = np.array(quant_max - 255 * scale)

                dtype = dtype + "_insert"
                map_name = quant_node.input_tensors[0].name
                pre_node = quant_node
                while True:
                    pre_node = model.get_node_by_name(pre_node.input_tensors[0].source_op[0])
                    if pre_node.op_type == 'Reorder':
                        map_name = pre_node.input_tensors[0].name
                    else:
                        break
                util.insert_quant_info(map_name, [quant_min, quant_max, dtype])
                for dst_op in dquant_output.dest_op:
                    dst_node = model.get_node_by_name(dst_op)
                    for idx, input_tensor in enumerate(dst_node.input_tensors):
                        if input_tensor.name == dquant_output.name:
                            for pre_quant_name in quant_node.input_tensors[0].source_op:
                                pre_quant_node = model.get_node_by_name(pre_quant_name)
                                for pre_quant_node_out_tensor in pre_quant_node.output_tensors:
                                    if pre_quant_node_out_tensor.name == quant_node.input_tensors[0].name:
                                        if quant_node.name in pre_quant_node_out_tensor.dest_op:
                                            pre_quant_node_out_tensor.dest_op.remove(quant_node.name)
                                        pre_quant_node_out_tensor.dest_op.append(dst_node.name)
                                        dst_node.input_tensors[idx] = pre_quant_node_out_tensor
                model.remove_nodes([pattern_nodes_name[0], pattern_nodes_name[1]])
            # Collect the weight quant info
            pattern = pattern_mapping_config['CollectTorchInfo'][1]['patterns']['in']
            patterns_nodes_name = util.search_pattern(pattern, model)
            for pattern_nodes_name in patterns_nodes_name:
                dquant_node = model.get_node_by_name(pattern_nodes_name[0])
                dquant_output = dquant_node.output_tensors[0]
                for dst_op in dquant_output.dest_op:
                    dst_node = model.get_node_by_name(dst_op)
                    for idx, input_tensor in enumerate(dst_node.input_tensors):
                        if input_tensor.name == dquant_output.name:
                            dquant_node.input_tensors[0].dest_op.remove(dquant_node.name)
                            dquant_node.input_tensors[0].dest_op.append(dst_node.name)
                            dst_node.input_tensors[idx] = dquant_node.input_tensors[0]
                model.remove_nodes([pattern_nodes_name[0]])

        if model.framework_modeling_config['framework'] == 'torch':
            CollectTorchInfo(model)
            return model

        # if ONNX QDQ model, only CollectQDQInfo
        # if ONNX QLinear model, RemoveQuantDequant and QLinearMatMul
        is_qlinear_graph = False
        for node in model.nodes:
            if node.op_type == "QLinearMatMul":
                is_qlinear_graph = True
        if not is_qlinear_graph:
            CollectQDQInfo(model)
        else:
            RemoveQuantDequant(model)
            QLinearMatMul(model)

        return model
