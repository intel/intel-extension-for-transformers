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

"""The InsertQuantNode Pattern."""

from .pattern import Pattern, pattern_registry
from collections import namedtuple, OrderedDict
from .. import graph_utils as util
from ..ops import Tensor
from .subgraph_matcher import EXECUTOR_TYPE
import numpy as np
import copy

@pattern_registry(pattern_type='InsertQuantNode')
class InsertQuantNode(Pattern):
    """The InsertQuantNode pattern.

    Fuse the original sub-graph into the custom acceleration 'InsertQuantNode' graph.
    The search strategy is based on the following pattern mapping configs for different models.
    """

    def __call__(self, model):
        """The __call__ function of this pattern class."""
        def get_scale_zp(tensor_min_data, tensor_max_data, dtype):
            # for asym quant
            if (dtype == 'u8'):
                max_sub_min = tensor_max_data - tensor_min_data
                scale = np.where(max_sub_min == 0., 0., 255. / max_sub_min)
                zero_point = -tensor_min_data * scale
            # for sym quant
            elif (dtype == 's8'):
                max_abs = np.maximum(np.fabs(tensor_min_data), np.fabs(tensor_max_data))
                scale = np.where(max_abs == 0., 0., 127. / max_abs)
                zero_point = 0.
            return scale, zero_point

        quant_info = util.get_quant_info()
        if not quant_info:
            return model
        if model.framework_modeling_config['framework'] == 'torch':
            unique_quant_nodes = {}
            node_idx = 0
            while node_idx < len(model.nodes):
                node = model.nodes[node_idx]
                if node.op_type in EXECUTOR_TYPE and \
                (EXECUTOR_TYPE[node.op_type] == "InnerProduct" or \
                    EXECUTOR_TYPE[node.op_type] == "Matmul"):
                    input_size = len(node.input_tensors)
                    if "reshape_dims" in node.attr:
                        input_size -= 1
                    insert_offset = input_size

                    idx = 0
                    while idx < input_size:
                        input_tensor = node.input_tensors[idx]
                        input_name = input_tensor.name
                        if input_name in quant_info and idx < 3:
                            quant_min = Tensor(
                                name=input_name + "_min",
                                shape=[quant_info[input_name][0].size],
                                data=np.array(quant_info[input_name][0].astype("float32")),
                                dtype="fp32")
                            quant_max = Tensor(
                                name=input_name + "_max",
                                shape=[quant_info[input_name][1].size],
                                data=np.array(quant_info[input_name][1].astype("float32")),
                                dtype="fp32")

                            if "insert" in quant_info[input_name][2]:
                                if input_name in unique_quant_nodes:
                                    quantize_op_name = unique_quant_nodes[input_name]
                                    quant_node = model.get_node_by_name(quantize_op_name)
                                    quant_node.output_tensors[0].dest_op.append(node.name)
                                    node.input_tensors[idx] = quant_node.output_tensors[0]
                                else:
                                    quant_dtype = "u8" if "u8" in quant_info[input_name][2] else "s8"
                                    quant_dtype = "s8" if EXECUTOR_TYPE[
                                        node.op_type] == "Matmul" else quant_dtype
                                    quant_output = Tensor(name=input_name + "_quant",
                                                        source_op=[node.name + "_quant_" + str(idx)],
                                                        dest_op=[node.name],
                                                        dtype=quant_dtype)
                                    quantize_op = util.construct_node(
                                        node_name=node.name + "_quant_" + str(idx),
                                        op_type='Quantize',
                                        input_tensors=[input_tensor, quant_min, quant_max],
                                        output_tensors=[quant_output],
                                        attr=OrderedDict({'output_dtype': quant_dtype}))
                                    unique_quant_nodes[input_name] = quantize_op.name
                                    node.input_tensors[idx] = quant_output
                                    insert_idx = model.get_node_id(node.name)
                                    model.insert_nodes(insert_idx, [quantize_op])
                                    node_idx += 1
                                # insert src0/src1 min and max tensor
                                model.change_node_input_tensors(node.name, insert_offset + 2 * idx + 0,
                                                                quant_min, 'insert')
                                model.change_node_input_tensors(node.name, insert_offset + 2 * idx + 1,
                                                                quant_max, 'insert')
                            if "weight" in quant_info[input_name][2]:
                                # insert weight min and max tensor
                                model.change_node_input_tensors(node.name, insert_offset + 2 * idx + 0,
                                                                quant_min, 'insert')
                                model.change_node_input_tensors(node.name, insert_offset + 2 * idx + 1,
                                                                quant_max, 'insert')
                            if "output" in quant_info[input_name][2]:
                                output_name = node.output_tensors[0].name
                                quant_min = Tensor(
                                    name=output_name + "_min",
                                    shape=[quant_info[input_name][3].size],
                                    data=np.array(quant_info[input_name][3].astype("float32")),
                                    dtype="fp32")
                                quant_max = Tensor(
                                    name=output_name + "_max",
                                    shape=[quant_info[input_name][4].size],
                                    data=np.array(quant_info[input_name][4].astype("float32")), 
                                    dtype="fp32")
                                # insert output min and max tensor
                                model.change_node_input_tensors(node.name, insert_offset + 4,
                                                                quant_min, 'insert')
                                model.change_node_input_tensors(node.name, insert_offset + 5,
                                                                quant_max, 'insert')
                                util.insert_quant_info(node.name, [])
                        idx += 1
                node_idx += 1

                        # remove duplicate tensors
            for node in model.nodes:
                if node.name in quant_info:
                    input_tensor_set = set()
                    remove_tensors_list = []
                    for idx, input_tensor in enumerate(node.input_tensors):
                        input_name = input_tensor.name
                        sz = len(input_tensor_set)
                        input_tensor_set.add(input_name)
                        new_sz = len(input_tensor_set)
                        if new_sz == sz:
                            remove_tensors_list.append(idx)
                    for remove_idx in remove_tensors_list:
                        model.change_node_input_tensors(node.name, remove_tensors_list[0], None,
                                                        'remove')

        elif model.framework_modeling_config['framework'] == 'onnxruntime':
            for node in model.nodes:
                if node.op_type in EXECUTOR_TYPE and \
                (EXECUTOR_TYPE[node.op_type] == "InnerProduct" or \
                    EXECUTOR_TYPE[node.op_type] == "Matmul"):
                    for idx, input_tensor in enumerate(node.input_tensors):
                        input_name = input_tensor.name
                        insert_offset = 1 if len(node.input_tensors) % 2 == 0 else 0
                        insert_offset = insert_offset - 2 if "append_op" not in node.attr and \
                                                        insert_offset == 1 else insert_offset
                        if input_name in quant_info and idx < 3:
                            quant_min = Tensor(
                                name=input_name + "_min",
                                shape=[quant_info[input_name][0].size],
                                data=np.array(quant_info[input_name][0].astype("float32")),
                                dtype="fp32")
                            quant_max = Tensor(
                                name=input_name + "_max",
                                shape=[quant_info[input_name][1].size],
                                data=np.array(quant_info[input_name][1].astype("float32")),
                                dtype="fp32")

                            if "insert" in quant_info[input_name][2]:
                                quant_dtype = "u8" if "u8" in quant_info[input_name][2] else "s8"
                                quant_dtype = "s8" if EXECUTOR_TYPE[
                                    node.op_type] == "Matmul" else quant_dtype
                                quant_output = Tensor(name=input_name + "_quant",
                                                    source_op=[node.name + "_quant_" + str(idx)],
                                                    dest_op=[node.name],
                                                    dtype=quant_dtype)
                                quantize_op = util.construct_node(
                                    node_name=node.name + "_quant_" + str(idx),
                                    op_type='Quantize',
                                    input_tensors=[input_tensor, quant_min, quant_max],
                                    output_tensors=[quant_output],
                                    attr=OrderedDict({'output_dtype': quant_dtype}))
                                node.input_tensors[idx] = quant_output
                                insert_idx = model.get_node_id(node.name)
                                model.insert_nodes(insert_idx, [quantize_op])
                                # insert src0/src1 min and max tensor
                                model.change_node_input_tensors(node.name, insert_offset + 2 * idx + 3,
                                                                quant_min, 'insert')
                                model.change_node_input_tensors(node.name, insert_offset + 2 * idx + 4,
                                                                quant_max, 'insert')
                            if "weight" in quant_info[input_name][2]:
                                # insert weight min and max tensor
                                model.change_node_input_tensors(node.name, insert_offset + 2 * idx + 3,
                                                                quant_min, 'insert')
                                model.change_node_input_tensors(node.name, insert_offset + 2 * idx + 4,
                                                                quant_max, 'insert')
                            if "output" in quant_info[input_name][2]:
                                output_name = node.output_tensors[0].name
                                quant_min = Tensor(
                                    name=output_name + "_min",
                                    shape=[quant_info[input_name][3].size],
                                    data=np.array(quant_info[input_name][3].astype("float32")),
                                    dtype="fp32")
                                quant_max = Tensor(
                                    name=output_name + "_max",
                                    shape=[quant_info[input_name][4].size],
                                    data=np.array(quant_info[input_name][4].astype("float32")), 
                                    dtype="fp32")
                                # insert output min and max tensor
                                model.change_node_input_tensors(node.name, insert_offset + 7,
                                                                quant_min, 'insert')
                                model.change_node_input_tensors(node.name, insert_offset + 8,
                                                                quant_max, 'insert')
                                util.insert_quant_info(node.name, [])

            # remove fall back quant nodes
            remove_list=[]
            for node in model.nodes:
                if node.op_type in EXECUTOR_TYPE and \
                (EXECUTOR_TYPE[node.op_type] == "InnerProduct" or \
                    EXECUTOR_TYPE[node.op_type] == "Matmul"):
                    src0_dtype = node.input_tensors[0].dtype == "u8" or \
                                node.input_tensors[0].dtype == "s8"
                    src1_dtype = node.input_tensors[1].dtype == "u8" or \
                                node.input_tensors[1].dtype == "s8"
                    if src0_dtype ^ src1_dtype:
                        src0_source_op = node.input_tensors[0].source_op
                        src1_source_op = node.input_tensors[1].source_op
                        if src0_dtype:
                            remove_list.append(src0_source_op[0])
                            node.input_tensors[0] = \
                            model.get_node_by_name(src0_source_op[0]).input_tensors[0]
                            node.input_tensors[0].dest_op = [node.name]
                        else: 
                            remove_list.append(src1_source_op[0])
                            node.input_tensors[1] = \
                            model.get_node_by_name(src1_source_op[0]).input_tensors[0]
                            node.input_tensors[1].dest_op = [node.name]
                        remove_tensors_list = []
                        for idx, input_name in enumerate(node.input_tensors):
                            if "_min" in input_name.name or "_max" in input_name.name:
                                remove_tensors_list.append(idx)
                        for remove_idx in remove_tensors_list:
                            model.change_node_input_tensors(node.name, remove_tensors_list[0], None,
                                                        'remove')
            model.remove_nodes(remove_list)

            # remove duplicate quant nodes and duplicate tensors
            remove_duplicate_set = set()
            quant_node_dict = {}
            duplicate_list=[]
            for node in model.nodes:
                sz = len(remove_duplicate_set)
                remove_duplicate_set.add(node.output_tensors[0].name)
                new_sz = len(remove_duplicate_set)
                if new_sz == sz:
                    duplicate_list.append(node.name)
                    remain_node_name = quant_node_dict[node.output_tensors[0].name]
                    dup_node = model.get_node_by_name(node.output_tensors[0].dest_op[0])
                    dup_node.input_tensors[0].source_op = [remain_node_name]
                else:
                    quant_node_dict[node.output_tensors[0].name] = node.name
            model.remove_nodes(duplicate_list)

            for node in model.nodes:
                if node.name in quant_info:
                    input_tensor_set = set()
                    remove_tensors_list = []
                    for idx, input_tensor in enumerate(node.input_tensors):
                        input_name = input_tensor.name
                        sz = len(input_tensor_set)
                        input_tensor_set.add(input_name)
                        new_sz = len(input_tensor_set)
                        if new_sz == sz:
                            remove_tensors_list.append(idx)
                    for remove_idx in remove_tensors_list:
                        model.change_node_input_tensors(node.name, remove_tensors_list[0], None,
                                                        'remove')

        # Bias compensation for inner product fp32 bias to int32
        unique_bias = []
        for node in model.nodes:
            if node.op_type in EXECUTOR_TYPE and EXECUTOR_TYPE[
                    node.op_type] == "InnerProduct" and len(node.input_tensors) > 4:
                bias_name = node.input_tensors[2].name
                if bias_name in unique_bias:
                    node.input_tensors[2] = copy.deepcopy(node.input_tensors[2])
                    bias_name = node.name + bias_name
                    node.input_tensors[2].name = bias_name
                unique_bias.append(bias_name)
                bias_fp32 = node.input_tensors[2].data
                weight_s8 = node.input_tensors[1].data
                offset = 0
                if 'append_op' in node.attr and node.attr['append_op'] in ['binary_add', 'sum']:
                    offset = 1
                input_data_min = node.input_tensors[offset + 3].data
                dtype = node.input_tensors[0].dtype
                input_scale, input_zero_point = get_scale_zp(node.input_tensors[offset + 3].data,
                                                            node.input_tensors[offset + 4].data,
                                                            dtype)
                weight_scale, weight_zero_point = get_scale_zp(node.input_tensors[offset + 5].data,
                                                            node.input_tensors[offset + 6].data,
                                                            's8')
                if dtype == "u8":
                    if "src1_perm" in node.attr and node.attr["src1_perm"] == '1,0':
                        bias_zero_point = input_scale * input_data_min * \
                            np.sum(weight_s8.astype(float), axis=0)
                    else:
                        bias_zero_point = input_scale * input_data_min * \
                            np.sum(weight_s8.astype(float), axis=-1)
                    bias_s32 = bias_fp32 * input_scale * weight_scale
                    bias_s32 = np.round(bias_s32 + bias_zero_point).astype(np.int32)
                    node.input_tensors[2].data = bias_s32
                    node.input_tensors[2].dtype = 's32'
                else:
                    bias_s32 = np.round(bias_fp32 * input_scale * weight_scale).astype(np.int32)
                    node.input_tensors[2].data = bias_s32
                    node.input_tensors[2].dtype = 's32'

        return model
