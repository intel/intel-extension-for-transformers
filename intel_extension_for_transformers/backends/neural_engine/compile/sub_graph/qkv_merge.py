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

"""The QKVMerge Pattern."""

from .pattern import Pattern, pattern_registry
from collections import namedtuple, OrderedDict
from .. import graph_utils as util
from ..ops import Tensor
import numpy as np


@pattern_registry(pattern_type="QKVMerge")
class QKVMerge(Pattern):
    """The QKVMerge pattern.

    Fuse the original sub-graph into the custom acceleration 'QKVMerge' graph.
    The search strategy is based on the following pattern mapping configs for different models.
    """
    def __call__(self, model):
        """The __call__ function of this pattern class."""
        def get_zero_ratio(matrix, block):
            sparse_ratio = -1
            if matrix.ndim == 2 and len(block) == 2:
                zero_block_count = 0
                block_row = int(matrix.shape[0] / block[0])
                block_col = int(matrix.shape[1] / block[1])
                for mr in range(block_row):
                    for mc in range(block_col):
                        is_zero_block = True
                        for br in range(block[0]):
                            for bc in range(block[1]):
                                if matrix[mr * block[0] + br][mc * block[1] +
                                                              bc] != 0:
                                    is_zero_block = False
                                    break
                            if not is_zero_block:
                                break
                        if is_zero_block == True:
                            zero_block_count += 1
                zero_ratio = float(zero_block_count) / (block_row * block_col)
            return zero_ratio

        pattern_mapping_config = {
            "QKVMerge": [{
                "patterns": {
                    "in": [
                        [
                            (0, "LayerNorm"),
                            (1, "Quantize"),
                            (2, "MatMulWithBias"),
                            (5, "Reshape"),
                            (10, "TransposeBatchMatMul"),
                        ],
                        [
                            (0, "LayerNorm"),
                            (1, "Quantize"),
                            (3, "MatMulWithBias"),
                            (6, "Reshape"),
                            (8, "TransposeBatchMatMul"),
                            (9, "Softmax"),
                            (10, "TransposeBatchMatMul"),
                        ],
                        [
                            (0, "LayerNorm"),
                            (1, "Quantize"),
                            (4, "MatMulWithBias"),
                            (7, "Reshape"),
                            (8, "TransposeBatchMatMul"),
                            (9, "Softmax"),
                            (10, "TransposeBatchMatMul"),
                        ],
                    ],
                },
                "search_mode": "op_type",
            }]
        }
        for i in range(len(pattern_mapping_config["QKVMerge"])):
            pattern = pattern_mapping_config["QKVMerge"][i]["patterns"]["in"]
            patterns_nodes_name = util.search_pattern(pattern, model)
            for j in range(len(patterns_nodes_name)):
                quantize = model.get_node_by_name(patterns_nodes_name[j][1])
                v_matmul = model.get_node_by_name(patterns_nodes_name[j][2])
                q_matmul = model.get_node_by_name(patterns_nodes_name[j][3])
                k_matmul = model.get_node_by_name(patterns_nodes_name[j][4])
                matmuls = []

                if ((v_matmul.attr.__contains__("output_dtype")
                        and q_matmul.attr.__contains__("output_dtype")
                        and k_matmul.attr.__contains__("output_dtype")
                        and q_matmul.attr["output_dtype"] == v_matmul.attr["output_dtype"] 
                        and q_matmul.attr["output_dtype"] == v_matmul.attr["output_dtype"])
                        or (v_matmul.attr.__contains__("output_dtype") == False
                        and q_matmul.attr.__contains__("output_dtype") == False
                        and k_matmul.attr.__contains__("output_dtype") == False)):
                    matmuls.append(v_matmul)
                    matmuls.append(q_matmul)
                    matmuls.append(k_matmul)
                else:
                    matmuls.append(q_matmul)
                    matmuls.append(k_matmul)

                zero_up_to_standard = True
                for k in range(len(matmuls)):
                    if get_zero_ratio(matmuls[k].input_tensors[1].data,
                                      [1, 4]) < 0.7:
                        zero_up_to_standard = False
                        break
                if zero_up_to_standard == True:
                    weight_tensor = []
                    bias_tensor = []
                    src_min_tensor = []
                    src_max_tensor = []
                    weight_min_tensor = []
                    weight_max_tensor = []
                    output_min_tensor = []
                    output_max_tensor = []

                    for k in range(len(matmuls)):
                        weight_tensor.append(matmuls[k].input_tensors[1])
                        bias_tensor.append(matmuls[k].input_tensors[2])
                        src_min_tensor.append(matmuls[k].input_tensors[3])
                        src_max_tensor.append(matmuls[k].input_tensors[4])
                        weight_min_tensor.append(matmuls[k].input_tensors[5])
                        weight_max_tensor.append(matmuls[k].input_tensors[6])
                        output_min_tensor.append(matmuls[k].input_tensors[7])
                        output_max_tensor.append(matmuls[k].input_tensors[8])

                    for k in range(len(matmuls)):
                        if k == 0:
                            weight_data = weight_tensor[k].data
                            bias_data = bias_tensor[k].data
                            src_min_data = src_min_tensor[k].data
                            src_max_data = src_max_tensor[k].data
                            weight_min_data = weight_min_tensor[k].data
                            weight_max_data = weight_max_tensor[k].data
                            output_min_data = output_min_tensor[k].data
                            output_max_data = output_max_tensor[k].data
                        else:
                            weight_data = np.hstack(
                                (weight_data, weight_tensor[k].data))
                            bias_data = np.hstack(
                                (bias_data, bias_tensor[k].data))
                            src_min_data = np.hstack(
                                (src_min_data, src_min_tensor[k].data))
                            src_max_data = np.hstack(
                                (src_max_data, src_max_tensor[k].data))
                            weight_min_data = np.hstack(
                                (weight_min_data, weight_min_tensor[k].data))
                            weight_max_data = np.hstack(
                                (weight_max_data, weight_max_tensor[k].data))
                            output_min_data = np.hstack(
                                (output_min_data, output_min_tensor[k].data))
                            output_max_data = np.hstack(
                                (output_max_data, output_max_tensor[k].data))

                    # create merge matmul weight and bias tensor
                    merge_matmul_weight_tensor = Tensor(
                        name="merge_matmul_weight_tensor" + str(j),
                        dest_op=["merge_matmul" + str(j)],
                        shape=[
                            weight_tensor[0].shape[0],
                            weight_tensor[0].shape[1] * len(matmuls),
                        ],
                        data=weight_data,
                        dtype=weight_tensor[0].dtype,
                    )

                    merge_matmul_bias_tensor = Tensor(
                        name="merge_matmul_bias_tensor" + str(j),
                        dest_op=["merge_matmul" + str(j)],
                        shape=[bias_tensor[0].shape[0] * len(matmuls)],
                        data=bias_data,
                        dtype=bias_tensor[0].dtype,
                    )

                    merge_matmul_src_min_tensor = Tensor(
                        name="merge_matmul_src_min_tensor" + str(j),
                        dest_op=["merge_matmul" + str(j)],
                        shape=[src_min_tensor[0].shape[0] * len(matmuls)],
                        data=src_min_data,
                        dtype=src_min_tensor[0].dtype,
                    )

                    merge_matmul_src_max_tensor = Tensor(
                        name="merge_matmul_src_max_tensor" + str(j),
                        dest_op=["merge_matmul" + str(j)],
                        shape=[src_max_tensor[0].shape[0] * len(matmuls)],
                        data=src_max_data,
                        dtype=src_max_tensor[0].dtype,
                    )

                    merge_matmul_weight_min_tensor = Tensor(
                        name="merge_matmul_weight_min_tensor" + str(j),
                        dest_op=["merge_matmul" + str(j)],
                        shape=[weight_min_tensor[0].shape[0] * len(matmuls)],
                        data=weight_min_data,
                        dtype=weight_min_tensor[0].dtype,
                    )

                    merge_matmul_weight_max_tensor = Tensor(
                        name="merge_matmul_weight_max_tensor" + str(j),
                        dest_op=["merge_matmul" + str(j)],
                        shape=[weight_max_tensor[0].shape[0] * len(matmuls)],
                        data=weight_max_data,
                        dtype=weight_max_tensor[0].dtype,
                    )

                    merge_matmul_output_min_tensor = Tensor(
                        name="merge_matmul_output_min_tensor" + str(j),
                        dest_op=["merge_matmul" + str(j)],
                        shape=[output_min_tensor[0].shape[0] * len(matmuls)],
                        data=output_min_data,
                        dtype=output_min_tensor[0].dtype,
                    )

                    merge_matmul_output_max_tensor = Tensor(
                        name="merge_matmul_output_max_tensor" + str(j),
                        dest_op=["merge_matmul" + str(j)],
                        shape=[output_max_tensor[0].shape[0] * len(matmuls)],
                        data=output_max_data,
                        dtype=output_max_tensor[0].dtype,
                    )

                    # create tensor between laynernorm and matmul
                    merge_matmul_out = Tensor(
                        name="merge_matmul_out" + str(j),
                        source_op=["merge_matmul" + str(j)],
                        dest_op=["Split" + str(j)],
                        dtype=matmuls[0].output_tensors[0].dtype,
                    )

                    # create merge matmul
                    quantize.output_tensors[0].dest_op = [
                        "merge_matmul" + str(j)
                    ]
                    merge_matmul_attr = OrderedDict({"src1_perm": "1,0"})
                    if (matmuls[0].attr.__contains__("output_dtype")):
                        merge_matmul_attr["output_dtype"] = matmuls[0].attr["output_dtype"]

                    merge_matmul = util.construct_node(
                        node_name="merge_matmul" + str(j),
                        op_type="MatMulWithBias",
                        # modify
                        input_tensors=[
                            quantize.output_tensors[0],
                            merge_matmul_weight_tensor,
                            merge_matmul_bias_tensor,
                            merge_matmul_src_min_tensor,
                            merge_matmul_src_max_tensor,
                            merge_matmul_weight_min_tensor,
                            merge_matmul_weight_max_tensor,
                            merge_matmul_output_min_tensor,
                            merge_matmul_output_max_tensor,
                        ],
                        output_tensors=[merge_matmul_out],
                        attr=merge_matmul_attr
                    )
                    # create split
                    split_attr = ""
                    for k in range(len(weight_tensor)):
                        split_attr += str(weight_tensor[k].shape[1])
                        if k < len(weight_tensor) - 1:
                            split_attr += ","
                    split_output_tensors = []
                    for k in range(len(matmuls)):
                        split_output_tensors.append(
                            matmuls[k].output_tensors[0])
                    split = util.construct_node(
                        node_name="Split" + str(j),
                        op_type="Split",
                        input_tensors=[merge_matmul_out],
                        # modify
                        output_tensors=split_output_tensors,
                        attr=OrderedDict({
                            "axis": 0,
                            "split": split_attr
                        }),
                    )
                    for k in range(len(split.output_tensors)):
                        split.output_tensors[k].source_op = "Split" + str(j)

                    # insert qkvmatmul and split
                    model.insert_nodes(
                        model.get_node_id(quantize.name) + 1, [split])
                    model.insert_nodes(model.get_node_id(split.name),
                                       [merge_matmul])
                    # remove Kmatmul Qmatmul Vmatmul
                    remove_name = []
                    for k in range(len(matmuls)):
                        remove_name.append(matmuls[k].name)
                    model.remove_nodes(remove_name)
        return model
