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

"""The StableDiffusion_InsertQuantNode Pattern."""

from .pattern import Pattern, pattern_registry
from collections import namedtuple, OrderedDict
from .. import graph_utils as util
from ..ops import Tensor
from .subgraph_matcher import EXECUTOR_TYPE
from .. import logger
import numpy as np


@pattern_registry(pattern_type='StableDiffusion_InsertQuantNode')
class StableDiffusion_InsertQuantNode(Pattern):
    """The StableDiffusion_InsertQuantNode pattern.

    Fuse the original sub-graph into the custom acceleration 'StableDiffusion_InsertQuantNode' graph.
    The search strategy is based on the following pattern mapping configs for different models.
    """

    def __call__(self, model):
        """The __call__ function of this pattern class."""
        quant_info = util.get_quant_info()
        if not quant_info:
            return model
        if model.framework_modeling_config['framework'] == 'torch':
            logger.error("Neural Engine does not support Stable Diffusion PyTorch Model now.")
            return model

        elif model.framework_modeling_config['framework'] == 'onnxruntime':
            for node in model.nodes:
                if node.op_type in EXECUTOR_TYPE and \
                (EXECUTOR_TYPE[node.op_type] in ['InnerProduct', 'Matmul', 'Convolution']):
                    for idx, input_tensor in enumerate(node.input_tensors):
                        input_name = input_tensor.name

                        insert_offset = 1 if len(node.input_tensors) % 2 == 0 else 0
                        insert_offset = insert_offset - 2 if "append_op" not in node.attr and \
                                                        insert_offset == 1 else insert_offset

                        if (EXECUTOR_TYPE[node.op_type] in ['InnerProduct']) \
                            and 'reshape' in node.attr and "append_op" not in node.attr:
                            insert_offset = insert_offset + 1
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

                                    #import pdb;pdb.set_trace()
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

                        elif input_name in quant_info and idx < 3:
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
                            # only for conv ?
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

        return model
