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

from .pattern import Pattern, pattern_registry
from collections import namedtuple, OrderedDict
from .. import graph_utils as util
from .subgraph_matcher import EXECUTOR_TYPE


@pattern_registry(pattern_type='CollectQDQInfo')
class CollectQDQInfo(Pattern):

    def __call__(self, model):

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
            ]
        }
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
                for idx, input_tensor in enumerate(matmul_node.input_tensors):
                    if input_tensor.name == dquant_output.name:
                        matmul_node.input_tensors[idx] = dquant_node.input_tensors[0]
            model.remove_nodes([pattern_nodes_name[0]])

        return model
