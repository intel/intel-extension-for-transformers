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

"""The StableDiffusion_CollectQuantInfo Pattern."""

from .pattern import Pattern, pattern_registry
from collections import namedtuple, OrderedDict
from .. import graph_utils as util
from .subgraph_matcher import EXECUTOR_TYPE
from .. import logger
import numpy as np
import copy

@pattern_registry(pattern_type='StableDiffusion_CollectQuantInfo')
class StableDiffusion_CollectQuantInfo(Pattern):
    """The StableDiffusion_CollectQuantInfo pattern.

    Fuse the original sub-graph into the custom acceleration 'StableDiffusion_CollectQuantInfo' graph.
    The search strategy is based on the following pattern mapping configs for different models.
    """

    def __call__(self, model):
        """The __call__ function of this pattern class."""
        pattern_mapping_config = {
            'CollectQDQInfo': [
                {
                    'patterns': {
                        'in': [[(0, ['QuantizeLinear', 'Quantize']), (1, 'Cast'),
                                (2, ['DequantizeLinear'])]],
                    },
                },
                {
                    'patterns': {
                        'in': [[(0, 'ConstantOfShape'), (1, 'Cast'), (2, 'DequantizeLinear')]],
                    },
                },
                {
                    'patterns': {
                        'in': [[(0, 'DequantizeLinear')]],
                    },
                },
            ],
        }

        def get_min_max_from_onnx(scale, zp, dtype='s8'):
            max_range = 127 if 's8' in dtype else 255
            quant_max = (max_range - zp) * scale
            quant_min = quant_max - 255 * scale
            return quant_min, quant_max

        def CollectQDQInfo(model):
            # Collect the activation quant info
            pattern = pattern_mapping_config['CollectQDQInfo'][0]['patterns']['in']
            patterns_nodes_name = util.search_pattern(pattern, model)
            util.quant_info_init()
            new_dict = {}
            rm_node_list = []
            for pattern_nodes_name in patterns_nodes_name:
                quant_node = model.get_node_by_name(pattern_nodes_name[0])
                dquant_node = model.get_node_by_name(pattern_nodes_name[2])
                dquant_output = dquant_node.output_tensors[0]
                dtype = "s8" if dquant_node.input_tensors[2].data.dtype == 'int8' else "u8"
                quant_min, quant_max = get_min_max_from_onnx(dquant_node.input_tensors[1].data,
                                                             dquant_node.input_tensors[2].data,
                                                             dtype)
                dtype = dtype + "_insert"
                util.insert_quant_info(quant_node.input_tensors[0].name,
                                       [quant_min, quant_max, dtype])
                util.insert_quant_info(quant_node.output_tensors[0].name,
                                       [quant_min, quant_max, dtype])
                util.insert_quant_info(dquant_node.input_tensors[0].name,
                                       [quant_min, quant_max, dtype])
                util.insert_quant_info(dquant_node.output_tensors[0].name,
                                       [quant_min, quant_max, dtype])
                for dst_op in dquant_output.dest_op:
                    dst_node = model.get_node_by_name(dst_op)
                    for idx, input_tensor in enumerate(dst_node.input_tensors):
                        if input_tensor.name == dquant_output.name:
                            for pre_quant_name in quant_node.input_tensors[0].source_op:
                                pre_quant_node = model.get_node_by_name(pre_quant_name)
                                if pre_quant_node.op_type in EXECUTOR_TYPE and \
                                    (EXECUTOR_TYPE[pre_quant_node.op_type] in ['InnerProduct',
                                                                    'Matmul', 'Convolution']):
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
                                if len(pre_quant_node.output_tensors) == 1:
                                    dst_node.input_tensors[idx] = copy.deepcopy(pre_quant_node.output_tensors[0])
                                else:
                                    for idx, it in enumerate(pre_quant_node.output_tensors):
                                        if it.name == quant_node.input_tensors[0].name:
                                            pre_quant_node.output_tensors[idx].dest_op.append(dst_node.name)
                                            dst_node.input_tensors[0] = \
                                                copy.deepcopy(pre_quant_node.output_tensors[idx])

                rm_node_list.extend(pattern_nodes_name[:-1])
            model.remove_nodes(rm_node_list)

            # Collect the weight quant info
            patterns_nodes_name = []
            for i in [1, 2]:
                pattern = pattern_mapping_config['CollectQDQInfo'][i]['patterns']['in']
                patterns_nodes_name.extend(util.search_pattern(pattern, model))
            rm_node_list = []
            for pattern_nodes_name in patterns_nodes_name:
                if len(pattern_nodes_name[:-1]) == 3:
                    dquant_node = model.get_node_by_name(pattern_nodes_name[2])
                    dquant_output = dquant_node.output_tensors[0]
                    dquant_node.input_tensors[2].data = copy.deepcopy(
                        model.get_node_by_name(pattern_nodes_name[0]).input_tensors[0].data)
                    cast_dtype = model.get_node_by_name(pattern_nodes_name[1]).attr['DstT']
                    try:
                        dquant_node.input_tensors[2].data = \
                            dquant_node.input_tensors[2].data.astype(cast_dtype)
                    except:
                        logger.error('Can not cast dtype {}'.format(cast_dtype))
                        return model
                elif len(pattern_nodes_name[:-1]) == 1:
                    dquant_node = model.get_node_by_name(pattern_nodes_name[0])
                    dquant_output = dquant_node.output_tensors[0]
                else:
                    logger.error('Invalid pattern')
                    return model
                is_bias = False
                const_dtype = dquant_node.input_tensors[2].data.dtype
                if const_dtype == "int32":
                    is_bias = True
                    dtype = "int32"
                elif const_dtype == "int8":
                    dtype = "s8"
                else:
                    dtype = "u8"
                if not is_bias:
                    quant_min, quant_max = get_min_max_from_onnx(dquant_node.input_tensors[1].data,
                                                                dquant_node.input_tensors[2].data,
                                                                dtype)
                    dtype = dtype + "_weight"
                    util.insert_quant_info(dquant_node.input_tensors[0].name,
                                           [quant_min, quant_max, dtype])
                    util.insert_quant_info(dquant_node.output_tensors[0].name,
                                           [quant_min, quant_max, dtype])
                    # transpose weight
                    for dst_op in dquant_output.dest_op:
                        down_node = model.get_node_by_name(dst_op)
                        if EXECUTOR_TYPE.get(down_node.op_type, down_node.op_type) == "Reorder":
                            assert len(down_node.output_tensors[0].dest_op) == 1
                            assert down_node.attr.get('dst_perm', None) == '1,0'
                            dquant_node.input_tensors[0].data = np.transpose(
                                dquant_node.input_tensors[0].data, (1,0))
                            dquant_node.input_tensors[0].shape = \
                                list(dquant_node.input_tensors[0].data.shape)
                            rm_node_list.append(down_node.name)
                            dquant_node.output_tensors[0] = copy.deepcopy(
                                down_node.output_tensors[0])
                            dquant_node.output_tensors[0].source_op = [dquant_node.name]
                            util.insert_quant_info(down_node.output_tensors[0].name,
                                                    [quant_min, quant_max, dtype])
                dquant_output = dquant_node.output_tensors[0]
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
                            matmul_node.input_tensors[idx] = copy.deepcopy(
                                                             dquant_node.input_tensors[0])
                rm_node_list.extend(pattern_nodes_name[:-1])
            model.remove_nodes(rm_node_list)

        def UpdateQuantInfo(model):
            update_op_type = ['Reorder', 'Reshape', 'Cast']
            def _update_stream(start_tensor, mode = 'upstream'):
                for name in start_tensor.source_op:
                    n = None
                    try:
                        n = model.get_node_by_name(name)
                    except:
                        n = None
                    if n and EXECUTOR_TYPE.get(n.op_type, n.op_type) in update_op_type:
                        int8_info = util.get_quant_info()[start_tensor.name]
                        if mode == 'upstream':
                            util.insert_quant_info(n.input_tensors[0].name, int8_info)
                            _update_stream(n.input_tensors[0], 'upstream')
                        else:
                            util.insert_quant_info(n.output_tensors[0].name, int8_info)
                            _update_stream(n.output_tensors[0], 'downstream')
                    else:
                        return

            for node in model.nodes:
                if EXECUTOR_TYPE.get(node.op_type, node.op_type) in ['InnerProduct', 'Convolution']:
                    # check input activation
                    int8_info = util.get_quant_info().get(node.input_tensors[0].name, None)
                    if not int8_info and node.input_tensors[1].data.dtype != "float32":
                        logger.warning("node {} 's input activation {} has no int8 min-max info".
                                    format(node.name, node.input_tensors[0].name))
                    else:
                        _update_stream(node.input_tensors[0], mode='upstream')
                    # check output activation
                    int8_info = util.get_quant_info().get(node.output_tensors[0].name, None)
                    if not int8_info and node.attr.get('output_dtype', 'fp32') != 'fp32':
                        logger.warning("node {} 's output activation {} has no int8 min-max info".
                                    format(node.name, node.output_tensors[0].name))
                    else:
                        _update_stream(node.output_tensors[0], mode='downstream')

        if model.framework_modeling_config['framework'] == 'torch':
            logger.error("Neural Engine does not support Stable Diffusion PyTorch Model now.")
            return model

        # if ONNX QDQ model, CollectQDQInfo and UpdateQuantInfo
        # if ONNX QLinear model, raise unimplemented error
        is_qlinear_graph = False
        for node in model.nodes:
            if node.op_type == "QLinearMatMul":
                is_qlinear_graph = True
                break
        if not is_qlinear_graph:
            CollectQDQInfo(model)
            if util.get_quant_info():
                UpdateQuantInfo(model)
        else:
            logger.error("QLinearMatMul is unimplemented.")

        return model
