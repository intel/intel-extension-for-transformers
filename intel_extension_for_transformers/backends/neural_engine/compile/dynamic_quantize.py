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

from collections import OrderedDict
from copy import deepcopy
import numpy as np
from .graph_utils import construct_node, pattern_mapping
from .ops.tensor import Tensor
from .graph import Graph


def _insert_quantize(graph: Graph, per_token: bool):
    def quantize_src_tensor(input_tensor_idx: int, op_type: str):
        quantize_type = "s8" if node.op_type == "Matmul" else "u8"
        input_tensor = node.input_tensors[input_tensor_idx]
        node_name = node.input_tensors[input_tensor_idx].name + \
            "_quant" + str(input_tensor_idx)
        quant_output = Tensor(name=input_tensor.name + "_quant",
                              source_op=[node_name],
                              dest_op=[node.name],
                              dtype=quantize_type)
        quant_min = Tensor(name=input_tensor.name + "_min", source_op=[node_name], dest_op=[node.name], dtype='fp32')
        quant_scale = Tensor(name=input_tensor.name + "_scale",
                             source_op=[node_name],
                             dest_op=[node.name],
                             dtype='fp32')
        per_token_ = node.op_type in ["InnerProduct", "Matmul"] and per_token
        quantize_op = construct_node(node_name=node_name,
                                     op_type='Quantize',
                                     input_tensors=[input_tensor],
                                     output_tensors=[quant_output, quant_min, quant_scale],
                                     attr=OrderedDict({
                                         'output_dtype': "s8" if per_token_ else quantize_type,
                                         "per_token": per_token_
                                     }))
        try:
            quant_id = graph.get_node_id(node_name)
            quant_node = graph.nodes[quant_id]
            for output_tensor_idx in range(3):
                quant_node.output_tensors[output_tensor_idx].dest_op.append(node.name)
            graph.remove_nodes([quant_node.name])
            graph.insert_nodes(graph.get_node_id(node.name), [quant_node])
        except ValueError:
            graph.insert_nodes(graph.get_node_id(node.name), [quantize_op])
        node.input_tensors[input_tensor_idx] = quant_output
        node.input_tensors.extend([quant_min, quant_scale])

    def quantize_weight_tensor(weight_tensor: Tensor, op_type: str):
        #weight should be quantized to S8
        tensor_min = Tensor(name=node.input_tensors[1].name + "_min", source_op=[], dest_op=[node.name], dtype='fp32')
        tensor_scale = Tensor(name=node.input_tensors[1].name + "_scale",
                              source_op=[],
                              dest_op=[node.name],
                              dtype='fp32')
        weight_perm = node.attr.get('src1_perm', "")
        weight_data = weight_tensor.data
        assert weight_data is not None, "The graph to be dynamic quantized shold be load with weight"
        if op_type == "Convolution":
            axis = tuple(range(1, len(weight_data.shape)))
            tensor_min.data = weight_data.min(axis, keepdims=True).astype(np.float32)
            max_data = weight_data.max(axis, keepdims=True).astype(np.float32)
            tensor_scale.data = np.maximum(np.fabs(tensor_min.data), np.fabs(max_data))
            tensor_scale.data = np.where(tensor_scale.data == 0., 0., tensor_scale.data / 127.0)
            weight_tensor.data = np.round(weight_tensor.data / tensor_scale.data).astype(np.int8)
        else:
            if weight_perm == "0,1":
                tensor_min.data = np.min(weight_data, axis=-1).astype(np.float32)
                max_data = np.max(weight_data, axis=-1).astype(np.float32)
                # tensor_min.data = np.minimum(tensor_min.data, 0.)
                # tensor_max.data = np.maximum(tensor_max.data, 0.)
                tensor_scale.data = np.maximum(np.fabs(tensor_min.data), np.fabs(max_data))
                tensor_scale.data = np.where(tensor_scale.data == 0., 0., tensor_scale.data / 127.)
                weight_tensor.data = (np.round((weight_tensor.data).T / tensor_scale.data).astype(np.int8)).T
            else:
                tensor_min.data = np.min(weight_data, axis=0).astype(np.float32)
                max_data = np.max(weight_data, axis=0).astype(np.float32)
                # tensor_min.data = np.minimum(tensor_min.data, 0.)
                # tensor_max.data = np.maximum(tensor_max.data, 0.)
                tensor_scale.data = np.maximum(np.fabs(tensor_min.data), np.fabs(max_data))
                tensor_scale.data = np.where(tensor_scale.data == 0., 0., tensor_scale.data / 127.)
                weight_tensor.data = np.round(weight_tensor.data / tensor_scale.data).astype(np.int8)
        tensor_min.shape = list(tensor_min.data.shape)
        tensor_scale.shape = list(tensor_scale.data.shape)
        weight_tensor.dtype = 's8'
        weight_tensor.name = node.input_tensors[1].name + "_quant"
        node.input_tensors.extend([tensor_min, tensor_scale])

    for node in filter(lambda x: x.op_type in ["InnerProduct", "Matmul", "Convolution"], reversed(graph.nodes)):
        reshape_tensor = []
        if "reshape_dims" in node.attr:
            reshape_tensor = [node.input_tensors[-1]]
            node.input_tensors = node.input_tensors[:-1]
        quantize_src_tensor(0, node.op_type)
        if isinstance(node.input_tensors[1].data, np.ndarray):
            quantize_weight_tensor(node.input_tensors[1], node.op_type)
        else:
            quantize_src_tensor(1, "s8")
        node.input_tensors.extend(reshape_tensor)


def _fuse_quatize(graph: Graph):
    pattern = {
        "patterns": {
            'in': [[(0, "ANY"), (1, 'Quantize')]],
            'out': [[(0, 'ANY')]]
        },
        'search_mode': 'op_type',
        'node_names': {
            0: 0
        },
        'input_tensors': {
            0: [[{
                0: [0]
            }], [[0], 1]]
        },
        'output_tensors': {
            0: [[{
                1: [0]
            }, {
                1: [1]
            }, {
                1: [2]
            }], [[0, 1, 2], 3]]
        },
        'returns': [0, 1]
    }

    for any_op in ["InnerProduct", "Matmul", "Softmax", "Convolution"]:  # ,"LayerNorm"
        now_pattern = pattern.copy()
        now_pattern["patterns"]["in"][0][0] = (0, any_op)
        now_pattern["patterns"]["out"][0][0] = (0, any_op)
        graph, new_node_names, ret_old_nodes = pattern_mapping("fuse_quatize", now_pattern, graph)
        if len(new_node_names) != 0:
            for idx in range(len(new_node_names)):
                new_node_idx = graph.get_node_id(new_node_names[idx][0])
                graph.nodes[new_node_idx].input_tensors = ret_old_nodes[idx][0].input_tensors
                graph.nodes[new_node_idx].attr = ret_old_nodes[idx][0].attr
                graph.nodes[new_node_idx].attr.update(ret_old_nodes[idx][1].attr)
                if any_op == "Softmax":
                    graph.nodes[new_node_idx].attr["output_dtype"] = "u8"


def _fuse_mha(graph: Graph):
    pattern = {
        "patterns": {
            'in': [[(0, "Matmul"), (1, 'Softmax'), (2, "Matmul")]],
            'out': [[(0, 'MultiHeadAttention')]]
        },
        'search_mode': 'op_type',
        'node_names': {
            0: 0
        },
        'input_tensors': {
            0: [[{
                0: [0]
            }], [[0], 1]]
        },
        'output_tensors': {
            0: [[{
                2: [0]
            }, {
                2: [1]
            }, {
                2: [2]
            }], [[0, 1, 2], 3]]
        },
        'returns': [0, 2]
    }
    graph, new_node_names, ret_old_nodes = pattern_mapping("fuse_mha", pattern, graph)
    if len(new_node_names) != 0:
        for idx in range(len(new_node_names)):
            new_node_idx = graph.get_node_id(new_node_names[idx][0])
            # set input tensors
            qkmatmul = ret_old_nodes[idx][0]
            avmatmul = ret_old_nodes[idx][1]
            graph.nodes[new_node_idx].input_tensors = qkmatmul.input_tensors
            if len(qkmatmul.input_tensors) > 2 and qkmatmul.input_tensors[2].source_op and graph.get_node_by_name(
                    qkmatmul.input_tensors[2].source_op[0]).op_type == "PaddingSequence":
                # graph.get_node_by_name(qkmatmul.input_tensors[2].source_op[0]).op_type = "SequenceLength"
                graph.get_node_by_name(qkmatmul.input_tensors[2].source_op[0]).attr = {"dst_shape": "-1,-1"}
            graph.nodes[new_node_idx].input_tensors.insert(2, avmatmul.input_tensors[1])
            graph.nodes[new_node_idx].input_tensors.append(avmatmul.input_tensors[4])
            graph.nodes[new_node_idx].input_tensors.append(avmatmul.input_tensors[5])
            if "reshape_dims" in avmatmul.attr:
                graph.nodes[new_node_idx].input_tensors.append(avmatmul.input_tensors[-1])
            #set attr
            graph.nodes[new_node_idx].attr = avmatmul.attr
            graph.nodes[new_node_idx].attr["V_perm"] = avmatmul.attr.pop("src1_perm")
            graph.nodes[new_node_idx].attr["K_perm"] = qkmatmul.attr["src1_perm"]
            graph.nodes[new_node_idx].attr["Q_perm"] = qkmatmul.attr["src0_perm"]
            if "output_scale" in qkmatmul.attr:
                graph.nodes[new_node_idx].attr["output_scale"] = qkmatmul.attr["output_scale"]
            graph.nodes[new_node_idx].attr["per_token"] = True


def _remove_unused_input(graph):
    new_input_tensors = []
    for input_ternsor in graph.nodes[0].output_tensors:
        if input_ternsor.location == [] or input_ternsor.location is None:
            new_input_tensors.append(input_ternsor)
    graph.nodes[0].output_tensors = new_input_tensors


def _quantize2bf16(graph):
    for i in range(len(graph.nodes)):
        if graph.nodes[i].op_type in ["InnerProduct", "Matmul", "Convolution"]:
            attr = graph.nodes[i].attr
            if attr.get("output_dtype", "fp32") == "fp32":
                graph.nodes[i].attr["output_dtype"] = "bf16"
    pattern = {
        "patterns": {
            'in': [[(0, "GroupNorm"), (1, 'Sigmoid'), (2, "BinaryOp")]],
            'out': [[(0, 'GroupNorm')]]
        },
        'search_mode': 'op_type',
        'node_names': {
            0: 0
        },
        'input_tensors': {
            0: [[{
                0: [0]
            }, {
                0: [1]
            }, {
                0: [2]
            }], [[0, 1, 2], 3]]
        },
        'output_tensors': {
            0: [[{
                2: [0]
            }], [[0], 1]]
        },
        'returns': [0]
    }
    graph, new_node_names, ret_old_nodes = pattern_mapping("fuse_GN", pattern, graph)
    if len(new_node_names) != 0:
        for j in range(len(new_node_names)):
            attr = OrderedDict()
            attr = ret_old_nodes[j][0].attr
            attr['append_op'] = 'swish'
            attr['swish_beta'] = 1
            groupnorm_node = graph.get_node_by_name(new_node_names[j][0])
            groupnorm_node.attr = attr


def _dynamic_quantization(base_model, per_token=True):
    """
    base_model is a engine ir path or a graph(fp32 or bf16)
    """
    # load fp32model
    if isinstance(base_model, Graph):
        graph = deepcopy(base_model)
    else:
        graph = Graph()
        graph.graph_init(base_model + '/conf.yaml', base_model + '/model.bin', load_weight=True)

    _insert_quantize(graph, per_token)
    _fuse_quatize(graph)
    if per_token:
        _fuse_mha(graph)
    _remove_unused_input(graph)
    return graph
