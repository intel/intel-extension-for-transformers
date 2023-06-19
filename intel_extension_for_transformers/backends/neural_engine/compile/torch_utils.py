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

"""The neural engine onnx utils."""

import numpy as np
from .ops.tensor import Tensor
from . import graph_utils as util
import copy

node_names = {}

def get_node_name(node):
    if node in node_names.keys():
        return node_names[node]
    else:
        name = node.kind() + "_" + str(len(node_names.keys()))
        node_names[node] = name
        return name

op_maps = {'aten::softmax': 'Softmax', 'prim::Constant': 'Constant', 'prim::ListConstruct': 'ListConstruct',
           'aten::linear': 'InnerProduct', 'aten::slice': 'Slice', 'aten::unsqueeze': 'Unsqueeze',
           'aten::rsqrt': 'Rsqrt', 'aten::masked_fill' : 'ConstantOfShape', 'aten::mean' : 'ReduceMean',
           'aten::embedding': 'Gather', 'aten::gather': 'Gather', 'aten::where': 'Where', 'aten::matmul': 'Matmul',
           'aten::reshape': 'Reshape', 'aten::gelu': 'Gelu', 'aten::max' : 'Max', 'aten::cos': 'Cos',
           'aten::layer_norm': 'LayerNorm', 'aten::size': 'Shape', 'aten::view': 'View',
           'aten::permute': 'Reorder', 'aten::transpose': 'Reorder', 'aten::expand' : 'Expand',
           'aten::select': 'Slice', 'aten::tanh': 'Tanh', 'aten::arange': 'Arange',
           'aten::reciprocal': 'Reciprocal', 'aten::einsum': 'Einsum', 'aten::sin': 'Sin',
           'aten::neg': 'Neg', 'aten::stack': 'Stack', 'aten::flatten': 'Flatten', 'aten::cat': 'Concat',
           'aten::rsub': 'Rsub', 'aten::mul': 'Mul', 'aten::add': 'Add', 'aten::add_': 'Add',
           'aten::sub': 'Sub', 'aten::gt': 'Greater', 'aten::lt': 'Less', 'aten::eq': 'Equal',
           'aten::quantize_per_tensor': 'Quantize', 'aten::dequantize': 'Dequantize',
           'prim::TupleUnpack': 'TupleUnpack','prim::TupleConstruct': 'TupleConstruct',
           'aten::floor_divide': 'Div', 'aten::pow': 'Pow',
           'aten::full': 'Full', 'aten::zeros': 'Zeros', 'aten::repeat': 'Repeat',
           'aten::silu': 'Swish', 'prim::ListUnpack': 'ListUnpack', 'aten::ne': 'NotEqual',
           'aten::div': 'Div', 'prim::padding_sequence' : 'PaddingSequence',
           'prim::slice_position_ids': 'SlicePositionIds', 'aten::squeeze': 'Squeeze',
           'aten::baddbmm': 'Baddbmm', 'prim::mybaddbmm': 'Baddbmm'}


def torch_extract_operator(node, model, nodes_dict, engine_graph=None):
    """Decorate the operator in Torch.

    Args:
        node: NodeProto
        model: Torchscript
        nodes_dict: dict, return value from graph_node_names_details
        tf_dtypes: dict, for get the dtype string

    Returns:
        op_type: node op type
        input_tensors: Tensor list, contains the node input tensors info
        output_tensors: Tensor list, contains the node output tensor info
    """
    op_type = node.kind()
    input_tensors = []
    output_tensors = []

    for i in range(node.inputsSize()):
        in_val = node.inputsAt(i)
        input_tensor_name = in_val.debugName()
        pre_node = in_val.node()
        source_ops = []
        source_ops.append(get_node_name(pre_node))
        if input_tensor_name in nodes_dict:
            input_tensor = nodes_dict[input_tensor_name]
            # input_tensor = copy.deepcopy(nodes_dict[input_tensor_name])
            # input_tensor.source_ops=source_ops
            # input_tensor.dest_op=[get_node_name(node)]
        else:
            input_tensor = Tensor(name=input_tensor_name,
                source_op=source_ops,
                dest_op=[get_node_name(node)],
                shape=None,
                data=None,
                dtype=None
                )
        input_tensors.append(input_tensor)

    for i in range(node.outputsSize()):
        out_val = node.outputsAt(i)
        output_tensor_name = out_val.debugName()
        dest_ops = []
        for val_user in out_val.uses():
            next_node = val_user.user
            if next_node.kind() != 'prim::Return':
                dest_ops.append(get_node_name(next_node))
        output_tensor = Tensor(name=output_tensor_name,
                source_op=[get_node_name(node)],
                dest_op=dest_ops,
                shape=None,
                data=None,
                dtype=None
                )
        output_tensors.append(output_tensor)

    return op_maps.get(op_type, op_type), input_tensors, output_tensors

