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

import sys
import numpy as np
import re
from collections import namedtuple, OrderedDict
from . import logger
from .ops.tensor import Tensor
from . import graph_utils as util

def get_children(node, input_name_to_nodes=None):
    """Get the node's output nodes in the graph."""
    if input_name_to_nodes is None:
        input_name_to_nodes = {}

    children = []
    for output in node.output:
        if output in input_name_to_nodes:
            for child in input_name_to_nodes[output]:
                children.append(child)
    return children

def get_node_children_names(model, node):
    """Get the node's output nodes' name in the graph.

    Args:
        model: ONNXModel
        node: NodeProto in onnx model

    Returns:
        outputs: names list
    """
    output_nodes = get_children(node)
    outputs = [node.name for node in output_nodes]
    return outputs


def get_initializer_children_names(model, initializer):
    """Get the initializer's output nodes' name in the graph.

    Args:
        model: ONNXModel
        initializer: initializer in onnx model

    Returns:
        outputs: names list
    """
    output_nodes = []
    for node in model.graph.node:
        for node_input in node.input:
            if node_input == initializer.name:
                output_nodes.append(node)
    outputs = [node.name for node in output_nodes]
    return outputs


def graph_node_names_details(model):
    """Parse the graph nodes ans get the graph_nodes_dict.

    Be used for Grpah class with cerating a new graph.
    The node_name is the key, node in value is for getting the Const
    tensor value and the input_tensor source op; output_names in value
    is the node ouput name list; outputs in value is for output_tensor dest op
    Args:
        model: ONNXModel
        
    Returns:
        node_names_details: the graph node info dict

    """
    node_details = namedtuple('node_details', ['node', 'outputs'])
    node_names_details = {}
    for initializer in model.graph.initializer:
        initializer_name = initializer.name
        each_node = node_details(node=initializer, outputs=[])
        if initializer_name not in node_names_details:
            each_node.outputs.extend(get_initializer_children_names(model, initializer))
            node_names_details[initializer_name] = each_node
    for node in model.graph.node:
        node_name = node.name
        output_names = node.output
        for index, output_name in enumerate(output_names):
            output_names[index] = util.names_from_input(output_name)[1]
        # onnx output has different name from node name
        for output_name in output_names:
            if output_name not in node_names_details:
                node_names_details[output_name] = node_name
        each_node = node_details(node=node, outputs=[])
        if node_name not in node_names_details:
            each_node.outputs.extend(get_node_children_names(model, node))
            node_names_details[node_name] = each_node
    for graph_input in model.graph.input:
        outputs = []
        node_name = graph_input.name
        for k, v in node_names_details.items():
            try:
                if node_name in v.node.input:
                    outputs.append(k)
            except BaseException:
                continue
        each_node = node_details(node=graph_input, outputs=outputs)
        # if node_name not in node_names_details:
        node_names_details[node_name] = each_node

    return node_names_details

def is_supported_onnx_graph(graph):
    """Check if the onnx graph supported."""
    for node in graph.node:
        if not is_supported_onnx_node(node.op_type):
            return False
    return True

def is_supported_onnx_node(node_name):
    """Check if the node type is supported."""
    supported_ops_type = ["Add", "Softmax", "Slice", "ReduceMean", "Reshape",
                        "Concat", "Gather", "QuantizeLinear", "Transpose", "MatMul",
                        "Sqrt", "Unsqueeze", "Shape", "Erf", "Pow", "DequantizeLinear",
                        "Cast", "Tanh", "Div", "Mul", "Sub", "Constant", "Relu", "Conv",
                        "Identity", "Split", "TopK"]
    if node_name in supported_ops_type:
        return True
    else:
        return False

def change_num_name(tensor_name):
    """For number string."""
    try:
        if str(int(tensor_name)) == tensor_name:
            tensor_name += '_tensor'
    except BaseException:
        pass
    return tensor_name


def bias_to_int32(bias_node, a_scale, b_scale):
    """Convert the int8 bias to int32 bias.

    Args:
        bias_node: bias_add in graph (from onnx framework)
        a_scale: matmul node input matrice a scale tensor
        b_scale: matmul node input matrice b scale tensor
        model: Grpah class

    Returns:
        int32 bias numpy array

    fp32_bias = (int8_bias - int8_bias_zero_point) * int8_bias_scale
    int32_bias = fp32_bias / (a_scale * b_scale)
    """
    dtype_list = [np.int8, np.uint8]
    input_tensors = bias_node.input_tensors
    if input_tensors[0].source_op == []:
        int8_bias = input_tensors[0].data
        int8_bias_scale = input_tensors[1].data
        int8_bias_zero_point = input_tensors[2].data
    else:
        int8_bias = input_tensors[3].data
        int8_bias_scale = input_tensors[4].data
        int8_bias_zero_point = input_tensors[5].data

    if int8_bias.dtype not in dtype_list:
        return
    int8_bias = int8_bias.astype(np.int32)
    fp32_bias = ((int8_bias - int8_bias_zero_point) * int8_bias_scale).astype(np.float32)
    a_scale = float(a_scale)
    b_scale = float(b_scale)
    int32_bias = np.round((fp32_bias / (a_scale * b_scale))).astype(np.int32)

    return int32_bias

def _bf16_tensor_to_array(tensor, base_dir: str = "") -> np.ndarray:
    """Converts a tensor def object to a numpy array.

    Inputs:
        tensor: a TensorProto object.
        base_dir: if external tensor exists, base_dir can help to find the path to it
    Returns:
        arr: the converted array.
    """
    onnx = util.LazyImport('onnx')
    from onnx import TensorProto
    from onnx.external_data_helper import load_external_data_for_tensor, uses_external_data
    from onnx.numpy_helper import convert_endian

    if tensor.HasField("segment"):
        raise ValueError("Currently not supporting loading segments.")
    if tensor.data_type == TensorProto.UNDEFINED:
        raise TypeError("The element type in the input tensor is not defined.")

    tensor_dtype = tensor.data_type
    np_dtype = np.dtype('float16')
    storage_np_dtype = np.dtype('uint16')
    storage_field = 'int32_data'
    dims = tensor.dims

    # Load raw data from external tensor if it exists
    if uses_external_data(tensor):
        load_external_data_for_tensor(tensor, base_dir)

    if tensor.HasField("raw_data"):
        # Raw_bytes support: using frombuffer.
        if sys.byteorder == 'big':
            # Convert endian from little to big
            convert_endian(tensor)

        return np.frombuffer(
            tensor.raw_data,
            dtype=np_dtype).reshape(dims)
    else:
        # float16/bfloat16 is stored as int32 (uint16 type); Need view to get the original value
        return (
            np.asarray(
                tensor.int32_data,
                dtype=np.uint16)
            .reshape(dims)
            .view(np.float16))


def onnx_extract_operator(node, framework_model, nodes_dict, engine_graph=None):
    """Decorate the operator in onnx.

    Args:
        node: NodeProto
        framework_model: ONNXModel
        nodes_dict: dict, return value from graph_node_names_details
        tf_dtypes: dict, for get the dtype string
        engine_graph: Engine Graph class

    Returns:
        op_type: node op type
        input_tensors: Tensor list, contains the node input tensors info
        output_tensors: Tensor list, contains the node output tensor info
    """
    onnx = util.LazyImport('onnx')
    from onnx import TensorProto
    from onnx.numpy_helper import to_array
    op_type = node.op_type
    input_tensors = []
    output_tensors = []

    """Input_tensors

    Note:
        each input_tensor has its own soure op, but same dest op
        so both have single string
    """
    input_names = []
    input_tensor_names = node.input
    for input_tensor_name in input_tensor_names:
        # The resize op could have the third input name but no second input.
        if input_tensor_name == '':
            continue
        origin_tensor_name, input_tensor_name = util.names_from_input(input_tensor_name)
        try:
            pre_node = nodes_dict[nodes_dict[input_tensor_name]].node
        except BaseException:
            # if origin_tensor_name in nodes_dict:
            pre_node = nodes_dict[origin_tensor_name].node
        
        data = None
        if pre_node in framework_model.graph.initializer:
            if pre_node.data_type == TensorProto.BFLOAT16:
                data = _bf16_tensor_to_array(pre_node)
            else:
                data = to_array(pre_node)
        else:
            if (pre_node not in framework_model.graph.input) and (pre_node.op_type == 'Constant'):
                data = to_array(pre_node.attribute[0].t)
        if isinstance(data, np.ndarray):
            if data.dtype == np.int64:
                data = data.astype(np.int32)
            dtype = util.get_data_dtype(data)
            shape = list(data.shape) if data.shape != () else [1]
            input_tensor = Tensor(name=input_tensor_name,
                                  source_op=[],
                                  dest_op=[node.name],
                                  shape=shape,
                                  data=data,
                                  dtype=dtype
                                  )
            input_tensors.append(input_tensor)

        else:
            input_tensor = Tensor(name=input_tensor_name,
                                  source_op=[pre_node.name],
                                  dest_op=[node.name],
                                  shape=None,
                                  data=None,
                                  dtype=None
                                  )
            input_tensors.append(input_tensor)
        input_names.append(node.name)

    """Output_tensors.
    
    Note:
        in onnx, NodeProto has the output attribute
    """
    output_tensor_names = node.output
    for output_tensor_name in output_tensor_names:
        output_tensor_name = util.names_from_input(output_tensor_name)[1]
        output_tensor = Tensor(name=output_tensor_name,
                               source_op=[node.name],
                               dest_op=nodes_dict[node.name].outputs,
                               shape=None,
                               data=None,
                               dtype=None
                               )

        output_tensors.append(output_tensor)

    return op_type, input_tensors, output_tensors


# for onnx cast dst type, see:
# https://github.com/microsoft/onnxruntime/blob/main/include/onnxruntime/
# core/framework/data_types.h#L265
ONNX_DTYPE_ID = {1: 'fp32',
                 2: 'uint8',
                 3: 'int8',
                 4: 'uint16',
                 5: 'int16',
                 6: 'int32',
                 7: 'int32',
                 8: 'string',
                 9: 'bool',
                 10: 'fp16',
                 11: 'double',
                 12: 'uint32',
                 13: 'uint64',
                 14: 'bf16'
                }
# key is the optimize level (higher means you can use more passes)
# value is the pass name
# for more passes, see: https://github.com/onnx/optimizer/blob/master/onnxoptimizer/pass_registry.h
ONNX_OPTIMIZER_PASS = {1: 'eliminate_identity',
                       2: 'eliminate_shape_gather',
                      }
