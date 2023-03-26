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
"""The neural engine compile module."""

from collections import OrderedDict
from .loaders.loader import Loader
from .extractors.extractor import Extractor
from .sub_graph.subgraph_matcher import SubGraphMatcher
from .graph_utils import get_model_fwk_name, construct_node, pattern_mapping
from .ops.tensor import Tensor
from .graph import Graph
import numpy as np
import pdb

COMPILES = OrderedDict({
    'loader': Loader,
    'extractor': Extractor,
    'sub_graph': SubGraphMatcher,
})

_NEAURAL_ENGINE_AUTOCAST_TYPE = "native"


class autocast:

    def __init__(self, dtype: str) -> None:
        self.prev_dtype = _NEAURAL_ENGINE_AUTOCAST_TYPE
        self.dtype = dtype

    def __enter__(self) -> None:
        self.prev_dtype = _NEAURAL_ENGINE_AUTOCAST_TYPE
        _set_ne_autocast_dtype(self.dtype)

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        _set_ne_autocast_dtype(self.prev_dtype)


def _set_ne_autocast_dtype(dtype: str):
    _NEAURAL_ENGINE_AUTOCAST_TYPE = dtype


def _config_validation(config):
    """The validation of the input config."""
    if config == None:
        return None

    import yaml
    from schema import Schema

    with open(config, 'r') as conf_file:
        conf = yaml.safe_load(conf_file)

    conf_schema = Schema({
        'pattern_switch': Schema({str: bool}, error='You should provide correct fused_patterns.')
    })

    return conf_schema.validate(conf)


def start_pipeline(model, config=None):
    """The compile pipeline."""
    compile_list = []
    # initialize the compile
    for compile_type in COMPILES.keys():
        compile_ = COMPILES[compile_type]()
        compile_list.append(compile_)
    # convert the model
    for compile_ in compile_list:
        model = compile_(model, pattern_config=config)
    return model


def compile(model, config=None) -> Graph:
    """The compile interface.

    Firstly, use model loader to get the computation graph with corresponding framework.
    The graph contains nodes and edges, the node is op and the edge is the tensor.
    Then extract the ops in the graph and pack them to our form.
    Next exploit these above ops to consist sub-graph, which can see as "a new big op", like LayerNorm.

    Note:
        There may have different computation flow in one subgraph.
    Finally, convert them to .yaml file and .bin file for model configuration and inference.
    """
    from .graph import Graph
    if not isinstance(model, Graph):
        if get_model_fwk_name(model) == 'neural engine':
            graph = Graph()
            graph.graph_init(model + '/conf.yaml', model + '/model.bin')
            model = graph
        else:
            config = _config_validation(config)
            model = start_pipeline(model, config=config)
    if _NEAURAL_ENGINE_AUTOCAST_TYPE == "dynamic_int8":
        model = _dynamic_quantization(model)
    return model


def _insert_Q(graph: Graph):

    def quantize_src_tensor(input_tensor_idx: int, quantize_type: str):
        input_tensor = node.input_tensors[input_tensor_idx]
        node_name = node.input_tensors[input_tensor_idx].name + "_quant" + str(input_tensor_idx)
        quant_output = Tensor(name=input_tensor.name + "_quant",
                              source_op=[node_name],
                              dest_op=[node.name],
                              dtype=quantize_type)
        quant_min = Tensor(name=input_tensor.name + "_min",
                           source_op=[node_name],
                           dest_op=[node.name],
                           dtype='fp32')
        quant_scale = Tensor(name=input_tensor.name + "_scale",
                             source_op=[node_name],
                             dest_op=[node.name],
                             dtype='fp32')
        quantize_op = construct_node(node_name=node_name,
                                     op_type='Quantize',
                                     input_tensors=[input_tensor],
                                     output_tensors=[quant_output, quant_min, quant_scale],
                                     attr=OrderedDict({'output_dtype': quantize_type}))
        try:
            quant_id = graph.get_node_id(node_name)
            quant_node = graph.nodes[quant_id]
            for output_tensor_idx in range(3):
                quant_node.output_tensors[output_tensor_idx].dest_op.append(node.name)
        except ValueError:
            graph.insert_nodes(graph.get_node_id(node.name), [quantize_op])
        node.input_tensors[input_tensor_idx] = quant_output
        node.input_tensors.extend([quant_min, quant_scale])

    def quantize_weight_tensor(weight_tensor: Tensor):
        tensor_min = Tensor(name=node.input_tensors[1].name + "_min",
                            source_op=[],
                            dest_op=[node.name],
                            dtype='fp32')
        tensor_scale = Tensor(name=node.input_tensors[1].name + "_scale",
                              source_op=[],
                              dest_op=[node.name],
                              dtype='fp32')
        weight_perm = node.attr['src1_perm']
        weight_data = weight_tensor.data
        if weight_perm == "0,1":  #1,0?
            tensor_min.data = np.min(weight_data, axis=-1).astype(np.float32)
            max_data = np.max(weight_data, axis=-1).astype(np.float32)
            # tensor_min.data = np.minimum(tensor_min.data, 0.)
            # tensor_max.data = np.maximum(tensor_max.data, 0.)
            tensor_scale.data = np.maximum(np.fabs(tensor_min.data), np.fabs(max_data))
            tensor_scale.data = np.where(tensor_scale.data == 0., 0.,
                                         127. / tensor_scale.data)  #only support s8 now
            zero_point = 0.
            weight_tensor.data = \
                (np.round((weight_tensor.data).T * tensor_scale.data).astype(np.int8)).T
        else:
            tensor_min.data = np.min(weight_data, axis=0).astype(np.float32)
            max_data = np.max(weight_data, axis=0).astype(np.float32)
            # tensor_min.data = np.minimum(tensor_min.data, 0.)
            # tensor_max.data = np.maximum(tensor_max.data, 0.)
            tensor_scale.data = np.maximum(np.fabs(tensor_min.data), np.fabs(max_data))
            tensor_scale.data = np.where(tensor_scale.data == 0., 0.,
                                         127. / tensor_scale.data)  #only support s8 now
            zero_point = 0.
            weight_tensor.data = np.round(weight_tensor.data * tensor_scale.data).astype(np.int8)
        tensor_min.shape = list(tensor_min.data.shape)
        tensor_scale.shape = list(tensor_scale.data.shape)
        weight_tensor.dtype = 's8'
        weight_tensor.name = node.input_tensors[1].name + "_quant"
        node.input_tensors.extend([tensor_min, tensor_scale])

    for node in filter(lambda x: x.op_type in ["InnerProduct", "Matmul"], reversed(graph.nodes)):
        reshape_tensor = []
        if "reshape_dims" in node.attr:
            reshape_tensor = [node.input_tensors[-1]]
            node.input_tensors = node.input_tensors[:-1]
        quantize_src_tensor(0, "s8" if node.op_type == "Matmul" else "u8")
        if isinstance(node.input_tensors[1].data, np.ndarray):
            quantize_weight_tensor(node.input_tensors[1])
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

    for any_op in ["InnerProduct", "Matmul", "Softmax"]:  #,"LayerNorm"
        now_pattern = pattern.copy()
        now_pattern["patterns"]["in"][0][0] = (0, any_op)
        now_pattern["patterns"]["out"][0][0] = (0, any_op)
        graph, new_node_names, ret_old_nodes = pattern_mapping("fuse_quatize", now_pattern, graph)
        if len(new_node_names) != 0:
            for idx in range(len(new_node_names)):
                new_node_idx = graph.get_node_id(new_node_names[idx][0])
                graph.nodes[new_node_idx].input_tensors = ret_old_nodes[idx][0].input_tensors
                graph.nodes[new_node_idx].attr = ret_old_nodes[idx][0].attr
                graph.nodes[new_node_idx].attr[
                    "output_dtype"] = "u8" if any_op == "Softmax" else ret_old_nodes[idx][1].attr[
                        "output_dtype"]


def _remove_unused_input(graph):
    new_input_tensors = []
    for input_ternsor in graph.nodes[0].output_tensors:
        if input_ternsor.location == [] or input_ternsor.location is None:
            new_input_tensors.append(input_ternsor)
    graph.nodes[0].output_tensors = new_input_tensors


def _dynamic_quantization(fp32_model):
    """
    fp32_model is a engine ir path or a fp32 graph
    """
    # load fp32model
    if isinstance(fp32_model, Graph):
        graph = fp32_model
    else:
        graph = Graph()
        graph.graph_init(fp32_model + '/conf.yaml', fp32_model + '/model.bin')

    _insert_Q(graph)
    _fuse_quatize(graph)
    _remove_unused_input(graph)
    return graph
