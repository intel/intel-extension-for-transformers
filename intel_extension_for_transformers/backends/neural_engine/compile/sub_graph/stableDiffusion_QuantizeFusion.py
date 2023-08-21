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

"""The StableDiffusion_QuantizeFusion Pattern."""

from .pattern import Pattern, pattern_registry
from collections import namedtuple, OrderedDict
from .. import graph_utils as util
from ..ops import Tensor
from .. import logger
from .subgraph_matcher import EXECUTOR_TYPE
import copy


@pattern_registry(pattern_type='StableDiffusion_QuantizeFusion')
class StableDiffusion_QuantizeFusion(Pattern):
    """The StableDiffusion_QuantizeFusion pattern.

    Fuse the original sub-graph into the custom acceleration 'StableDiffusion_QuantizeFusion' graph.
    The search strategy is based on the following pattern mapping configs for different models.
    """
    def __call__(self, model):
        """The __call__ function of this pattern class."""
        def search_quant_fusion(node):
            if node.input_tensors[0].source_op == []:
                return (None, False)
            pre_node = model.get_node_by_name(node.input_tensors[0].source_op[0])
            if pre_node.input_tensors == []:
                return (None, False)
            is_from_quant = False
            if pre_node.input_tensors[0].source_op:
                try:
                    is_from_quant = True if model.get_node_by_name(pre_node.input_tensors[0].\
                                            source_op[0]).op_type == 'Quantize' else False
                except:
                    is_from_quant = False
            if pre_node.input_tensors[0].name in quant_info and len(pre_node.input_tensors) >= 6 \
               or (pre_node.op_type == "Softmax") \
               or (EXECUTOR_TYPE.get(pre_node.op_type, pre_node.op_type) in \
                   ["InnerProduct", "Matmul", "Convolution"] and (not quant_info or is_from_quant)):
                # GroupNorm does not support u8 data as the input.
                for dest_op in pre_node.output_tensors[0].dest_op:
                    dest_node = model.get_node_by_name(dest_op)
                    if dest_node.op_type == 'GroupNorm':
                        return (None, False)

                return (pre_node, True)
            elif pre_node.op_type == "Reshape":
                return search_quant_fusion(pre_node)
            else:
                return (None, False)

        quant_info = util.get_quant_info()
        if model.inquire_config_item("framework") == 'torch':
            if not quant_info:
                return model

        remove_node_name = []
        quant_node = None
        # fuse quant nodes to previous innerproduct or matmul output dtype to enhance perf
        for node in model.nodes:
            if node.op_type == "Quantize":
                dtype = node.attr['output_dtype'] 
                quant_node, can_fuse = search_quant_fusion(node)
                if can_fuse:
                    if dtype == 'u8' or dtype == 's8':
                        if quant_node.op_type == "Softmax":
                            def is_lat_model(model, p=None):
                                if p == None:
                                    p = [[(0, 'TopK'),(1, 'GatherElements')]]
                                match_result = util.search_pattern(p, model)
                                return len(match_result) != 0
                            if is_lat_model(model):
                                node.attr = OrderedDict({'output_dtype': "u8"})
                                continue
                            else:
                                model.change_node_input_tensors(quant_node.name, 1, node.input_tensors[1],
                                                                'insert')
                                model.change_node_input_tensors(quant_node.name, 2, node.input_tensors[2],
                                                                'insert')
                                quant_node.attr['output_dtype'] = "u8"
                        else:
                            if model.inquire_config_item("framework") == 'torch':
                                t_len = len(quant_node.input_tensors)
                                model.change_node_input_tensors(quant_node.name, t_len,
                                                                node.input_tensors[1], 'insert')
                                model.change_node_input_tensors(quant_node.name, t_len + 1,
                                                                node.input_tensors[2], 'insert')
                            else:
                                model.change_node_input_tensors(quant_node.name, -2,
                                                                node.input_tensors[1], 'modify')
                                model.change_node_input_tensors(quant_node.name, -1,
                                                                node.input_tensors[2], 'modify')
                            quant_node.attr['output_dtype'] = node.attr['output_dtype']
                    elif dtype == 'bf16':
                        quant_node.attr['output_dtype'] = dtype

                    quant_node_dest_op = []
                    for n in model.nodes[0:]:
                        if n.name == node.name:
                            continue
                        else:
                            for idx, input_tensor in enumerate(n.input_tensors):
                                if node.output_tensors[0].name == input_tensor.name:
                                    tensor_keeped = copy.deepcopy(node.input_tensors[0])
                                    tensor_keeped.source_op = [quant_node.name]
                                    tensor_keeped.dest_op = [n.name]
                                    n.input_tensors[idx] = tensor_keeped
                                    quant_node_dest_op.append(n.name)
                                    continue
                    if quant_node_dest_op:
                        quant_node.output_tensors[0].dest_op = quant_node_dest_op
                    remove_node_name.append(node.name)

        if (len(remove_node_name) != 0):
            logger.info('StableDiffusion_QuantizeFusion mathched...')
            logger.debug('StableDiffusion_QuantizeFusion = {}'.format(remove_node_name))
        model.remove_nodes(remove_node_name)

        return model
    