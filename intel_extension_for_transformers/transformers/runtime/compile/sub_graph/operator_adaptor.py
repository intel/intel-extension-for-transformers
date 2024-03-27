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

"""The OperatorAdaptor pattern."""

from .pattern import Pattern, pattern_registry
from collections import namedtuple, OrderedDict
from .subgraph_matcher import EXECUTOR_TYPE
from .. import graph_utils as util
from ..ops.tensor import Tensor
import numpy as np
import copy


@pattern_registry(pattern_type='OperatorAdaptor')
class OperatorAdaptor(Pattern):

    """The OperatorAdaptor pattern.

    Modify some operators after all fusion patterns. For example, sweep input tensors to make
    sure that Neural Engine backend op can receive its required inputs in different framework.
    """
    def __call__(self, model):
        """The __call__ function of this pattern class.

        Args:
            model (class): the Graph class
            sweep_nodes_info (dict): Supply the handling information for model to sweep inputs.
                                     The key (str) is the node name and the value (str) is
                                     the target position.
        """
        def _sweep_inputs(model, sweep_nodes_info):
            for name, target_pos in sweep_nodes_info.items():
                node_id = model.get_node_id(name)
                target_inputs = [model.nodes[node_id].input_tensors[j] for j in target_pos]
                model.nodes[node_id].input_tensors = target_inputs

        def _reshape_non_2d_src_before_inner_product(model, node_names):
            for name in node_names:
                node_id = model.get_node_id(name)
                ip_node = model.nodes[node_id]
                hidden_size = int(ip_node.input_tensors[1].data.shape[0])
                pre_node = model.get_node_by_name(ip_node.input_tensors[0].source_op[0])
                reshape_node_name = "reshape_2d_" + name
                pre_node.output_tensors[0].dest_op.append(reshape_node_name)
                input_tensors = [copy.deepcopy(pre_node.output_tensors[0])]
                output_tensors = [Tensor(name=reshape_node_name+":0",
                    source_op=[reshape_node_name], dest_op=[name])]
                ip_node.input_tensors[0] = output_tensors[0]
                reshape_node = util.construct_node(reshape_node_name, 'Reshape',
                    input_tensors=input_tensors, output_tensors=output_tensors, attr=OrderedDict(
                    {'dst_shape': '-1,' + str(hidden_size)}))
                model.insert_nodes(node_id, [reshape_node])

        def _remove_squeeze_after_inner_product(model, node_names):
            for name in node_names:
                node_id = model.get_node_id(name)
                squeeze_node = model.nodes[node_id]
                pre_node = model.get_node_by_name(squeeze_node.input_tensors[0].source_op[0])
                assert EXECUTOR_TYPE.get(pre_node.op_type, pre_node.op_type) == 'InnerProduct'
                pre_node.attr['squeeze_dims'] = squeeze_node.attr['axes']
                pre_node.output_tensors[0] = copy.deepcopy(squeeze_node.output_tensors[0])
                pre_node.output_tensors[0].source_op= [pre_node.name]
            model.remove_nodes(node_names)

        # k: op_type, v :target_pos
        sweep_op = {'Gather': [1,0]}
        sweep_nodes_info = {}
        ip_need_reshape_2d_names = []
        ip_squeeze_names = []
        for n in model.nodes:
            if n.op_type in sweep_op.keys():
                # skip the Gather op which has required inputs (0: idx, 1: weight)
                if n.op_type == 'Gather' and n.filling_method in ['extract_from_onnxruntime',
                                                                  'extract_from_torch']:
                    if isinstance(n.input_tensors[0].data, np.ndarray):
                        sweep_nodes_info[n.name] = sweep_op[n.op_type]
            elif n.op_type in ['MatMul', 'MatMulWithBias', 'MatMulWithBiasAdd']:
                if isinstance(n.input_tensors[1].data, np.ndarray):
                    pre_node = None
                    try:
                        pre_node = model.get_node_by_name(n.input_tensors[0].source_op[0])
                    except:
                        pre_node = None
                    if pre_node and pre_node.op_type == 'Transpose' and \
                       len(pre_node.attr['dst_perm']) > 3:
                        ip_need_reshape_2d_names.append(n.name)
            elif n.op_type in ['Squeeze']:
                pre_node = None
                try:
                    pre_node = model.get_node_by_name(n.input_tensors[0].source_op[0])
                except:
                    pre_node = None
                if pre_node and EXECUTOR_TYPE.get(pre_node.op_type, pre_node.op_type) == \
                   'InnerProduct' and n.attr:
                    ip_squeeze_names.append(n.name)
            else:
                continue
        if len(sweep_nodes_info) > 0:
            _sweep_inputs(model, sweep_nodes_info)
        if len(ip_need_reshape_2d_names) > 0:
            _reshape_non_2d_src_before_inner_product(model, ip_need_reshape_2d_names)
        if len(ip_squeeze_names) > 0:
            _remove_squeeze_after_inner_product(model, ip_squeeze_names)

        return model
