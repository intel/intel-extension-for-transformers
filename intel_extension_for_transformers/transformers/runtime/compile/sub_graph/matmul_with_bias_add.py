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

"""The MatMulWithBiasAdd Pattern."""

from .pattern import Pattern, pattern_registry
from collections import namedtuple, OrderedDict
from .subgraph_matcher import EXECUTOR_TYPE
from .. import graph_utils as util
import copy


@pattern_registry(pattern_type='MatMulWithBiasAdd')
class MatMulWithBiasAdd(Pattern):
    """The MatMulWithBiasAdd pattern.

    Fuse the original sub-graph into the custom acceleration 'MatMulWithBiasAdd' graph.
    The search strategy is based on the following pattern mapping configs for different models.
    """
    def __call__(self, model):
        """The __call__ function of this pattern class."""
        match_ret = {}
        for node in model.nodes:
            if node.op_type in ['MatMulWithBias', 'InnerProduct'] and node.name not in match_ret:
                dst_ops = []
                for d in node.output_tensors[0].dest_op:
                    dst_ops.append(model.get_node_by_name(d))
                if node.output_tensors[0].name not in model.output_tensors_name and \
                   len(dst_ops) == 1 and EXECUTOR_TYPE.get(dst_ops[0].op_type,
                   dst_ops[0].op_type) == 'BinaryAdd':
                    add_s_ops = [model.get_node_by_name(t.source_op[0]) for t in
                                    dst_ops[0].input_tensors]
                    if [EXECUTOR_TYPE.get(o.op_type, o.op_type) for o in add_s_ops] == \
                       ['InnerProduct', 'InnerProduct']:
                        ids = [model.get_node_id(o.name) for o in add_s_ops]
                        match_ret[model.nodes[max(ids)].name] = dst_ops[0].name
                    else:
                        match_ret[node.name] = dst_ops[0].name

        for k, v in match_ret.items():
            mat_node_id = model.get_node_id(k)
            mat_node = copy.deepcopy(model.nodes[mat_node_id])
            a_node = model.get_node_by_name(v)
            a_node_name = a_node.name
            append_idx = 1 if mat_node.output_tensors[0].name == a_node.input_tensors[0].name \
                           else 0
            mat_node.input_tensors.append(copy.deepcopy(a_node.input_tensors[append_idx]))
            mat_node.output_tensors[0] = copy.deepcopy(a_node.output_tensors[0])
            mat_node.attr['append_op'] = 'sum'
            mat_node.name = a_node_name
            for t in mat_node.input_tensors:
                t.dest_op = [mat_node.name]
            mat_node.output_tensors[0].source_op = [mat_node.name]
            mat_node.op_type = 'MatMulWithBiasAdd'
            model.remove_nodes([k, v])
            model.insert_nodes(mat_node_id, [mat_node])

        return model
