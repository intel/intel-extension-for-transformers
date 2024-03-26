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

"""The RemoveUnusedOperator pattern."""

from .pattern import Pattern, pattern_registry
from .subgraph_matcher import EXECUTOR_TYPE


@pattern_registry(pattern_type='RemoveUnusedOperator')
class RemoveUnusedOperator(Pattern):

    """The RemoveUnusedOperator pattern.

    Remove some operators which are not be used after all fusion patterns.
    """
    def __call__(self, model):
        """The __call__ function of this pattern class.

        Args:
            model (class): the Graph class
        """
        match_ret = []
        for node in model.nodes:
            if EXECUTOR_TYPE.get(node.op_type, node.op_type) in ['Shape'] and \
               node.name not in match_ret:
                if len(node.output_tensors[0].dest_op) == 0:
                    match_ret.append(node.name)
                else:
                    ot_name = node.output_tensors[0].name
                    useless = True
                    for i in range(model.get_node_id(node.name) + 1, len(model.nodes)):
                        if ot_name in [t.name for t in model.nodes[i].input_tensors]:
                            useless = False
                            break
                    if useless:
                        match_ret.append(node.name)

        model.remove_nodes(match_ret)

        return model
