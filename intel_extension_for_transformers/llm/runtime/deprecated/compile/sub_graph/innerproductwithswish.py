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
"""The InnerproductWithSwish Pattern."""

import copy
from collections import OrderedDict

from .. import graph_utils as util
from .pattern import Pattern, pattern_registry


@pattern_registry(pattern_type="InnerproductWithSwish")
class InnerproductWithSwish(Pattern):
    """The InnerproductWithSwish pattern.

    Fuse the original sub-graph into the custom acceleration 'InnerproductWithSwish' graph.
    The search strategy is based on the following pattern mapping configs for different models.
    """

    def __call__(self, model):
        """The __call__ function of this pattern class."""
        pattern_mapping_config = {
            "InnerproductWithSwish": [
                {
                    "patterns": {
                        "in": [[(0, "InnerProduct"), (1, "Swish")]],
                        "out": [[(0, "InnerProduct")]],
                    },
                    "search_mode": "op_type",
                    "node_names": {0: 0},
                    "input_tensors": {0: [[{0: [0]}, {0: [1]}], [[0, 1], 2]]},
                    "output_tensors": {0: [[{1: [0]}], [[0], 1]]},
                    "returns": [0],
                },
            ]
        }

        def _set_attr(new_node_names, ret_old_nodes, model):
            for i in range(len(new_node_names)):
                mat_node_idx = model.get_node_id(new_node_names[i][0])
                ret_mat_node = ret_old_nodes[i][0]
                if len(ret_mat_node.input_tensors) == 3:
                    model.nodes[mat_node_idx].input_tensors.append(
                        copy.deepcopy(ret_mat_node.input_tensors[-1])
                    )
                attr = OrderedDict()
                attr["append_op"] = "swish"
                model.nodes[mat_node_idx].attr = attr

        pattern_dict = pattern_mapping_config["InnerproductWithSwish"][0]
        model, new_node_names, ret_old_nodes = util.pattern_mapping(
            "InnerproductWithSwish", pattern_dict, model
        )
        if len(new_node_names) != 0:
            _set_attr(new_node_names, ret_old_nodes, model)
            return model

        return model
