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

"""The GatherWithAdd Pattern."""

from .pattern import Pattern, pattern_registry
from collections import namedtuple, OrderedDict
from .. import graph_utils as util
import numpy as np


@pattern_registry(pattern_type='EmbeddingsTo2DBeforeInnerProduct')
class EmbeddingsTo2DBeforeInnerProduct(Pattern):
    """The EmbeddingsTo2DBeforeInnerProduct pattern.

    Fuse the original sub-graph into the custom acceleration 'EmbeddingsTo2DBeforeInnerProduct' graph.
    The search strategy is based on the following pattern mapping configs for different models.
    It's better to call it after the 'OperatorAdaptor' and 'QuantGatherToBF16' patterns.
    """
    def __call__(self, model):
        """The __call__ function of this pattern class."""
        if model.framework_modeling_config['framework'] != 'torch':
            return model
        sub_g = None
        # int8 or bf16
        if util.get_quant_info() or util.get_autocast_info()['cast_type'] in ['bf16', 'int8']:
            sub_g = [[(0, 'Gather'), (1, ['Add', 'BinaryAdd']), (2, 'LayerNorm'), (3, 'Quantize'),
                      (4, 'InnerProduct')]]
        # fp32
        else:
            sub_g = [[(0, 'Gather'), (1, ['Add', 'BinaryAdd']), (2, 'LayerNorm'),
                      (3, 'InnerProduct')]]
        match_ret = util.search_pattern(sub_g, model)
        visited = []
        for ret in match_ret:
            g_node = model.get_node_by_name(ret[0])
            ln_node = model.get_node_by_name(ret[2])
            if ln_node.name not in visited and g_node.attr and g_node.attr.get('embedding', False)\
               and isinstance(g_node.input_tensors[1].data, np.ndarray):
                visited.append(ln_node.name)
                embedding_size = g_node.input_tensors[1].data.shape[-1]
                ln_node.attr['reshape'] = '-1,' + str(embedding_size)

        return model
