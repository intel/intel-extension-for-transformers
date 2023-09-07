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
"""The Transformer2Dmodel_FFNInputSlice pattern."""

from .pattern import Pattern, pattern_registry
from collections import OrderedDict
from .. import graph_utils as util
from .. import logger


@pattern_registry(pattern_type='Transformer2Dmodel_FFNInputSlice')
class Transformer2Dmodel_FFNInputSlice(Pattern):
    """The Transformer2Dmodel_FFNInputSlice pattern.

    Fuse the original sub-graph into the custom acceleration 'Transformer2Dmodel_FFNInputSlice' graph.
    The search strategy is based on the following pattern mapping configs for the stable textEncoderV1-5.
    """
    def __call__(self, model):
        """The __call__ function of this pattern class."""
        pattern_mapping_config = {
            # must be placed ahead of the FFNSlice_1 pattern
            'Transformer2Dmodel_FFNInputSlice': [
                {
                    'patterns': {
                        'in': [[(0, 'MatMulWithBias'), (1, 'Shape'), (2, 'Gather'), (3, 'Add'), (4, 'Div'),
                                (5, 'Mul'), (6, 'Slice')]],
                    },
                },
            ]
        }
        #
        pattern = pattern_mapping_config['Transformer2Dmodel_FFNInputSlice'][0]['patterns']['in']
        patterns_nodes_name = util.search_pattern(pattern, model)
        logger.info('Transformer2Dmodel_FFNInputSlice mathched...')
        logger.debug('Transformer2Dmodel_FFNInputSlice = {}'.format(patterns_nodes_name))
        if len(patterns_nodes_name) != 0:
            for j in range(len(patterns_nodes_name)):
                matmul_node = model.get_node_by_name(patterns_nodes_name[j][0])
                bias_shape = matmul_node.input_tensors[1].data.shape[1]

                slice_node = model.get_node_by_name(patterns_nodes_name[j][6])
                if 'starts' in slice_node.attr and slice_node.attr['starts'] == '0':
                    slice_node.input_tensors.pop(1)
                    attr = OrderedDict()
                    attr['starts'] = 0
                    attr['ends'] = int(bias_shape / 2)
                    attr['axes'] = 1
                    attr['steps'] = 1
                    slice_node.attr = attr
                else:
                    attr = OrderedDict()
                    attr['starts'] = int(bias_shape / 2)
                    attr['ends'] = bias_shape
                    attr['axes'] = 1
                    attr['steps'] = 1
                    slice_node.attr = attr

        return model