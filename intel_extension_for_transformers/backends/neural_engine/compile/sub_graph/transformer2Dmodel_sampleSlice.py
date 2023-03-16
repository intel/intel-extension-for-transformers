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
"""The Transformer2Dmodel_SampleSlice pattern."""

from .pattern import Pattern, pattern_registry
from .. import graph_utils as util
from .. import logger


@pattern_registry(pattern_type='Transformer2Dmodel_SampleSlice')
class Transformer2Dmodel_SampleSlice(Pattern):
    """The Transformer2Dmodel_SampleSlice pattern.

    Fuse the original sub-graph into the custom acceleration 'Transformer2Dmodel_SampleSlice' graph.
    The search strategy is based on the following pattern mapping configs for the stable textEncoderV1-5.
    """
    def __call__(self, model):
        """The __call__ function of this pattern class."""
        pattern_mapping_config = {
            'Transformer2Dmodel_SampleSlice': [
                {
                    'patterns': {
                        'in': [[(0, 'Concat'), (1, 'Slice'), (3, 'Concat'), (4, 'Cast')],
                               [(0, 'Concat'), (2, 'Slice'), (3, 'Concat')]],
                    },
                },
            ]
        }

        pattern = pattern_mapping_config['Transformer2Dmodel_SampleSlice'][0]['patterns']['in']
        patterns_nodes_name = util.search_pattern(pattern, model)
        logger.info('Transformer2Dmodel_SampleSlice mathched...')
        logger.debug('Transformer2Dmodel_SampleSlice = {}'.format(patterns_nodes_name))
        if len(patterns_nodes_name) != 0:
            for j in range(len(patterns_nodes_name)):
                #the first node
                concat_node = model.get_node_by_name(patterns_nodes_name[j][0])
                assert concat_node.op_type == 'Concat'
                if concat_node.attr['axis'] == -1:
                    concat_node.attr['axis'] = 1

                # the second node
                slice_node = model.get_node_by_name(patterns_nodes_name[j][1])
                assert slice_node.op_type == 'Slice'
                if slice_node.attr['ends'] == '-1':
                    slice_node.attr['ends'] = '320'

                #the third node
                concat_node = model.get_node_by_name(patterns_nodes_name[j][3])
                assert concat_node.op_type == 'Concat'
                if concat_node.attr['axis'] == -1:
                    concat_node.attr['axis'] = 1

                cast_node = model.get_node_by_name(patterns_nodes_name[j][4])
                assert cast_node.op_type == 'Cast'
                gemm_node = model.get_node_by_name(cast_node.output_tensors[0].dest_op[0])
                assert gemm_node.op_type == 'MatMulWithBias'

                concat_node.output_tensors[0].dest_op = gemm_node.name
                gemm_node.input_tensors[0] = concat_node.output_tensors[0]
                model.remove_nodes([cast_node.name])

            return model

        return model
