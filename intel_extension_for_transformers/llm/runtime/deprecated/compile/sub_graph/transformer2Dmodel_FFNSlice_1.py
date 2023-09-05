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
"""The Transformer2Dmodel_FFNInputSlice_1 pattern."""

from .pattern import Pattern, pattern_registry
from .. import graph_utils as util
from .. import logger


@pattern_registry(pattern_type='Transformer2Dmodel_FFNInputSlice_1')
class Transformer2Dmodel_FFNInputSlice_1(Pattern):
    """The Transformer2Dmodel_FFNInputSlice_1 pattern.

    Fuse the original sub-graph into the custom acceleration 'Transformer2Dmodel_FFNInputSlice_1' graph.
    The search strategy is based on the following pattern mapping configs for the stable textEncoderV1-5.
    """
    def __call__(self, model):
        """The __call__ function of this pattern class."""
        pattern_mapping_config = {
            'Transformer2Dmodel_FFNInputSlice_1': [
                {
                    'patterns': {
                        'in': [[(0, 'Shape'), (1, 'Gather'), (2, 'Add'), (3, 'Div'), (4, 'Mul'),
                                (6, 'Slice')], [(3, 'Div'), (5, 'Mul'), (6, 'Slice')]],
                        'out': [[(0, 'Slice')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 6,
                    },
                    'input_tensors': {
                        0: [[{
                            0: [0]
                        }], [[0], 1]],
                    },
                    'output_tensors': {
                        0: [[{
                            6: [0]
                        }], [[0], 1]]
                    },
                    'returns': [0, 1, 2, 3, 4, 5, 6]
                },
            ]
        }

        for i in range(len(pattern_mapping_config['Transformer2Dmodel_FFNInputSlice_1'])):
            pattern_dict = pattern_mapping_config['Transformer2Dmodel_FFNInputSlice_1'][i]
            model, new_node_names, ret_old_nodes = util.pattern_mapping("Transformer2Dmodel_FFNInputSlice_1",
                                                                        pattern_dict, model)

            if len(new_node_names) != 0:
                logger.info('Transformer2Dmodel_FFNInputSlice_1 mathched...')
                logger.debug('Transformer2Dmodel_FFNInputSlice_1 = {}'.format(new_node_names))
                for j in range(len(new_node_names)):
                    # the first new node
                    assert ret_old_nodes[j][6].op_type == 'Slice'
                    slice_node_idx = model.get_node_id(new_node_names[j][0])
                    model.nodes[slice_node_idx].attr = ret_old_nodes[j][6].attr

                return model

        return model
