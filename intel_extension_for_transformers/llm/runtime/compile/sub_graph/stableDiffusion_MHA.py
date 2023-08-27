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

from .pattern import Pattern, pattern_registry
from collections import namedtuple, OrderedDict
from .. import graph_utils as util
import numpy as np
from .. import logger


@pattern_registry(pattern_type='StableDiffusion_MHA')
class StableDiffusion_MHA(Pattern):
    def __call__(self, model):
        pattern_mapping_config = {
            'StableDiffusion_MHA': [
                # for multi head attention
                # Bert based models
                {
                    'patterns': {
                        'in': [[(0, 'TransposeBatchMatMul'), (1, 'Softmax'), (2, 'TransposeBatchMatMul')]],
                        'out': [[(0, 'MultiHeadAttention')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 2,
                    },
                    'input_tensors': {
                        0: [
                            [
                                {
                                    0: [0]
                                },  # Q
                                {
                                    0: [1]
                                },  # K
                                {
                                    2: [1]
                                },  # V
                            ],
                            [[0, 1, 2], 3]
                        ]
                    },
                    'output_tensors': {
                        0: [[{
                            2: [0]
                        }], [[0], 1]]
                    },
                    'returns': [0, 1, 2]
                },
            ]
        }

        def _set_attr(new_node_names, ret_old_nodes, model):
            for i in range(len(new_node_names)):
                new_node = model.get_node_by_name(new_node_names[i][0])
                attr = OrderedDict()
                if len(ret_old_nodes[i]) == 3:
                    if 'src0_perm' in ret_old_nodes[i][0].attr.keys():
                        attr['Q_perm'] = ret_old_nodes[i][0].attr['src0_perm']
                    if 'src1_perm' in ret_old_nodes[i][0].attr.keys():
                        attr['K_perm'] = ret_old_nodes[i][0].attr['src1_perm']
                    if 'output_scale' in ret_old_nodes[i][0].attr.keys():
                        attr['output_scale'] = ret_old_nodes[i][0].attr['output_scale']
                    if 'src1_perm' in ret_old_nodes[i][2].attr.keys():
                        attr['V_perm'] = ret_old_nodes[i][2].attr['src1_perm']
                    if 'dst_perm' in ret_old_nodes[i][2].attr.keys():
                        attr['dst_perm'] = ret_old_nodes[i][2].attr['dst_perm']
                    if 'reshape' in ret_old_nodes[i][2].attr.keys():
                        attr['reshape'] = ret_old_nodes[i][2].attr['reshape']
                    if 'version' in ret_old_nodes[i][1].attr.keys() and ret_old_nodes[i][1].attr['version'] == 'V2':
                        attr['stable_softmax'] = True
                    attr['output_dtype'] = 'bf16'
                elif len(ret_old_nodes[i]) == 6:
                    if 'src0_perm' in ret_old_nodes[i][0].attr.keys():
                        attr['Q_perm'] = ret_old_nodes[i][0].attr['src0_perm']
                    if 'src1_perm' in ret_old_nodes[i][0].attr.keys():
                        attr['K_perm'] = ret_old_nodes[i][0].attr['src1_perm']
                    if ret_old_nodes[i][2].attr.get('algorithm', None) == 'div':
                        assert isinstance(ret_old_nodes[i][2].input_tensors[1].data, np.ndarray)
                        attr['output_scale'] = 1 / ret_old_nodes[i][2].input_tensors[1].data.item()
                    if 'src1_perm' in ret_old_nodes[i][5].attr.keys():
                        attr['V_perm'] = ret_old_nodes[i][5].attr['src1_perm']
                    if 'dst_perm' in ret_old_nodes[i][5].attr.keys():
                        attr['dst_perm'] = ret_old_nodes[i][5].attr['dst_perm']
                    if 'reshape' in ret_old_nodes[i][5].attr.keys():
                        attr['reshape'] = ret_old_nodes[i][5].attr['reshape']
                    if 'output_dtype' in ret_old_nodes[i][5].attr.keys():
                        attr['output_dtype'] = ret_old_nodes[i][5].attr['output_dtype']
                new_node.attr = attr
                if len(new_node.input_tensors) == 15:
                    mask_1 = new_node.input_tensors[4]
                    if model.get_node_by_name(mask_1.source_op[0]).op_type == "PaddingSequence":
                        new_node.input_tensors[3], new_node.input_tensors[4] = \
                            new_node.input_tensors[4], new_node.input_tensors[3]

        for i in range(len(pattern_mapping_config['StableDiffusion_MHA'])):
            pattern_dict = pattern_mapping_config['StableDiffusion_MHA'][i]
            model, new_node_names, ret_old_nodes = util.pattern_mapping("StableDiffusion_MHA", pattern_dict,
                                                                        model)

            if len(new_node_names) != 0:
                logger.info('StableDiffusion_MHA mathched...')
                logger.debug('StableDiffusion_MHA = {}'.format(new_node_names))
                _set_attr(new_node_names, ret_old_nodes, model)
                return model

        return model
