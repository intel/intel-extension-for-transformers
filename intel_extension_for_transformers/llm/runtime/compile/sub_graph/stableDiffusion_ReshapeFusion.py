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
"""The StableDiffusion_ReshapeFusion Pattern."""

from .pattern import Pattern, pattern_registry
from collections import OrderedDict
from .. import graph_utils as util
from .subgraph_matcher import EXECUTOR_TYPE
from .. import logger


@pattern_registry(pattern_type='StableDiffusion_ReshapeFusion')
class StableDiffusion_ReshapeFusion(Pattern):
    """The StableDiffusion_ReshapeFusion pattern.

    Fuse the original sub-graph into the custom acceleration 'StableDiffusion_ReshapeFusion' graph.
    The search strategy is based on the following pattern mapping configs for different models.
    """

    def __call__(self, model):
        """The __call__ function of this pattern class."""
        pattern_mapping_config = {
            'StableDiffusion_ReshapeFusion': [
                {
                    # unet
                    'patterns': {
                        'in': [[(0, 'Gather'), (1, 'Reshape'), (2, 'Reshape'), (3, 'BinaryAdd'),
                                (4, 'Reshape'), (5, 'Reshape')]],
                        'out': [[(0, 'Gather')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 0
                    },
                    'input_tensors': {
                        0: [[{
                            0: [0]
                        }, {
                            0: [1]
                        }, {
                            3: [0]
                        }, {
                            1: [1]
                        }], [[0, 1, 2, 3], 4]]
                    },
                    'output_tensors': {
                        0: [[{
                            5: [0]
                        }], [[0], 1]]
                    },
                    'returns': [0, 1, 2, 3]
                },
            ]
        }
        # fuse reshape nodes to previous innerproduct or matmul attr to enhance perf
        remove_node_name = []
        for node in model.nodes:
            if node.op_type == "Reshape" and node.input_tensors[0].source_op:
                pre_node = model.get_node_by_name(node.input_tensors[0].source_op[0])
                if pre_node.op_type in EXECUTOR_TYPE \
                   and (EXECUTOR_TYPE[pre_node.op_type] == "InnerProduct" or \
                        EXECUTOR_TYPE[pre_node.op_type] == "Matmul"):
                    # Multiple dest_ops will cause node.input_tensors[1] left in the output.
                    if len(node.input_tensors[1].dest_op) != 1:
                        continue

                    pre_node.attr['reshape'] = node.attr['dst_shape']
                    pre_node.output_tensors[0] = node.output_tensors[0]
                    pre_node.output_tensors[0].source_op = [pre_node.name]
                    if 'dims' in node.attr:
                        pre_node.attr['reshape_dims'] = str(node.attr['dims'])
                        pre_node.input_tensors.append(node.input_tensors[1])

                    if node.output_tensors[0].dest_op != []:
                        next_node = model.get_node_by_name(node.output_tensors[0].dest_op[0])
                        for input_tensor in next_node.input_tensors:
                            if input_tensor.name == node.output_tensors[0].name:
                                input_tensor.source_op = [pre_node.name]

                    remove_node_name.append(node.name)

        model.remove_nodes(remove_node_name)

        for i in range(len(pattern_mapping_config['StableDiffusion_ReshapeFusion'])):
            pattern_dict = pattern_mapping_config['StableDiffusion_ReshapeFusion'][i]
            model, new_node_names, ret_old_nodes = util.pattern_mapping("StableDiffusion_ReshapeFusion",
                                                                        pattern_dict, model)

            if len(new_node_names) != 0:
                logger.info('StableDiffusion_ReshapeFusion mathched...')
                logger.debug('StableDiffusion_ReshapeFusion = {}'.format(new_node_names))
                for i in range(len(new_node_names)):
                    offset = 0
                    if len(new_node_names[i]) == 3:
                        offset = 1
                    gather_node = model.get_node_by_name(new_node_names[i][offset])
                    attr = OrderedDict()
                    attr["axis"] = ret_old_nodes[i][0].attr['axis']
                    attr["batch_dims"] = ret_old_nodes[i][0].attr['batch_dims']
                    attr["append_op"] = "binary_add"
                    attr["reshape"] = ret_old_nodes[i][2].attr['dst_shape']
                    attr["reshape_dims"] = ret_old_nodes[i][2].attr['dims']
                    attr["mul"] = ret_old_nodes[i][2].attr['mul']
                    gather_node.attr = attr
                    if len(new_node_names[i]) >= 2:
                        import copy
                        newattr = copy.deepcopy(attr)
                        model.get_node_by_name(new_node_names[i][offset + 1]).attr = newattr
                    if len(ret_old_nodes[i][3].output_tensors[0].dest_op) == 3:
                        matMulWithBiasAdd_node = model.get_node_by_name(
                            ret_old_nodes[i][3].output_tensors[0].dest_op[1])
                        assert matMulWithBiasAdd_node.op_type == 'MatMulWithBiasAdd'
                        matMulWithBiasAdd_node.input_tensors[3] = gather_node.output_tensors[0]
        return model
