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

"""The InputData Pattern."""

import copy
from .pattern import Pattern, pattern_registry
from collections import namedtuple, OrderedDict
from .. import graph_utils as util


@pattern_registry(pattern_type='InputData')
class InputData(Pattern):
    """The InputData pattern.

    Fuse the original sub-graph into the custom acceleration 'InputData' graph.
    The search strategy is based on the following pattern mapping configs for different models.
    """
    def __call__(self, model):
        """The __call__ function of this pattern class."""
        pattern_mapping_config = {
            'InputData': [
                {
                    'patterns': {
                        'in': [[(0, 'input_ids'), (1, 'segment_ids'), (2, 'input_mask')]],
                        'out': [[(0, 'Input')]]
                    },
                    'search_mode': 'node_name',
                    'node_names': {
                        0: 'input_data'
                    },
                    'input_tensors': {
                        0: [[], [[], 0]]
                    },
                    'output_tensors': {
                        0: [[{
                            0: [0]
                        }, {
                            1: [0]
                        }, {
                            2: [0]
                        }], [[0, 1, 2], 3]]
                    },
                    'returns': [0, 1, 2]
                },
                
                # onnx model from huggingface
                {
                    'patterns': {
                        'in': [[(0, 'input_ids'), (1, 'token_type_ids'), (2, 'attention_mask')]],
                        'out': [[(0, 'Input')]]
                    },
                    'search_mode': 'node_name',
                    'node_names': {
                        0: 'input_data'
                    },
                    'input_tensors': {
                        0: [[], [[], 0]]
                    },
                    'output_tensors': {
                        0: [[{
                            0: [0]
                        }, {
                            1: [0]
                        }, {
                            2: [0]
                        }], [[0, 1, 2], 3]]
                    },
                    'returns': [0, 1, 2]
                },

                # bert_base
                {
                    'patterns': {
                        'in': [[(0, 'IteratorGetNext')]],
                        'out': [[(0, 'Input')]]
                    },
                    'search_mode': 'node_name',
                    'node_names': {
                        0: 'input_data'
                    },
                    'input_tensors': {
                        0: [[], [[], 0]]
                    },
                    'output_tensors': {
                        0: [[{
                            0: [0]
                        }, {
                            0: [4]
                        }, {
                            0: [1]
                        }], [[0, 1, 2], 3]]
                    },
                    'returns': []
                },

                # dlrm
                {
                    'patterns': {
                        'in': [[(0, 'dense_x'), (1, 'offsets'), (2, 'indices')]],
                        'out': [[(0, 'Input')]]
                    },
                    'search_mode': 'node_name',
                    'node_names': {
                        0: 'input_data'
                    },
                    'input_tensors': {
                        0: [[], [[], 0]]
                    },
                    'output_tensors': {
                        0: [[{
                            0: [0]
                        }, {
                            1: [0]
                        }, {
                            2: [0]
                        }], 
                        [[0, 1, 2], 3]]
                    },
                    'returns': [0, 1, 2]
                },

                # distil_bert_base
                {
                    'patterns': {
                        'in': [[(0, 'input_ids'), (1, 'attention_mask') ]],
                        'out': [[(0, 'Input')]]
                    },
                    'search_mode': 'node_name',
                    'node_names': {
                        0: 'input_data'
                    },
                    'input_tensors': {
                        0: [[], [[], 0]]
                    },
                    'output_tensors': {
                        0: [[{
                            0: [0]
                        }, {
                            1: [0]
                        }
                        ], [[0, 1], 2]]
                    },
                    'returns': []
                },

                # stale diffusion text encdoer
                {
                    'patterns': {
                        'in': [[(0, 'input_ids')]],
                        'out': [[(0, 'Input')]]
                    },
                    'search_mode': 'node_name',
                    'node_names': {
                        0: 'input_data'
                    },
                    'input_tensors': {
                        0: [[], [[], 0]]
                    },
                    'output_tensors': {
                        0: [[{
                            0: [0]
                        }], [[], 1]]
                    },
                    'returns': []
                },

                # stale diffusion unet
                {
                    'patterns': {
                        'in': [[(0, 'sample'), (1, 'timestep'), (2, 'encoder_hidden_states')]],
                        'out': [[(0, 'Input')]]
                    },
                    'search_mode': 'node_name',
                    'node_names': {
                        0: 'input_data'
                    },
                    'input_tensors': {
                        0: [[], [[], 0]]
                    },
                    'output_tensors': {
                        0: [[{
                            0: [0]
                        }, {
                            1: [0]
                        }, {
                            2: [0]
                        }], 
                        [[0, 1, 2], 3]]
                    },
                    'returns': [0, 1, 2]
                },

                # stale diffusion vae decoder
                {
                    'patterns': {
                        'in': [[(0, 'latent_sample')]],
                        'out': [[(0, 'Input')]]
                    },
                    'search_mode': 'node_name',
                    'node_names': {
                        0: 'input_data'
                    },
                    'input_tensors': {
                        0: [[], [[], 0]]
                    },
                    'output_tensors': {
                        0: [[{
                            0: [0]
                        }], [[], 1]]
                    },
                    'returns': []
                },
                
                # minilmv2-lat-roberta
                {
                    'patterns': {
                        'in': [[(0, 'input_ids'), (1, 'input_mask') ]],
                        'out': [[(0, 'Input')]]
                    },
                    'search_mode': 'node_name',
                    'node_names': {
                        0: 'input_data'
                    },
                    'input_tensors': {
                        0: [[], [[], 0]]
                    },
                    'output_tensors': {
                        0: [[{
                            0: [0]
                        }, {
                            1: [0]
                        }
                        ], [[0, 1], 2]]
                    },
                    'returns': []
                },

                # vit
                {
                    'patterns': {
                        'in': [[(0, ['input', 'pixel_values'])]],
                        'out': [[(0, 'Input')]]
                    },
                    'search_mode': 'node_name',
                    'node_names': {
                        0: 'input_data'
                    },
                    'input_tensors': {
                        0: [[], [[], 0]]
                    },
                    'output_tensors': {
                        0: [[{0: [0]}], [[0], 1]]
                    },
                    'returns': [0]
                },

            ]
        }

        for i in range(len(pattern_mapping_config['InputData'])):
            pattern_dict = pattern_mapping_config['InputData'][i]
            model, new_node_names, ret_old_nodes = util.pattern_mapping("InputData",
                                                                        pattern_dict, model)
            if len(new_node_names) != 0:
                model.nodes[0].attr = None
                for j in range(len(model.nodes[0].output_tensors)):
                    if model.nodes[0].output_tensors[j].shape is None:
                        model.nodes[0].output_tensors[j].shape = [-1, -1]

                return model

        # if no input_data, insert input node for subgraph
        if model.nodes[0].op_type != "Input":
            onnx_input_nodes_list = []
            model_input_tensors = []
            for node in model.nodes:
                if node.op_type == "ONNXINPUT":
                    onnx_input_nodes_list.append(node.name)
                    for input_tensor in node.output_tensors:
                         model_input_tensors.append(copy.deepcopy(input_tensor))
            input_data_node = util.construct_node('input_data',
                                                   'Input',
                                                   output_tensors=model_input_tensors)
            model.insert_nodes(0, [input_data_node])
            model.nodes[0].attr = None
            model.remove_nodes(onnx_input_nodes_list)

        return model
