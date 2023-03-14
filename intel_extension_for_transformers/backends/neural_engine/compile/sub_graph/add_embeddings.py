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

"""The AddEmbeddings Pattern."""

from .pattern import Pattern, pattern_registry
from collections import namedtuple, OrderedDict
from .. import graph_utils as util
import copy


@pattern_registry(pattern_type='AddEmbeddings')
class AddEmbeddings(Pattern):
    """The AddEmbeddings pattern.

    Fuse the original sub-graph into the custom acceleration 'AddEmbeddings' graph.
    The search strategy is based on the following pattern mapping configs for different models.
    """
    def __call__(self, model):
        """The __call__ function of this pattern class."""
        pattern_mapping_config = {
            'AddEmbeddings': [
                {
                    'patterns': {
                        'in': [[(0, ['AddV2', 'Add']), (1, ['AddV2', 'Add']), (2, 'LayerNorm'),
                                (3, 'Reshape')]],
                        'out': [[(0, 'BinaryAdd'), (1, 'Reshape'), (2, 'Reshape'),
                                (3, 'LayerNorm')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 0,
                        1: 'embeddings/after_add_reshape',
                        2: 'embeddings_add/reshape_2d',
                        3: 3
                    },
                    'input_tensors': {
                        0: [[{
                            0: [1]
                        }, {
                            1: [1]
                        }, {
                            0: [0]
                        }], [[0, 1, 2], 3]],
                        1: [[{
                            'input_data': [0]
                        }], [[1], 2]],
                        2: [[], [[], 1]],
                        3: [[{2: [1]}, {2: [2]}],
                            [[1, 2], 3]],
                    },
                    'output_tensors': {
                        0: [[], [[], 1]],
                        1: [[], [[], 1]],
                        2: [[], [[], 1]],
                        3: [[{
                            3: [0]
                        }], [[0], 1]]
                    },
                    'returns': [0, 2]
                },

                # geminet
                {
                    'patterns': {
                        'in': [[(0, ['AddV2', 'Add']), (1, ['AddV2', 'Add']), (2, 'LayerNorm')]],
                        'out': [[(0, 'BinaryAdd'), (1, 'Reshape'), (2, 'Reshape'),
                                (3, 'LayerNorm')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 0,
                        1: 'embeddings/after_add_reshape',
                        2: 'embeddings_add/reshape_2d',
                        3: 2
                    },
                    'input_tensors': {
                        0: [[{
                            0: [1]
                        }, {
                            1: [1]
                        }, {
                            0: [0]
                        }], [[0, 1, 2], 3]],
                        1: [[{
                            'input_data': [0]
                        }], [[1], 2]],
                        2: [[], [[], 1]],
                        3: [[{2: [1]}, {2: [2]}],
                            [[1, 2], 3]]
                    },
                    'output_tensors': {
                        0: [[], [[], 1]],
                        1: [[], [[], 1]],
                        2: [[], [[], 1]],
                        3: [[{
                            2: [0]
                        }], [[0], 1]]
                    },
                    'returns': [0, 2]
                },


                # distil_bert_base
                # vit: only need 1 reshape node in 'out'
                {
                    'patterns': {
                        'in': [[(0, ['AddV2', 'Add']), (1, 'LayerNorm')]],
                        'out': [[(0, 'BinaryAdd'), (1, 'Reshape'), (2, 'Reshape'),
                                (3, 'LayerNorm')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 0,
                        1: 'embeddings/after_add_reshape',
                        2: 'embeddings_add/reshape_2d',
                        3: 1
                    },
                    'input_tensors': {
                        0: [[{
                            0: [1]
                        },
                        {
                            0: [0]
                        }], [[0, 1], 2]],
                        1: [[{
                            'input_data': [0]
                        }], [[1], 2]],
                        2: [[], [[], 1]],
                        3: [[{1: [1]}, {1: [2]}],
                            [[1, 2], 3]]
                    },
                    'output_tensors': {
                        0: [[ {0: [0]}], [[0], 1]],
                        1: [[], [[], 1]],
                        2: [[], [[], 1]],
                        3: [[{
                            1: [0]
                        }], [[0], 1]]
                    },
                    'returns': [0, 1]
                },

                # opennmt encoder
                {
                    'patterns': {
                        'in': [[(0, ['AddV2', 'Add']), (1, 'Transpose'), (2, 'LayerNorm'), ]],
                        'out': [[(0, 'BinaryAdd'), (1, 'Transpose'), (2, 'Reshape'),
                                 (3, 'LayerNorm')]]
                    },
                    'search_mode': 'op_type',
                    'node_names': {
                        0: 0,
                        1: 1,
                        2: 'embeddings_transpose_reshape_2d',
                        3: 2,
                    },
                    'input_tensors': {
                        0: [[{
                            0: [1]
                        },
                        {
                            0: [0]
                        }], [[0, 1], 2]],
                        1: [[], [[], 1]],
                        2: [[], [[], 1]],
                        3: [[{
                            2: [1]
                        }, {
                            2: [2]
                        }], [[1, 2], 3]],
                    },
                    'output_tensors': {
                        0: [[{
                            0: [0]
                        }], [[0], 1]],
                        1: [[], [[], 1]],
                        2: [[{
                            1: [0]
                        }], [[], 1]],
                        3: [[{
                            2: [0]
                        }], [[], 1]],
                    },
                    'returns': [0, 1, 2]
                },
            ]
        }

        def _set_attr(hidden_size, epsilon, node_names, model, is_vit = False):
            attr1 = OrderedDict()
            attr1['append_op'] = 'sum'
            attr2 = OrderedDict()
            attr2['dst_shape'] = '-1,-1,' + str(hidden_size)
            attr2['dims'] = '0,1'
            attr3 = OrderedDict()
            attr3['dst_shape'] = '-1,' + str(hidden_size)
            attr4 = OrderedDict()
            attr4['epsilon'] = float(epsilon)

            binary_add_node_idx = model.get_node_id(node_names[0])
            model.nodes[binary_add_node_idx].attr = attr1

            reshape_1_node_idx = model.get_node_id(node_names[1])
            model.nodes[reshape_1_node_idx].attr = attr2

            reshape_2_node_idx = model.get_node_id(node_names[2])
            model.nodes[reshape_2_node_idx].attr = attr3
            # In vit we need to remove the first reshape node
            if is_vit:
                model.nodes[reshape_2_node_idx].input_tensors = model.nodes[binary_add_node_idx].output_tensors
                model.remove_nodes([node_names[1]])

            ln_node_idx = model.get_node_id(node_names[3])
            model.nodes[ln_node_idx].attr = attr4
        if model.framework_modeling_config['framework'] == 'onnxruntime':
          # shape = [bs, seq_len, hidden_size] after embeddings
          for i in range(len(pattern_mapping_config['AddEmbeddings']) - 1):
              pattern_dict = pattern_mapping_config['AddEmbeddings'][i]
              model, new_node_names, ret_old_nodes = util.pattern_mapping("AddEmbeddings", 
                                                                          pattern_dict, model)
              if len(new_node_names) != 0:
                  for j in range(len(new_node_names)):
                      ln_node = ret_old_nodes[j][1]
                      add_node = ret_old_nodes[j][0]
                      is_vit = False
                      if add_node.input_tensors[-1].data is not None:
                          is_vit = True
                      hidden_size = int(ln_node.input_tensors[-1].shape[-1])
                      epsilon = ln_node.attr['epsilon']
                      _set_attr(hidden_size, epsilon, new_node_names[j], model, is_vit)
                      if len(pattern_dict['patterns']['in'][0]) == 2:
                          binary_add_node_idx = model.get_node_id(new_node_names[j][0])
                          model.nodes[binary_add_node_idx].attr = OrderedDict()
  
                  return model
  
          # shape = [seq_len, bs, hidden_size] after embeddings
          for pattern_dict in pattern_mapping_config['AddEmbeddings'][-1:]:
              model, new_node_names, ret_old_nodes = util.pattern_mapping("AddEmbeddings",
                                                                          pattern_dict, model)
              if len(new_node_names) != 0:
                  for j in range(len(new_node_names)):
                      reshape_idx = 0
                      idx = 0
                      for n in ret_old_nodes[j]:
                          if model.get_node_by_name(new_node_names[j][idx]).op_type == "Reshape":
                              reshape_idx = idx
                              idx += 1
                          model.get_node_by_name(new_node_names[j][idx]).attr = copy.deepcopy(n.attr)
                          idx += 1
                      hidden_size = str(model.inquire_config_item("hidden_size"))
                      model.get_node_by_name(new_node_names[j][reshape_idx]).attr = OrderedDict(
                          {'dst_shape': '-1,' + hidden_size})
                  return model

        return model
