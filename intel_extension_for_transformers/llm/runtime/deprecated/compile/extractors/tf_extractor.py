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

"""The neural engine tensorflow extractor file.

   Extract all nodes in the input tensorflow model and convert them to engine python operators.
   All these python operators will compose the engine graph in the order of the original tensorflow
   calculation graph. Please noticed that the mapping between operators is not one-to-one
   correspondence. For more related deatils, please refer to 'ops' APIs.
"""

from .. import logger
from ..graph.graph import Graph
from ..ops.op import OPERATORS
from ..tf_utils import graph_node_names_details
from ..graph_utils import names_from_input


class TensorflowExtractor(object):
    """The tensorflowExtractor Class.

    Decorate the node in model.graph_def, and the new node has the attributes like input_tensors
    and output_tensors, these tensors record the source/dest op name. All of these nodes
    (in a list) will compose a graph, which is Graph class, as the return object.
    
    Args:
        model: TensorflowModel

    Return:
        Graph: Graph class, the new graph object
    """
    @classmethod
    def __call__(self, model):
        """The __call__ function of the extractor."""
        nodes = model.graph_def.node
        graph_nodes_dict = graph_node_names_details(nodes)
        logger.info('Start to extarct tensorflow model ops...')
        new_graph = Graph()
        new_graph.framework_modeling_config['framework'] = 'tensorflow'
        for node in nodes:
            # ignore the Const nodes, they have no source inputs
            if node.op == 'Const':
                continue
            else:
                op_type = node.op
                if op_type not in OPERATORS.keys():
                    op_type = "OpAny"
                new_node = OPERATORS[op_type]()
                new_node.extract('tensorflow', node, model, graph_nodes_dict)
                new_graph.insert_nodes(len(new_graph.nodes), [new_node])

        return new_graph
