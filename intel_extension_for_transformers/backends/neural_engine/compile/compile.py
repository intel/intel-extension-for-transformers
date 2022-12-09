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

"""The neural engine compile module."""

from collections import OrderedDict
from .loaders.loader import Loader
from .extractors.extractor import Extractor
from .sub_graph.subgraph_matcher import SubGraphMatcher
from .graph_utils import get_model_fwk_name

COMPILES = OrderedDict({
    'loader': Loader,
    'extractor': Extractor,
    'sub_graph': SubGraphMatcher,
})


def _config_validation(config):
    """The validation of the input config."""
    if config == None:
        return None

    import yaml
    from schema import Schema

    with open(config, 'r') as conf_file:
        conf = yaml.safe_load(conf_file)

    conf_schema = Schema(
        {'pattern_switch': Schema({str: bool}, error='You should provide correct fused_patterns.')})

    return conf_schema.validate(conf)


def start_pipeline(model, config=None):
    """The compile pipeline."""
    compile_list = []
    # initialize the compile
    for compile_type in COMPILES.keys():
        compile_ = COMPILES[compile_type]()
        compile_list.append(compile_)
    # convert the model
    for compile_ in compile_list:
        model = compile_(model, pattern_config=config)
    return model


def compile(model, config=None):
    """The compile interface.

    Firstly, use model loader to get the computation graph with corresponding framework. 
    The graph contains nodes and edges, the node is op and the edge is the tensor.
    Then extract the ops in the graph and pack them to our form.
    Next exploit these above ops to consist sub-graph, which can see as "a new big op", like LayerNorm.

    Note:
        There may have different computation flow in one subgraph.
    Finally, convert them to .yaml file and .bin file for model configuration and inference.
    """
    if get_model_fwk_name(model) == 'neural engine':
        from .graph import Graph
        graph = Graph()
        graph.graph_init(model + '/conf.yaml', model + '/model.bin')
        model = graph
    else:
        config = _config_validation(config)
        model = start_pipeline(model, config=config)
    return model
