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
from .dynamic_quantize import _dynamic_quantization
from .graph import Graph
from . import graph_utils as util
from copy import deepcopy
from .optimizer import Optimizer

COMPILES = OrderedDict({
    'loader': Loader,
    'extractor': Extractor,
    'sub_graph': SubGraphMatcher,
})


class autocast:
    def __init__(self, cast_type: str, *args, **kwargs) -> None:
        util.autocast_init()
        self.prev_cast_type = util.get_autocast_info()['cast_type']
        self.cast_type = cast_type
        self.weight_dtype = None
        if 'weight_dtype' in kwargs:
            self.weight_dtype = kwargs['weight_dtype']

    def __enter__(self) -> None:
        self.prev_cast_type = util.get_autocast_info()['cast_type']
        util.set_autocast("cast_type", self.cast_type)
        if self.weight_dtype:
            util.set_autocast("weight_dtype", self.weight_dtype)

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        util.set_autocast("cast_type", self.prev_cast_type)


def _config_validation(config):
    """The validation of the pattern config."""
    if config == None:
        return None

    # the config is a dict or text file.
    if isinstance(config, dict) != True:
        with open(config, 'r') as conf_file:
            import yaml
            config = yaml.safe_load(conf_file)

    from schema import Schema
    conf_schema = Schema({'pattern_switch': Schema({str: bool}, error='The format of the pattern config is wrong.')})

    return conf_schema.validate(config)


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


def compile(model, config=None) -> Graph:
    """The compile interface.

    Firstly, use model loader to get the computation graph with corresponding framework.
    The graph contains nodes and edges, the node is op and the edge is the tensor.
    Then extract the ops in the graph and pack them to our form.
    Next exploit these above ops to consist sub-graph, which can see as "a new big op", like LayerNorm.

    Note:
        There may have different computation flow in one subgraph.
    Finally, convert them to .yaml file and .bin file for model configuration and inference.
    """
    from .graph import Graph
    try:
        util.get_autocast_info()
    except:
        util.autocast_init()
    if not isinstance(model, Graph):
        if get_model_fwk_name(model) == 'neural engine':
            graph = Graph()
            graph.graph_init(model + '/conf.yaml',
                             model + '/model.bin',
                             load_weight=util.get_autocast_info()['cast_type'] == "dynamic_int8")
            model = graph
        else:
            config = _config_validation(config)
            model = start_pipeline(model, config=config)
        optimizer = Optimizer(model)
        optimizer.optimize()
    if util.get_autocast_info()['cast_type'] == "dynamic_int8":
        model = _dynamic_quantization(model)
    return model
