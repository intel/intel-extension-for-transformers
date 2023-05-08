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
"""The neural engine optimizer module."""

from .graph import Graph
from . import graph_utils as util
from . import logger

OPTIMIZED_WEIGHT_FORMAT_TAG = {'FP8': ['ANY', 'INT8', 'FP8_4E3M', 'FP8_5E2M']}


class Optimizer:
    """The defintion of the neural engine optimizer."""

    def __init__(self, graph, input_shape=None, *args, **kwargs):
        """The optimizer initialization.
        
        Args:
            graph: neural engine Graph class
            input_shape: list of list, model input data shape list
        """
        assert isinstance(graph, Graph), 'graph must be an instance of Graph class'
        self.graph = graph
        self.input_shape = input_shape
        self.cast_dtype = util.get_autocast_info()['cast_type']
        self.weight_dtype = util.get_autocast_info().get('weight_dtype', 'native')
        try:
            util.get_environ_info()
        except:
            util.environ_info_init()

    def optimize(self):
        """Optimize the graph."""
        self.weight_optimization()
        # Set env vars before inference. These env vars could help accelerate inference speed.
        util.set_environ_vars(util.get_environ_info())

    def weight_optimization(self):
        """Optimize weight format."""
        if self.cast_dtype == 'bf16' and self.weight_dtype.upper() in \
           OPTIMIZED_WEIGHT_FORMAT_TAG['FP8']:
            self._weight_fp8_dispatch(self.weight_dtype.upper())

    def _weight_fp8_dispatch(self, w_tag):
        """Optimize BF16 graph by using FP8 weight format."""
        tag2env = {'INT8': 'NE_WEIGHT_INT8', 'FP8_4E3M': 'NE_WEIGHT_FP8_4E3M',
                   'FP8_5E2M': 'NE_WEIGHT_FP8_5E2M'}
        util.del_environ_vars(list(tag2env.values()))
        util.remove_environ_info_items(list(tag2env.values()))
        if w_tag == 'ANY':
            # TODO: Consider to add best fp8 weight format search
            best_tag = 'INT8'
            logger.info('Using FP8 weight storage format {} for BF16 model inference'.format(
                        best_tag))
            util.insert_environ_info(tag2env[best_tag], '1')
        elif w_tag in tag2env:
            env_key = tag2env[w_tag]
            logger.info('Using FP8 weight storage format {} for BF16 model inference'.format(
                        w_tag))
            util.insert_environ_info(env_key, '1')
        else:
            logger.warning('Unknown FP8 weight compression format, please use {}'.format(
                           OPTIMIZED_WEIGHT_FORMAT_TAG['FP8']))
