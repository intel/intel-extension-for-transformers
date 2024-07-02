#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2022 Intel Corporation
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
"""Config: provide config classes for optimization processes."""

import yaml
from enum import Enum

from neural_compressor.utils.utility import DotDict
from .utils.metrics import Metric
from .utils.objectives import Objective, performance

from typing import List, Union
from xmlrpc.client import boolean


WEIGHTS_NAME = "pytorch_model.bin"


class Provider(Enum):
    """Optimization functionalities provider: INC or NNCF."""
    INC = "inc"


def check_value(name, src, supported_type, supported_value=[]):  # pragma: no cover
    """Check if the given object is the given supported type and in the given supported value.

    Example::

        def datatype(self, datatype):
            if check_value('datatype', datatype, list, ['fp32', 'bf16', 'uint8', 'int8']):
                self._datatype = datatype
    """
    if isinstance(src, list) and any([not isinstance(i, supported_type) for i in src]):
        assert False, ("Type of {} items should be {} but not {}".format(
            name, str(supported_type), [type(i) for i in src]))
    elif not isinstance(src, list) and not isinstance(src, supported_type):
        assert False, ("Type of {} should be {} but not {}".format(
            name, str(supported_type), type(src)))

    if len(supported_value) > 0:
        if isinstance(src, str) and src not in supported_value:
            assert False, ("{} is not in supported {}: {}. Skip setting it.".format(
                src, name, str(supported_value)))
        elif isinstance(src, list) and all([isinstance(i, str) for i in src]) and \
            any([i not in supported_value for i in src]):
            assert False, ("{} is not in supported {}: {}. Skip setting it.".format(
                src, name, str(supported_value)))

    return True
def constructor_register(cls):
    yaml_key = "!{}".format(cls.__name__)

    def constructor(loader, node):
        instance = cls.__new__(cls)
        yield instance

        state = loader.construct_mapping(node, deep=True)
        instance.__init__(**state)

    yaml.add_constructor(
        yaml_key,
        constructor,
        yaml.SafeLoader,
    )
    return cls


class DynamicLengthConfig(object):
    """Configure the dynamic length config for Quantized Length Adaptive Transformer.

    Args:
        max_length: Limit the maximum length of each layer
        length_config: The length number for each layer
        const_rate: Length drop ratio
        num_sandwich: Sandwich num used in training
        length_drop_ratio_bound: Length dropout ratio list
        layer_dropout_prob: The layer dropout with probability
        layer_dropout_bound: Length dropout ratio
        dynamic_training: Whether to use dynamic training
        load_store_file: The path for store file
        evo_iter: Iterations for evolution search
        population_size: Population limitation for evolution search
        mutation_size: Mutation limitation for evolution search
        mutation_prob: Mutation probability used in evolution search
        crossover_size: Crossover limitation for evolution search
        num_cpus: The cpu nums used in evolution search
        distributed_world_size: Distributed world size in evolution search training
        latency_constraint: Latency constraint used in evolution search
        evo_eval_metric: The metric name used in evolution search
    """
    def __init__(
        self,
        max_length: int = None,
        length_config: str = None,
        const_rate: float = None,
        num_sandwich: int = 2,
        length_drop_ratio_bound: float = 0.2,
        layer_dropout_prob: float = None,
        layer_dropout_bound: int = 0,
        dynamic_training: bool = False,
        load_store_file: str = None,
        evo_iter: int = 30,
        population_size: int = 20,
        mutation_size: int = 30,
        mutation_prob: float = 0.5,
        crossover_size: int = 30,
        num_cpus: int = 48,
        distributed_world_size: int = 5,
        latency_constraint: bool = True,
        evo_eval_metric = 'eval_f1'
    ):
        """Init a DynamicLengthConfig object."""
        super().__init__()

        self.length_config = length_config
        self.const_rate = const_rate
        self.max_length = max_length
        self.num_sandwich = num_sandwich
        self.length_drop_ratio_bound = length_drop_ratio_bound
        self.layer_dropout_prob = layer_dropout_prob
        self.layer_dropout_bound = layer_dropout_bound
        self.dynamic_training = dynamic_training
        self.load_store_file = load_store_file
        self.evo_iter = evo_iter
        self.population_size = population_size
        self.mutation_size = mutation_size
        self.mutation_prob = mutation_prob
        self.crossover_size = crossover_size
        self.num_cpus = num_cpus
        self.distributed_world_size = distributed_world_size
        self.latency_constraint = latency_constraint
        self.evo_eval_metric = evo_eval_metric

class BenchmarkConfig:
    """Config Class for Benchmark.

    Args:
        backend (str, optional): the backend used for benchmark. Defaults to "torch".
        warmup (int, optional): skip iters when collecting latency. Defaults to 5.
        iteration (int, optional): total iters when collecting latency. Defaults to 20.
        cores_per_instance (int, optional): the core number for 1 instance. Defaults to 4.
        num_of_instance (int, optional): the instance number. Defaults to -1.
        torchscript (bool, optional): Enable it if you want to jit trace it \
                                      before benchmarking. Defaults to False.
        generate (bool, optional): Enable it if you want to use model.generate \
                                   when benchmarking. Defaults to False.
    """
    def __init__(
        self,
        backend: str = "torch",  # select from ["torch", "ipex", "neural_engine"]
        batch_size: int = 1,
        warmup: int = 5,
        iteration: int = 20,
        cores_per_instance: int = 4,
        num_of_instance: int = -1,
        torchscript: bool = False,
        generate: bool = False,
        **kwargs,
    ):
        """Init a BenchmarkConfig object."""
        self.backend = backend
        self.batch_size = batch_size
        self.warmup = warmup
        self.iteration = iteration
        self.cores_per_instance = cores_per_instance
        self.num_of_instance = num_of_instance
        self.torchscript = torchscript
        self.generate = generate
        self.kwargs = kwargs

    @property
    def backend(self):
        """Get backend."""
        return self._backend

    @backend.setter
    def backend(self, backend):
        """Set backend."""
        self._backend = backend

    @property
    def batch_size(self):
        """Get batch_size."""
        return self._batch_size

    @batch_size.setter
    def batch_size(self, batch_size):
        """Set batch_size."""
        self._batch_size = batch_size

    @property
    def warmup(self):
        """Get warmup."""
        return self._warmup

    @warmup.setter
    def warmup(self, warmup):
        """Set warmup."""
        self._warmup = warmup

    @property
    def iteration(self):
        """Get iteration."""
        return self._iteration

    @iteration.setter
    def iteration(self, iteration):
        """Set iteration."""
        self._iteration = iteration

    @property
    def cores_per_instance(self):
        """Get cores_per_instance."""
        return self._cores_per_instance

    @cores_per_instance.setter
    def cores_per_instance(self, cores_per_instance):
        """Set cores_per_instance."""
        self._cores_per_instance = cores_per_instance

    @property
    def num_of_instance(self):
        """Get num_of_instance."""
        return self._num_of_instance

    @num_of_instance.setter
    def num_of_instance(self, num_of_instance):
        """Set num_of_instance."""
        self._num_of_instance = num_of_instance

    @property
    def torchscript(self):
        """Get torchscript."""
        return self._torchscript

    @torchscript.setter
    def torchscript(self, torchscript):
        """Set torchscript."""
        self._torchscript = torchscript

    @property
    def generate(self):
        """Get generate."""
        return self._generate

    @generate.setter
    def generate(self, generate):
        """Set generate."""
        self._generate = generate

    @property
    def kwargs(self):
        """Get kwargs."""
        return self._kwargs

    @kwargs.setter
    def kwargs(self, kwargs):
        """Set kwargs."""
        self._kwargs = kwargs

@constructor_register
class PrunerV2:
    """Similar to torch optimizer's interface."""

    def __init__(self,
                 target_sparsity=None, pruning_type=None, pattern=None, op_names=None,
                 excluded_op_names=None,
                 start_step=None, end_step=None, pruning_scope=None, pruning_frequency=None,
                 min_sparsity_ratio_per_op=None, max_sparsity_ratio_per_op=None,
                 sparsity_decay_type=None, pruning_op_types=None, reg_type=None,
                 criterion_reduce_type=None, parameters=None, resume_from_pruned_checkpoint=None):
        self.pruner_config = DotDict({
            'target_sparsity': target_sparsity,
            'pruning_type': pruning_type,
            'pattern': pattern,
            'op_names': op_names,
            'excluded_op_names': excluded_op_names,  ##global only
            'start_step': start_step,
            'end_step': end_step,
            'pruning_scope': pruning_scope,
            'pruning_frequency': pruning_frequency,
            'min_sparsity_ratio_per_op': min_sparsity_ratio_per_op,
            'max_sparsity_ratio_per_op': max_sparsity_ratio_per_op,
            'sparsity_decay_type': sparsity_decay_type,
            'pruning_op_types': pruning_op_types,
            'reg_type': reg_type,
            'criterion_reduce_type': criterion_reduce_type,
            'parameters': parameters,
            'resume_from_pruned_checkpoint': resume_from_pruned_checkpoint
        })


class WeightPruningConfig:
    """Similar to torch optimizer's interface."""
    def __init__(self, pruning_configs=[{}],  ##empty dict will use global values
                 target_sparsity=0.9, pruning_type="snip_momentum", pattern="4x1", op_names=[],
                 excluded_op_names=[],
                 start_step=0, end_step=0, pruning_scope="global", pruning_frequency=1,
                 min_sparsity_ratio_per_op=0.0, max_sparsity_ratio_per_op=0.98,
                 sparsity_decay_type="exp", pruning_op_types=['Conv', 'Linear'],
                 **kwargs):
        """Init a WeightPruningConfig object."""
        self.pruning_configs = pruning_configs
        self._weight_compression = DotDict({
            'target_sparsity': target_sparsity,
            'pruning_type': pruning_type,
            'pattern': pattern,
            'op_names': op_names,
            'excluded_op_names': excluded_op_names,  ##global only
            'start_step': start_step,
            'end_step': end_step,
            'pruning_scope': pruning_scope,
            'pruning_frequency': pruning_frequency,
            'min_sparsity_ratio_per_op': min_sparsity_ratio_per_op,
            'max_sparsity_ratio_per_op': max_sparsity_ratio_per_op,
            'sparsity_decay_type': sparsity_decay_type,
            'pruning_op_types': pruning_op_types,
        })
        self._weight_compression.update(kwargs)

    @property
    def weight_compression(self):
        """Get weight_compression."""
        return self._weight_compression

    @weight_compression.setter
    def weight_compression(self, weight_compression):
        """Set weight_compression."""
        self._weight_compression = weight_compression
