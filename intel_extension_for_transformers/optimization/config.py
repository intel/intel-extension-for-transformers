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
from neural_compressor.conf.config import (
    Distillation_Conf, Pruner, Pruning_Conf, Quantization_Conf
)
from neural_compressor.conf.dotdict import DotDict, deep_set
from intel_extension_for_transformers.optimization.utils.metrics import Metric
from intel_extension_for_transformers.optimization.utils.objectives import Objective, performance
from intel_extension_for_transformers.optimization.quantization import QuantizationMode, SUPPORTED_QUANT_MODE
from intel_extension_for_transformers.optimization.distillation import (
    Criterion, DistillationCriterionMode, SUPPORTED_DISTILLATION_CRITERION_MODE
)
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


class QuantizationConfig(object):
    """Configure the quantization process.

    Args:
        framework: Which framework you used
        approach: Which quantization approach to use
        strategy: Which quantization tuning strategy to use
        timeout: Tuning timeout(seconds), 0 means early stop. Combined with max_trials field to decide when to exit
        max_trials: Max tune times
        metrics: Used to evaluate accuracy of tuning model, no need for NoTrainerOptimize
        objectives: Objective with accuracy constraint guaranteed
        config_file: Path to the config file
        sampling_size: How many samples to use
        use_bf16: Whether to use bf16
        recipes: apply recipes for quantization, neural_compressor support below recipes:
            'smooth_quant': whether do smooth quant
            'smooth_quant_args': parameters for smooth_quant
            'fast_bias_correction': whether do fast bias correction
            'weight_correction': whether do weight correction
            'gemm_to_matmul': whether convert gemm to matmul and add, only valid for onnx models
            'graph_optimization_level': support 'DISABLE_ALL', 'ENABLE_BASIC', 'ENABLE_EXTENDED', 'ENABLE_ALL'
                                      only valid for onnx models
            'first_conv_or_matmul_quantization': whether quantize the first conv or matmul
            'last_conv_or_matmul_quantization': whether quantize the last conv or matmul
            'pre_post_process_quantization': whether quantize the ops in preprocess and postprocess
            'add_qdq_pair_to_weight': whether add QDQ pair for weights, only vaild for onnxrt_trt_ep
            'optypes_to_exclude_output_quant': don't quantize output of specified optypes
            'dedicated_qdq_pair': whether dedicate QDQ pair, only vaild for onnxrt_trt_ep.
    """
    def __init__(
        self,
        framework: str = "pytorch",
        approach: str = "PostTrainingStatic",
        strategy: str = "basic",
        timeout: int = 0,
        max_trials: int = 100,
        metrics: Union[Metric, List] = None,
        objectives: Union[Objective, List] = performance,
        config_file: str = None,
        sampling_size: int = 100,
        use_bf16: bool = False,
        recipes: dict = None,
    ):
        """Init a QuantizationConfig object."""
        super().__init__()
        if config_file is None:
            self.inc_config = Quantization_Conf()
        else:
            self.inc_config = Quantization_Conf(config_file)
        self.framework = framework
        if approach is not None:
            self.approach = approach
        if strategy is not None:
            self.strategy = strategy
        if timeout is not None:
            self.timeout = timeout
        if max_trials is not None:
            self.max_trials = max_trials
        if metrics is not None:
            self.metrics = metrics
        else:
            self._metrics = None
        if objectives is not None:
            self.objectives = objectives
        else:
            self._objectives = None
        if sampling_size is not None:
            self.sampling_size = sampling_size
        self.inc_config.usr_cfg.use_bf16 = use_bf16
        if recipes is not None:
            self.recipes = recipes

    @property
    def approach(self):
        """Get the quantization approach."""
        return self.inc_config.usr_cfg.quantization.approach

    @approach.setter
    def approach(self, approach):
        """Set the quantization approach."""
        approach = approach.upper()
        assert approach in SUPPORTED_QUANT_MODE, \
            f"quantization approach: {approach} is not support!" + \
            "PostTrainingStatic, PostTrainingDynamic and QuantizationAwareTraining are supported!"
        self.inc_config.usr_cfg.quantization.approach = QuantizationMode[approach].value

    @property
    def input_names(self):
        """Get the input names."""
        return self.inc_config.usr_cfg.model.inputs

    @input_names.setter
    def input_names(self, input_names):
        """Set the input names."""
        assert isinstance(input_names, list), "input_names must be a list"
        self.inc_config.usr_cfg.model.inputs = input_names

    @property
    def output_names(self):
        """Get the output names."""
        return self.inc_config.usr_cfg.model.outputs

    @output_names.setter
    def output_names(self, output_names):
        """Set the output names."""
        assert isinstance(output_names, list), "output_names must be a list"
        self.inc_config.usr_cfg.model.outputs = output_names

    @property
    def metrics(self):
        """Get the metrics."""
        return self._metrics

    @metrics.setter
    def metrics(self, metrics: Union[Metric, List]):
        """Set the metrics."""
        self._metrics = metrics
        rel_or_abs = {True: "relative", False: "absolute"}
        assert isinstance(metrics[0] if isinstance(metrics, list) else metrics, Metric), \
            "metric should be a Metric calss!"
        if isinstance(metrics, Metric) or len(metrics) == 1:
            self.inc_config.usr_cfg.tuning.accuracy_criterion = {
                rel_or_abs[metrics[0].is_relative]
                if isinstance(metrics, list) else rel_or_abs[metrics.is_relative]:
                metrics[0].criterion if isinstance(metrics, list) else metrics.criterion,
                "higher_is_better": metrics[0].greater_is_better if isinstance(metrics, list) else
                metrics.greater_is_better
            }
        else:
            weights = [metric.weight_ratio for metric in metrics]
            if not any(weights):
                weight = 1 / len(metrics)
                for metric in metrics:
                    metric.weight_ratio = weight
            else:   # pragma: no cover
                assert all(weights), "Please set the weight ratio for all metrics!"

            assert all(metric.is_relative == metrics[0].is_relative for metric in metrics), \
                "Unsupport different is_relative for different metric now, will support soon!"
            assert all(metric.criterion == metrics[0].criterion for metric in metrics), \
                "Unsupport different criterion for different metric now, will support soon!"

            self.inc_config.usr_cfg.tuning.accuracy_criterion = {
                rel_or_abs[metrics[0].is_relative]: metrics[0].criterion,
                "higher_is_better": metrics[0].greater_is_better
            }

    @property
    def framework(self):
        """Get the framework."""
        return self.inc_config.usr_cfg.model.framework

    @framework.setter
    def framework(self, framework):
        """Set the framework."""
        assert framework in ["pytorch", "pytorch_fx", "pytorch_ipex", "tensorflow"], \
            "framework: {} is not support!".format(framework)
        self.inc_config.usr_cfg.model.framework = framework

    @property
    def objectives(self):
        """Get the objectives."""
        return self._objectives

    @objectives.setter
    def objectives(self, objectives: Union[List, Objective]):
        """Set the objectives."""
        self._objectives = objectives
        if isinstance(objectives, Objective) or len(objectives) == 1:
            self.inc_config.usr_cfg.tuning.objective = objectives.name \
                if isinstance(objectives, Objective) else objectives[0].name
        else:
            weights = [objective.weight_ratio for objective in objectives]
            if not any(weights):
                weight = 1 / len(objectives)
                for objective in objectives:
                    objective.weight_ratio = weight
            else:
                assert all(weights), "Please set the weight ratio for all metrics!"

            self.inc_config.usr_cfg.tuning.multi_objective = {
                "objective": [objective.name for objective in objectives],
                "higher_is_better": [objective.greater_is_better for objective in objectives],
                "weight": [objective.weight_ratio for objective in objectives],
            }

    @property
    def strategy(self):
        """Get the strategy."""
        return self.inc_config.usr_cfg.tuning.strategy.name

    @strategy.setter
    def strategy(self, strategy):
        """Set the strategy."""
        assert strategy in ["basic", "bayesian", "mse", "mse_v2"], \
            "strategy: {} is not support!".format(strategy)
        self.inc_config.usr_cfg.tuning.strategy.name = strategy
        if strategy == "mse_v2":
            self.inc_config.usr_cfg.tuning.strategy_kwargs = {"confidence_batches": 1}

    @property
    def timeout(self):
        """Get the timeout."""
        return self.inc_config.usr_cfg.tuning.exit_policy.timeout

    @timeout.setter
    def timeout(self, timeout):
        """Set the timeout."""
        assert isinstance(timeout, int), "timeout should be integer!"
        self.inc_config.usr_cfg.tuning.exit_policy.timeout = timeout

    @property
    def op_wise(self):
        """Get the op_wise dict."""
        return self.inc_config.usr_cfg.quantization.op_wise

    @op_wise.setter
    def op_wise(self, op_wise):
        """Set the op_wise dict."""
        self.inc_config.usr_cfg.quantization.op_wise = op_wise

    @property
    def optype_wise(self):
        """Get the optype_wise dict."""
        return self.inc_config.usr_cfg.quantization.optype_wise

    @optype_wise.setter
    def optype_wise(self, optype_wise):
        """Set the optype_wise dict."""
        self.inc_config.usr_cfg.quantization.optype_wise = optype_wise

    @property
    def max_trials(self):
        """Get the number of maximum trials."""
        return self.inc_config.usr_cfg.tuning.exit_policy.max_trials

    @max_trials.setter
    def max_trials(self, max_trials):
        """Set the number of maximum trials."""
        assert isinstance(max_trials, int), "max_trials should be integer!"
        self.inc_config.usr_cfg.tuning.exit_policy.max_trials = max_trials

    @property
    def performance_only(self):
        """Get the boolean whether to use performance only."""
        return self.inc_config.usr_cfg.tuning.exit_policy.performance_only

    @performance_only.setter
    def performance_only(self, performance_only):
        """Set the boolean whether to use performance only."""
        assert isinstance(performance_only, boolean), "performance_only should be boolean!"
        self.inc_config.usr_cfg.tuning.exit_policy.performance_only = performance_only

    @property
    def random_seed(self):
        """Get the random seed."""
        return self.inc_config.usr_cfg.tuning.random_seed

    @random_seed.setter
    def random_seed(self, random_seed):
        """Set the random seed."""
        assert isinstance(random_seed, int), "random_seed should be integer!"
        self.inc_config.usr_cfg.tuning.random_seed = random_seed

    @property
    def tensorboard(self):
        """Get the boolean whether to use tensorboard."""
        return self.inc_config.usr_cfg.tuning.tensorboard

    @tensorboard.setter
    def tensorboard(self, tensorboard):
        """Set the boolean whether to use tensorboard."""
        assert isinstance(tensorboard, boolean), "tensorboard should be boolean!"
        self.inc_config.usr_cfg.tuning.tensorboard = tensorboard

    @property
    def output_dir(self):
        """Get the output directory."""
        return self.inc_config.usr_cfg.tuning.workspace.path

    @output_dir.setter
    def output_dir(self, path):
        """Set the output directory."""
        assert isinstance(path, str), "save_path should be a string of directory!"
        self.inc_config.usr_cfg.tuning.workspace.path = path

    @property
    def resume_path(self):
        """Get the resume path."""
        return self.inc_config.usr_cfg.tuning.workspace.resume

    @resume_path.setter
    def resume_path(self, path):
        """Set the resume path."""
        assert isinstance(path, str), "resume_path should be a string of directory!"
        self.inc_config.usr_cfg.tuning.workspace.resume = path

    @property
    def sampling_size(self):
        """Get the sampling size."""
        return self.inc_config.usr_cfg.quantization.calibration.sampling_size

    @sampling_size.setter
    def sampling_size(self, sampling_size):
        """Set the sampling size."""
        if isinstance(sampling_size, int):
            self.inc_config.usr_cfg.quantization.calibration.sampling_size = [sampling_size]
        elif isinstance(sampling_size, list):
            self.inc_config.usr_cfg.quantization.calibration.sampling_size = sampling_size
        else:
            assert False, "The sampling_size must be a list of int numbers"

    @property
    def recipes(self):
        """Get the sampling size."""
        return self.inc_config.usr_cfg.quantization.recipes

    @recipes.setter
    def recipes(self, recipes):
        """Set recipes."""
        if recipes is not None and not isinstance(recipes, dict):
            raise ValueError("recipes should be a dict.")

        # Support PyTorch only
        def smooth_quant(val=None):
            if val is not None:
                return check_value("smooth_quant", val, bool)
            else:
                return False

        # Support PyTorch only
        def smooth_quant_args(val=None):
            if val is not None:
                check_value("smooth_quant_args", val, dict)
                for k, v in val.items():
                    if k == "alpha":
                        assert isinstance(v, str) or isinstance(v, float),\
                            "Smooth_quant_args.alpha should be a float or 'auto'."
                return True
            else:
                return {}

        # Support tensorflow, but not enabled now
        def fast_bias_correction(val=None):  # pragma: no cover
            if val is not None:
                return check_value("fast_bias_correction", val, bool)
            else:
                return False

        # Support tensorflow, but not enabled now
        def weight_correction(val=None):  # pragma: no cover
            if val is not None:
                return check_value("weight_correction", val, bool)
            else:
                return False

        # Support Tensorflow only
        def first_conv_or_matmul_quantization(val=None):
            if val is not None:
                return check_value("first_conv_or_matmul_quantization", val, bool)
            else:
                return True

        # Support Tensorflow only
        def last_conv_or_matmul_quantization(val=None):
            if val is not None:
                return check_value("last_conv_or_matmul_quantization", val, bool)
            else:
                return True

        RECIPES = {"smooth_quant": smooth_quant, # Only for PyTorch
                   "smooth_quant_args": smooth_quant_args, # Only for PyTorch
                   "fast_bias_correction": fast_bias_correction, # Support PyTorch and Tensorflow, not used now.
                   "weight_correction": weight_correction, # Support PyTorch and Tensorflow, not used now.
                   "first_conv_or_matmul_quantization": first_conv_or_matmul_quantization, # Only for Tensorflow
                   "last_conv_or_matmul_quantization": last_conv_or_matmul_quantization, # Only for Tensorflow
                   }
        _recipes = {}
        for k in RECIPES.keys():
            if k in recipes and RECIPES[k](recipes[k]):
                _recipes.update({k: recipes[k]})
            else:
                _recipes.update({k: RECIPES[k]()})
        deep_set(self.inc_config.usr_cfg, 'quantization.recipes', _recipes)


class PruningConfig(object):
    """Configure the pruning process.

    Args:
        framework: Which framework you used
        epochs: How many epochs to prune
        epoch_range: Epoch range list
        initial_sparsity_ratio: Initial sparsity goal, and not needed if pruner_config argument is defined
        target_sparsity_ratio: Target sparsity goal, and not needed if pruner_config argument is defined
        metrics: Used to evaluate accuracy of tuning model, not needed for NoTrainerOptimizer
        pruner_config: Defined pruning behavior, if it is None, then NLP wil create a default pruner with
            'BasicMagnitude' pruning typel
        config_file: Path to the config file
    """
    def __init__(
        self,
        framework: str = "pytorch",
        epochs: int = 1,
        epoch_range: List = [0, 4],
        initial_sparsity_ratio: float=0.0,
        target_sparsity_ratio: float = 0.97,
        metrics: Metric = None,
        pruner_config: Union[List, Pruner] = None,
        config_file: str = None
    ):
        """Init a PruningConfig object."""
        super().__init__()
        self.inc_config = Pruning_Conf(config_file)
        self.framework = framework

        if initial_sparsity_ratio is not None:
            self.initial_sparsity_ratio = initial_sparsity_ratio
        if target_sparsity_ratio is not None:
            self.target_sparsity_ratio = target_sparsity_ratio
        if epoch_range is not None:
            self.epoch_range = epoch_range
        if metrics is not None:
            self.metrics = metrics
        else:
            self._metrics = None
        if pruner_config is not None:
            self.pruner_config = pruner_config
        else:
            self.init_prune_config()
        self.epochs = epochs


    def init_prune_config(self):
        """Init the pruning config."""
        pruner_config = Pruner()
        self.inc_config.usr_cfg.pruning.approach.weight_compression['pruners'] = [pruner_config]

    @property
    def pruner_config(self):
        """Get the pruner config."""
        return self.inc_config.usr_cfg.pruning.approach.weight_compression.pruners

    @pruner_config.setter
    def pruner_config(self, pruner_config):
        """Set the pruner config."""
        if isinstance(pruner_config, list):
            self.inc_config.usr_cfg.pruning.approach.weight_compression.pruners = pruner_config
        else:
            self.inc_config.usr_cfg.pruning.approach.weight_compression.pruners = [pruner_config]

    @property
    def target_sparsity_ratio(self):
        """Get the target sparsity ratio."""
        return self.inc_config.usr_cfg.pruning.approach.weight_compression.target_sparsity

    @target_sparsity_ratio.setter
    def target_sparsity_ratio(self, target_sparsity_ratio):
        """Set the target sparsity ratio."""
        self.inc_config.usr_cfg.pruning.approach.weight_compression.target_sparsity = \
            target_sparsity_ratio

    @property
    def initial_sparsity_ratio(self):
        """Get the initial sparsity ratio."""
        return self.inc_config.usr_cfg.pruning.approach.weight_compression.initial_sparsity

    @initial_sparsity_ratio.setter
    def initial_sparsity_ratio(self, initial_sparsity_ratio):
        """Set the initial sparsity ratio."""
        self.inc_config.usr_cfg.pruning.approach.weight_compression.initial_sparsity = \
            initial_sparsity_ratio

    @property
    def epoch_range(self):
        """Get the epoch range."""
        return [self.inc_config.usr_cfg.pruning.approach.weight_compression.start_epoch,
                self.inc_config.usr_cfg.pruning.approach.weight_compression.end_epoch]

    @epoch_range.setter
    def epoch_range(self, epoch_range):
        """Set the epoch range."""
        assert isinstance(epoch_range, list) and len(epoch_range) == 2, \
          "You should set epoch_range like [a,b] format to match the pruning start and end epoch."
        self.inc_config.usr_cfg.pruning.approach.weight_compression.start_epoch = epoch_range[0]
        self.inc_config.usr_cfg.pruning.approach.weight_compression.end_epoch = epoch_range[1]

    @property
    def epochs(self):
        """Get the epochs."""
        eps = self.inc_config.usr_cfg.pruning.train.epoch \
            if hasattr(self.inc_config.usr_cfg.pruning, "train") else 1
        return eps

    @epochs.setter
    def epochs(self, epochs):
        """Set the epochs."""
        assert isinstance(epochs, int) and epochs > 0, \
          "You should set epochs > 0 and int, not {}.".format(epochs)
        self.inc_config.usr_cfg.pruning["train"] = {"epoch": epochs}

    @property
    def framework(self):
        """Get the framework."""
        return self.inc_config.usr_cfg.model.framework

    @framework.setter
    def framework(self, framework):
        """Set the framework."""
        assert framework.lower() in ["pytorch", "pytorch_fx", "tensorflow"], \
            "framework: {} is not support!".format(framework)
        self.inc_config.usr_cfg.model.framework = framework.lower()

    @property
    def metrics(self):
        """Get the metrics."""
        return self._metrics

    @metrics.setter
    def metrics(self, metrics: Metric):
        """Set the metrics."""
        self._metrics = metrics


class DistillationConfig(object):
    """Configure the distillation process.

    Args:
        framework: Which framework you used
        criterion: Criterion of training, example: "KnowledgeLoss"
        metrics: Metrics for distillation
        inc_config: Distillation config
    """
    def __init__(
        self,
        framework: str = "pytorch",
        criterion: Criterion = None,
        metrics: Metric = None,
        inc_config = None
    ):
        """Init a DistillationConfig object."""
        super().__init__()
        self.inc_config = Distillation_Conf(inc_config)
        self.framework = framework
        if criterion is not None:
            self.criterion = criterion
        if metrics is not None:
            self.metrics = metrics
        else:
            self._metrics = None

    @property
    def framework(self):
        """Get the framework."""
        return self.inc_config.usr_cfg.model.framework

    @framework.setter
    def framework(self, framework):
        """Set the framework."""
        assert framework in ["pytorch", "pytorch_fx", "tensorflow"], \
            "framework: {} is not support!".format(framework)
        self.inc_config.usr_cfg.model.framework = framework

    @property
    def criterion(self):
        """Get the criterion."""
        return self.inc_config.usr_cfg.distillation.train.criterion

    @criterion.setter
    def criterion(self, criterion: Criterion):
        """Set the criterion."""
        assert criterion.name.upper() in SUPPORTED_DISTILLATION_CRITERION_MODE, \
            "The criterion name must be in ['KnowledgeLoss', 'IntermediateLayersLoss']"
        if criterion.name.upper() == DistillationCriterionMode.KNOWLEDGELOSS.name:
            assert criterion.temperature is not None, \
                "Please pass the temperature to Criterion.temperature!"
            assert criterion.loss_types is not None, \
                "Please pass the loss_types to Criterion.loss_types!"
            assert criterion.loss_weight_ratio is not None, \
                "Please pass the loss_weight_ratio to Criterion.loss_weight_ratio!"
            self.inc_config.usr_cfg.distillation.train.criterion = {
                DistillationCriterionMode.KNOWLEDGELOSS.value: {
                    "temperature": criterion.temperature,
                    "loss_types": criterion.loss_types,
                    "loss_weights": criterion.loss_weight_ratio
                }
            }

        if criterion.name.upper() == DistillationCriterionMode.INTERMEDIATELAYERSLOSS.name:
            assert criterion.layer_mappings is not None, \
                "Please pass the layer_mappings to Criterion.layer_mappings!"
            assert criterion.loss_types is not None, \
                "Please pass the loss_types to Criterion.loss_types!"
            assert criterion.loss_weight_ratio is not None, \
                "Please pass the loss_weight_ratio to Criterion.loss_weight_ratio!"
            assert criterion.add_origin_loss is not None, \
                "Please pass the add_origin_loss to Criterion.add_origin_loss!"
            self.inc_config.usr_cfg.distillation.train.criterion = {
                DistillationCriterionMode.INTERMEDIATELAYERSLOSS.value: {
                    "layer_mappings": criterion.layer_mappings,
                    "loss_types": criterion.loss_types,
                    "loss_weights": criterion.loss_weight_ratio,
                    "add_origin_loss": criterion.add_origin_loss
                }
            }

    @property
    def metrics(self):
        """Get the metrics."""
        return self._metrics

    @metrics.setter
    def metrics(self, metrics):
        """Set the metrics."""
        assert isinstance(metrics, Metric), \
            "metric should be a Metric calss!"
        self._metrics = metrics


class TFDistillationConfig(object):
    """Configure the distillation process for Tensorflow.

    Args:
        loss_types: Type of loss
        loss_weights: Weight ratio of loss
        train_steps: Steps of training
        temperature: Parameter for KnowledgeDistillationLoss
    """
    def __init__(
        self,
        loss_types: list = [],
        loss_weights: list = [],
        train_steps: list = [],
        temperature: float = 1.0
    ):
        """Init a TFDistillationConfig object."""
        super().__init__()
        self.loss_types = loss_types
        self.loss_weights = loss_weights
        self.train_steps = train_steps
        self.temperature = temperature


class FlashDistillationConfig(object):
    """The flash distillation configuration used by AutoDistillationConfig."""
    def __init__(
        self,
        block_names: list = [],
        layer_mappings_for_knowledge_transfer: list = [],
        loss_types: list = [],
        loss_weights: list = [],
        add_origin_loss: list = [],
        train_steps: list = [],
    ):
        """Init a FlashDistillationConfig object."""
        super().__init__()
        self.block_names = block_names
        self.layer_mappings_for_knowledge_transfer = layer_mappings_for_knowledge_transfer
        self.loss_types = loss_types
        self.loss_weights = loss_weights
        self.add_origin_loss = add_origin_loss
        self.train_steps = train_steps


class AutoDistillationConfig(object):
    """Configure the auto disillation process.

    Args:
        framework: Which framework you used
        search_space: Search space of NAS
        search_algorithm: Search algorithm used in NAS, e.g. Bayesian Optimization
        metrics: Metrics used to evaluate the performance of the model architecture candidate
        max_trials: Maximum trials in NAS process
        seed: Seed of random process
        knowledge_transfer: Configuration controlling the behavior of knowledge transfer stage
            in the autodistillation
        regular_distillation: Configuration controlling the behavior of regular distillation stage
            in the autodistillation
    """
    def __init__(
        self,
        framework: str = "pytorch",
        search_space: dict = {},
        search_algorithm: str = "BO",
        metrics: Union[List, Metric] = None,
        max_trials: int = None,
        seed: int = None,
        knowledge_transfer: FlashDistillationConfig = None,
        regular_distillation: FlashDistillationConfig = None,
    ):
        """Init a AutoDistillationConfig object."""
        super().__init__()
        self.config = DotDict({
            'model':{'name': 'AutoDistillation'},
            'auto_distillation':{
                'search':{},
                'flash_distillation':{
                    'knowledge_transfer':{},
                    'regular_distillation':{}
                    }
                }
            }
        )
        self.framework = framework
        self.search_space = search_space
        self.search_algorithm = search_algorithm
        if max_trials is not None:
            self.max_trials = max_trials
        if seed is not None:
            self.seed = seed
        if knowledge_transfer is not None:
            self.knowledge_transfer = knowledge_transfer
        if regular_distillation is not None:
            self.regular_distillation = regular_distillation
        if metrics is not None:
            self.metrics = metrics

    @property
    def knowledge_transfer(self):
        """Get the knowledge transfer."""
        return self.config.auto_distillation.flash_distillation.knowledge_transfer

    @knowledge_transfer.setter
    def knowledge_transfer(self, knowledge_transfer):
        """Set the knowledge transfer."""
        if knowledge_transfer is None:
            knowledge_transfer = FlashDistillationConfig()
        for k, v in knowledge_transfer.__dict__.items():
            if v:
                self.config.auto_distillation.flash_distillation.knowledge_transfer[k] = v

    @property
    def regular_distillation(self):
        """Get the regular distillation."""
        return self.config.auto_distillation.flash_distillation.regular_distillation

    @regular_distillation.setter
    def regular_distillation(self, regular_distillation):
        """Set the regular distillation."""
        if regular_distillation is None:
            regular_distillation = FlashDistillationConfig()
        for k, v in regular_distillation.__dict__.items():
            if v:
                self.config.auto_distillation.flash_distillation.regular_distillation[k] = v

    @property
    def framework(self):
        """Get the framework."""
        return self.config.model.framework

    @framework.setter
    def framework(self, framework):
        """Set the framework."""
        assert framework in ["pytorch"], \
            "framework: {} is not support!".format(framework)
        self.config.model.framework = framework

    @property
    def search_space(self):
        """Get the search space."""
        return self.config.auto_distillation.search.search_space

    @search_space.setter
    def search_space(self, search_space: dict):
        """Set the search space."""
        self.config.auto_distillation.search.search_space = search_space

    @property
    def search_algorithm(self):
        """Get the search algorithm."""
        return self.config.auto_distillation.search.search_algorithm

    @search_algorithm.setter
    def search_algorithm(self, search_algorithm: str):
        """Set the search algorithm."""
        self.config.auto_distillation.search.search_algorithm = search_algorithm

    @property
    def max_trials(self):
        """Get the max trials."""
        return self.config.auto_distillation.search.max_trials

    @max_trials.setter
    def max_trials(self, max_trials: int):
        """Set the max trials."""
        self.config.auto_distillation.search.max_trials = max_trials

    @property
    def seed(self):
        """Get the seed."""
        return self.config.auto_distillation.search.seed

    @seed.setter
    def seed(self, seed: int):
        """Set the seed."""
        self.config.auto_distillation.search.seed = seed

    @property
    def metrics(self):
        """Get the metrics."""
        return self._metrics

    @metrics.setter
    def metrics(self, metrics: Union[List, Metric]):
        """Set the metrics."""
        self._metrics = metrics
        if isinstance(metrics, Metric):
            metrics = [metrics]
        self.config.auto_distillation.search.metrics = []
        self.config.auto_distillation.search.higher_is_better = []
        for metric in metrics:
            self.config.auto_distillation.search.metrics.append(metric.name)
            self.config.auto_distillation.search.higher_is_better.append(
                metric.greater_is_better
                )


class NASConfig(object):
    """config parser.

    Args:
        approach: The approach of the NAS.
        search_algorithm: The search algorithm for NAS procedure.

    """

    def __init__(self,
        framework: str = "pytorch",
        approach: str = "basic",
        search_space: dict = {},
        search_algorithm: str = "BO",
        metrics: Union[List, Metric] = None,
        max_trials: int = None,
        seed: int = None,
        ):
        super().__init__()
        self.config = DotDict({
                'model': {'name': 'nas', 'framework': 'NA'},
                'nas': {
                    'approach': approach,
                    'search': {
                        'search_space': search_space,
                        'search_algorithm': search_algorithm
                    }
                },
            }
        )
        self.framework = framework
        self.search_space = search_space
        self.search_algorithm = search_algorithm
        if max_trials is not None:
            self.max_trials = max_trials
        if seed is not None:
            self.seed = seed
        if metrics is not None:
            self.metrics = metrics
        if approach and approach != 'basic':
            self.config[approach] = DotDict({})
            self.__setattr__(approach, self.config[approach])

    @property
    def framework(self):
        """Get the framework."""
        return self.config.model.framework

    @framework.setter
    def framework(self, framework):
        """Set the framework."""
        assert framework in ["pytorch"], \
            "framework: {} is not support!".format(framework)
        self.config.model.framework = framework

    @property
    def search_space(self):
        """Get the search space."""
        return self.config.nas.search.search_space

    @search_space.setter
    def search_space(self, search_space: dict):
        """Set the search space."""
        self.config.nas.search.search_space = search_space

    @property
    def search_algorithm(self):
        """Get the search algorithm."""
        return self.config.nas.search.search_algorithm

    @search_algorithm.setter
    def search_algorithm(self, search_algorithm: str):
        """Set the search algorithm."""
        self.config.nas.search.search_algorithm = search_algorithm

    @property
    def max_trials(self):
        """Get the max trials."""
        return self.config.nas.search.max_trials

    @max_trials.setter
    def max_trials(self, max_trials: int):
        """Set the max trials."""
        self.config.nas.search.max_trials = max_trials

    @property
    def seed(self):
        """Get the seed."""
        return self.config.nas.search.seed

    @seed.setter
    def seed(self, seed: int):
        """Set the seed."""
        self.config.nas.search.seed = seed

    @property
    def metrics(self):
        """Get the metrics."""
        return self._metrics

    @metrics.setter
    def metrics(self, metrics: Union[List, Metric]):
        """Set the metrics."""
        self._metrics = metrics
        if isinstance(metrics, Metric):
            metrics = [metrics]
        self.config.nas.search.metrics = []
        self.config.nas.search.higher_is_better = []
        for metric in metrics:
            self.config.nas.search.metrics.append(metric.name)
            self.config.nas.search.higher_is_better.append(
                metric.greater_is_better
                )

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
    """
    similiar to torch optimizer's interface
    """

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
    """Similiar to torch optimizer's interface."""
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
