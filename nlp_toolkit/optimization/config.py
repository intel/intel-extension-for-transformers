#  Copyright 2021 The HuggingFace Team. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import os
import yaml
from enum import Enum
from functools import reduce
from neural_compressor.conf.config import (
    Distillation_Conf, Pruner, Pruning_Conf, Quantization_Conf
)
from neural_compressor.utils import logger
from nlp_toolkit.optimization.base import Metric, Objective
from nlp_toolkit.optimization.pruning import PruningMode, SUPPORTED_PRUNING_MODE
from nlp_toolkit.optimization.quantization import QuantizationMode, SUPPORTED_QUANT_MODE
from nlp_toolkit.optimization.distillation import (
    Criterion, DistillationCriterionMode, SUPPORTED_DISTILLATION_CRITERION_MODE
)
from transformers.file_utils import cached_path, hf_bucket_url
from typing import Any, List, Optional, Union
from xmlrpc.client import boolean


CONFIG_NAME = "best_configure.yaml"
WEIGHTS_NAME = "pytorch_model.bin"


class Provider(Enum):
    INC = "inc"
    NNCF = "nncf"


class DeployConfig:
    def __init__(self, config_path: str):
        """
        Args:
            config_path (:obj:`str`):
                Path to the YAML configuration file used to control the tuning behavior.
        Returns:
            config: DeployConfig object.
        """

        self.path = config_path
        self.config = self._read_config()
        self.usr_cfg = self.config

    def _read_config(self):
        with open(self.path, "r") as f:
            try:
                config = yaml.load(f, Loader=yaml.Loader)
            except yaml.YAMLError as err:
                logger.error(err)

        return config

    def get_config(self, keys: str):
        return reduce(lambda d, key: d.get(key) if d else None, keys.split("."), self.usr_cfg)

    def set_config(self, keys: str, value: Any):
        d = self.usr_cfg
        keys = keys.split(".")
        for key in keys[:-1]:
            d = d.setdefault(key, {})
        d[keys[-1]] = value

    @classmethod
    def from_pretrained(cls, config_name_or_path: str, config_file_name: Optional[str] = None, **kwargs):
        """
        Instantiate an DeployConfig object from a configuration file which can either be hosted on
        huggingface.co or from a local directory path.

        Args:
            config_name_or_path (:obj:`str`):
                Repository name in the Hugging Face Hub or path to a local directory containing the configuration file.
            config_file_name (:obj:`str`, `optional`):
                Name of the configuration file.
            cache_dir (:obj:`str`, `optional`):
                Path to a directory in which a downloaded configuration should be cached if the standard cache should
                not be used.
            force_download (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to force to (re-)download the configuration files and override the cached versions if
                they exist.
            resume_download (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to delete incompletely received file. Attempts to resume the download if such a file
                exists.
            revision(:obj:`str`, `optional`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
                git-based system for storing models and other artifacts on huggingface.co, so ``revision`` can be any
                identifier allowed by git.
        Returns:
            config: DeployConfig object.
        """

        cache_dir = kwargs.get("cache_dir", None)
        force_download = kwargs.get("force_download", False)
        resume_download = kwargs.get("resume_download", False)
        revision = kwargs.get("revision", None)

        config_file_name = config_file_name if config_file_name is not None else CONFIG_NAME
        if os.path.isdir(config_name_or_path):
            config_file = os.path.join(config_name_or_path, config_file_name)
        elif os.path.isfile(config_name_or_path):
            config_file = config_name_or_path
        else:
            config_file = hf_bucket_url(config_name_or_path, filename=config_file_name, revision=revision)

        try:
            resolved_config_file = cached_path(
                config_file,
                cache_dir=cache_dir,
                force_download=force_download,
                resume_download=resume_download,
            )
        except EnvironmentError as err:
            logger.error(err)
            msg = (
                f"Can't load config for '{config_name_or_path}'. Make sure that:\n\n"
                f"-'{config_name_or_path}' is a correct model identifier listed on 'https://huggingface.co/models'\n\n"
                f"-or '{config_name_or_path}' is a correct path to a directory containing a {config_file_name} file\n\n"
            )

            if revision is not None:
                msg += (
                    f"- or '{revision}' is a valid git identifier (branch name, a tag name, or a commit id) that "
                    f"exists for this model name as listed on its model page on 'https://huggingface.co/models'\n\n"
                )

            raise EnvironmentError(msg)

        config = cls(resolved_config_file)

        return config


class QuantizationConfig(object):
    def __init__(
        self,
        framework: str = "pytorch",
        approach: str = None,
        timeout: int = None,
        max_trials: int = None,
        metrics: Union[Metric, List] = None,
        objectives: Union[Objective, List] = None,
    ):
        super().__init__()
        self.quant_config = Quantization_Conf()
        self.framework = framework
        if approach is not None:
            self.approach = approach
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

    @property
    def approach(self):
        return self.quant_config.usr_cfg.quantization.approach

    @approach.setter
    def approach(self, approach):
        approach = approach.upper()
        assert approach in SUPPORTED_QUANT_MODE, \
            f"quantization approach: {approach} is not support!" + \
            "PostTrainingStatic, PostTrainingDynamic and QuantizationAwareTraining are supported!"
        self.quant_config.usr_cfg.quantization.approach = QuantizationMode[approach].value

    @property
    def metrics(self):
        return self._metrics

    @metrics.setter
    def metrics(self, metrics: Union[Metric, List]):
        self._metrics = metrics
        rel_or_abs = {True: "relative", False: "absolute"}
        assert isinstance(metrics[0] if isinstance(metrics, list) else metrics, Metric), \
            "metric should be a Metric calss!"
        if isinstance(metrics, Metric) or len(metrics) == 1:
            self.quant_config.usr_cfg.tuning.accuracy_criterion = {
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
            else:
                assert all(weights), "Please set the weight ratio for all metrics!"

            assert all(metric.is_relative == metrics[0].is_relative for metric in metrics), \
                "Unsupport different is_relative for different metric now, will support soon!"
            assert all(metric.criterion == metrics[0].criterion for metric in metrics), \
                "Unsupport different criterion for different metric now, will support soon!"

            self.quant_config.usr_cfg.tuning.accuracy_criterion = {
                rel_or_abs[metrics[0].is_relative]: metrics[0].criterion,
                "higher_is_better": metrics[0].greater_is_better
            }

    @property
    def framework(self):
        return self.quant_config.usr_cfg.model.framework

    @framework.setter
    def framework(self, framework):
        assert framework in ["pytorch", "pytorch_fx"], \
            "framework: {} is not support!".format(framework)
        self.quant_config.usr_cfg.model.framework = framework

    @property
    def objectives(self):
        return self._objectives

    @objectives.setter
    def objectives(self, objectives: Union[List, Objective]):
        self._objectives = objectives
        if isinstance(objectives, Objective) or len(objectives) == 1:
            self.quant_config.usr_cfg.tuning.objective = objectives.name \
                if isinstance(objectives, Objective) else objectives[0].name
        else:
            weights = [objective.weight_ratio for objective in objectives]
            if not any(weights):
                weight = 1 / len(objectives)
                for objective in objectives:
                    objective.weight_ratio = weight
            else:
                assert all(weights), "Please set the weight ratio for all metrics!"

            self.quant_config.usr_cfg.tuning.multi_objective = {
                "objective": [objective.name for objective in objectives],
                "higher_is_better": [objective.greater_is_better for objective in objectives],
                "weight": [objective.weight_ratio for objective in objectives],
            }

    @property
    def strategy(self):
        return self.quant_config.usr_cfg.tuning.strategy.name

    @strategy.setter
    def strategy(self, strategy):
        assert strategy in ["basic", "bayesian", "mse"], \
            "strategy: {} is not support!".format(strategy)
        self.quant_config.usr_cfg.tuning.strategy.name = strategy

    @property
    def timeout(self):
        return self.quant_config.usr_cfg.tuning.exit_policy.timeout

    @timeout.setter
    def timeout(self, timeout):
        assert isinstance(timeout, int), "timeout should be integer!"
        self.quant_config.usr_cfg.tuning.exit_policy.timeout = timeout

    @property
    def max_trials(self):
        return self.quant_config.usr_cfg.tuning.exit_policy.max_trials

    @max_trials.setter
    def max_trials(self, max_trials):
        assert isinstance(max_trials, int), "max_trials should be integer!"
        self.quant_config.usr_cfg.tuning.exit_policy.max_trials = max_trials

    @property
    def performance_only(self):
        return self.quant_config.usr_cfg.tuning.exit_policy.performance_only

    @performance_only.setter
    def performance_only(self, performance_only):
        assert isinstance(performance_only, boolean), "performance_only should be boolean!"
        self.quant_config.usr_cfg.tuning.exit_policy.performance_only = performance_only

    @property
    def random_seed(self):
        return self.quant_config.usr_cfg.tuning.random_seed

    @random_seed.setter
    def random_seed(self, random_seed):
        assert isinstance(random_seed, int), "random_seed should be integer!"
        self.quant_config.usr_cfg.tuning.random_seed = random_seed

    @property
    def tensorboard(self):
        return self.quant_config.usr_cfg.tuning.tensorboard

    @tensorboard.setter
    def tensorboard(self, tensorboard):
        assert isinstance(tensorboard, boolean), "tensorboard should be boolean!"
        self.quant_config.usr_cfg.tuning.tensorboard = tensorboard

    @property
    def output_dir(self):
        return self.quant_config.usr_cfg.tuning.workspace.path

    @output_dir.setter
    def output_dir(self, path):
        assert isinstance(path, str), "save_path should be a string of directory!"
        self.quant_config.usr_cfg.tuning.workspace.path = path

    @property
    def resume_path(self):
        return self.quant_config.usr_cfg.tuning.workspace.resume

    @resume_path.setter
    def resume_path(self, path):
        assert isinstance(path, str), "resume_path should be a string of directory!"
        self.quant_config.usr_cfg.tuning.workspace.resume = path


class PruningConfig(object):
    def __init__(
        self,
        framework: str = "pytorch",
        approach: str = "BasicMagnitude",
        target_sparsity_ratio: float = None,
        epoch_range: List = None,
        metrics: Union[List, Metric] = None,
        custom_pruner: Pruner = None,
    ):
        super().__init__()
        self.prune_config = Pruning_Conf()
        self.framework = framework
        self.init_prune_config()
        self.approach = approach
        if target_sparsity_ratio is not None:
            self.target_sparsity_ratio = target_sparsity_ratio
        if epoch_range is not None:
            self.epoch_range = epoch_range
        if metrics is not None:
            self.metrics = metrics
        if custom_pruner is not None:
            self.custom_pruner = custom_pruner

    def init_prune_config(self):
        pruner = Pruner()
        self.prune_config.usr_cfg.pruning.approach.weight_compression['pruners'] = [pruner]

    @property
    def custom_pruner(self):
        return self.prune_config.usr_cfg.pruning.approach.weight_compression.pruners

    @custom_pruner.setter
    def custom_pruner(self, pruner):
        self.prune_config.usr_cfg.pruning.approach.weight_compression.pruners = [pruner]

    @property
    def approach(self):
        return self.prune_config.usr_cfg.pruning.approach.weight_compression.pruners[0].prune_type

    @approach.setter
    def approach(self, approach):
        assert approach.upper() in SUPPORTED_PRUNING_MODE, \
            "pruning approach must be in {}!".format(
                [mode.lower() for mode in SUPPORTED_PRUNING_MODE]
            )
        self.prune_config.usr_cfg.pruning.approach.weight_compression.pruners[0].prune_type = \
            PruningMode[approach.upper()].value

    @property
    def target_sparsity_ratio(self):
        return self.prune_config.usr_cfg.pruning.approach.weight_compression.target_sparsity

    @target_sparsity_ratio.setter
    def target_sparsity_ratio(self, target_sparsity_ratio):
        self.prune_config.usr_cfg.pruning.approach.weight_compression.target_sparsity = \
            target_sparsity_ratio

    @property
    def epoch_range(self):
        return [self.prune_config.usr_cfg.pruning.approach.weight_compression.start_epoch,
                self.prune_config.usr_cfg.pruning.approach.weight_compression.end_epoch]

    @epoch_range.setter
    def epoch_range(self, epoch_range):
        assert isinstance(epoch_range, list) and len(epoch_range) == 2, \
          "You should set epoch_range like [a,b] format to match the pruning start and end epoch."
        self.prune_config.usr_cfg.pruning.approach.weight_compression.start_epoch = epoch_range[0]
        self.prune_config.usr_cfg.pruning.approach.weight_compression.end_epoch = epoch_range[1]

    @property
    def framework(self):
        return self.prune_config.usr_cfg.model.framework

    @framework.setter
    def framework(self, framework):
        assert framework.lower() in ["pytorch"], \
            "framework: {} is not support!".format(framework)
        self.prune_config.usr_cfg.model.framework = framework.lower()

    @property
    def metrics(self):
        return self._metrics

    @metrics.setter
    def metrics(self, metrics: Union[Metric, List]):
        self._metrics = metrics


class DistillationConfig(object):
    def __init__(
        self,
        framework: str = "pytorch",
        criterion: Criterion = None,
        metrics: Union[List, Metric] = None,
    ):
        super().__init__()
        self.distill_config = Distillation_Conf()
        self.framework = framework
        if criterion is not None:
            self.criterion = criterion
        if metrics is not None:
            self.metrics = metrics

    @property
    def framework(self):
        return self.distill_config.usr_cfg.model.framework

    @framework.setter
    def framework(self, framework):
        assert framework in ["pytorch"], \
            "framework: {} is not support!".format(framework)
        self.distill_config.usr_cfg.model.framework = framework

    @property
    def criterion(self):
        return self.distill_config.usr_cfg.distillation.train.criterion

    @criterion.setter
    def criterion(self, criterion: Criterion):
        assert criterion.name.upper() in SUPPORTED_DISTILLATION_CRITERION_MODE, \
            "The criterion name must be in ['KnowledgeLoss', 'IntermediateLayersLoss']"
        if criterion.name.upper() == DistillationCriterionMode.KNOWLEDGELOSS.name:
            assert criterion.temperature is not None, \
                "Please pass the temperature to Criterion.temperature!"
            assert criterion.loss_types is not None, \
                "Please pass the loss_types to Criterion.loss_types!"
            assert criterion.loss_weight_ratio is not None, \
                "Please pass the loss_weight_ratio to Criterion.loss_weight_ratio!"
            self.distill_config.usr_cfg.distillation.train.criterion = {
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
            self.distill_config.usr_cfg.distillation.train.criterion = {
                DistillationCriterionMode.INTERMEDIATELAYERSLOSS.value: {
                    "layer_mappings": criterion.layer_mappings,
                    "loss_types": criterion.loss_types,
                    "loss_weights": criterion.loss_weight_ratio,
                    "add_origin_loss": criterion.add_origin_loss
                }
            }

    @property
    def metrics(self):
        return self._metrics

    @metrics.setter
    def metrics(self, metrics):
        self._metrics = metrics


class ProviderConfig(object):
    def __init__(self):
        super().__init__()
        self.quantization = QuantizationConfig()
        self.pruning = PruningConfig()
        self.distillation = DistillationConfig()
        self._nncf_config = None

    @property
    def nncf_config(self):
        return self._nncf_config

    @nncf_config.setter
    def nncf_config(self, nncf_config):
        self._nncf_config = nncf_config
