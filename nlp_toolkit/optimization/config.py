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

import nlp_toolkit
import os
import yaml
from enum import Enum
from functools import reduce
from neural_compressor.conf.config import Quantization_Conf
from neural_compressor.conf.config import Pruning_Conf
from neural_compressor.conf.config import Distillation_Conf
from neural_compressor.utils import logger
from transformers.file_utils import cached_path, hf_bucket_url
from typing import Any, Optional, Dict
from xmlrpc.client import boolean


CONFIG_NAME = "best_configure.yaml"
WEIGHTS_NAME = "pytorch_model.bin"
QUANTIZED_WEIGHTS_NAME = "best_model_weights.pt"


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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.quant_config = Quantization_Conf()

    @property
    def approach(self):
        return self.quant_config.usr_cfg.quantization.approach

    @approach.setter
    def approach(self, approach):
        assert approach in \
            ["post_training_static_quant", "quant_aware_training",
             "post_training_dynamic_quant"], \
            "quantization approach: {} is not support!".format(approach)
        self.quant_config.usr_cfg.quantization.approach = approach

    @property
    def metric_tolerance(self):
        return self.quant_config.usr_cfg.tuning.accuracy_criterion

    @metric_tolerance.setter
    def metric_tolerance(self, tolerance):
        assert isinstance(tolerance, dict), \
            "metric_tolerance should be a dictionary like:{'relative': 0.01} or {'absolute': 0.1}"
        if not isinstance(tolerance[list(tolerance.keys())[0]], (int, float)):
            raise TypeError(f"Supported type for performance tolerance are int and float, \
                              got {type(tolerance[list(tolerance.keys())[0]])}")
        if list(tolerance.keys())[0] == "relative" \
           and not -1 < tolerance[list(tolerance.keys())[0]] < 1:
            raise ValueError("Relative performance tolerance must not be <=-1 or >=1.")
        if list(tolerance.keys())[0] in self.quant_config.usr_cfg.tuning.accuracy_criterion:
            setattr(
                self.quant_config.usr_cfg.tuning.accuracy_criterion, list(tolerance.keys())[0],
                tolerance[list(tolerance.keys())[0]])
        else:
            if "relative" in self.quant_config.usr_cfg.tuning.accuracy_criterion:
                del self.quant_config.usr_cfg.tuning.accuracy_criterion["relative"]
            elif "absolute" in self.quant_config.usr_cfg.tuning.accuracy_criterion:
                del self.quant_config.usr_cfg.tuning.accuracy_criterion["absolute"]
            setattr(
                self.quant_config.usr_cfg.tuning.accuracy_criterion, list(tolerance.keys())[0],
                tolerance[list(tolerance.keys())[0]])

    @property
    def framework(self):
        return self.quant_config.usr_cfg.model.framework

    @framework.setter
    def framework(self, framework):
        assert framework in ["pytorch", "pytorch_fx"], \
            "framework: {} is not support!".format(framework)
        self.quant_config.usr_cfg.model.framework = framework

    @property
    def strategy(self):
        return self.quant_config.usr_cfg.tuning.strategy.name

    @strategy.setter
    def strategy(self, strategy):
        assert strategy in ["basic", "bayesian", "mse"], \
            "strategy: {} is not support!".format(strategy)
        self.quant_config.usr_cfg.tuning.strategy.name = strategy

    @property
    def objective(self):
        return self.quant_config.usr_cfg.tuning.objective

    @objective.setter
    def objective(self, objective):
        assert objective in ["performance", "modelsize", "footprint"], \
            "objective: {} is not support!".format(objective)
        self.quant_config.usr_cfg.tuning.objective = objective

    @property
    def multi_objective(self):
        return self.quant_config.usr_cfg.tuning.multi_objective.objective

    @multi_objective.setter
    def multi_objective(self, multi_objective):
        for objective in multi_objective:
            assert objective in ["performance", "modelsize", "footprint"], \
                "objective: {} is not support!".format(objective)
        self.quant_config.usr_cfg.tuning.multi_objective.objective = multi_objective

    @property
    def multi_objective_weight(self):
        return self.quant_config.usr_cfg.tuning.multi_objective.weight

    @multi_objective_weight.setter
    def multi_objective_weight(self, multi_objective_weight):
        self.quant_config.usr_cfg.tuning.multi_objective.weights = multi_objective_weight

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
    def save_path(self):
        return self.quant_config.usr_cfg.tuning.workspace.path

    @save_path.setter
    def save_path(self, path):
        assert isinstance(path, str), "save_path should be a string of directory!"
        self.quant_config.usr_cfg.tuning.workspace.path = path

    @property
    def resume_path(self):
        return self.quant_config.usr_cfg.tuning.workspace.resume

    @resume_path.setter
    def resume_path(self, path):
        assert isinstance(path, str), "resume_path should be a string of directory!"
        self.quant_config.usr_cfg.tuning.workspace.resume = path

    @property
    def metrics(self):
        return self.quant_config.usr_cfg.evaluation.accuracy

    @metrics.setter
    def metrics(self, metrics):
        self.quant_config.usr_cfg.evaluation.accuracy = metrics

class PruningConfig(object):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prune_config = Pruning_Conf()
        self.init_prune_config()

    def init_prune_config(self):
        from neural_compressor.conf.config import Pruner
        Pruner = Pruner()
        self.prune_config.usr_cfg.pruning.approach.weight_compression['pruners'] = [Pruner]

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
        assert approach in \
            ["basic_magnitude"], \
            "pruning approach: {} is not support!".format(approach)
        self.prune_config.usr_cfg.pruning.approach.weight_compression.pruners[0].prune_type = approach

    @property
    def target_sparsity(self):
        return self.prune_config.usr_cfg.pruning.approach.weight_compression.target_sparsity

    @target_sparsity.setter
    def target_sparsity(self, target_sparsity):
        self.prune_config.usr_cfg.pruning.approach.weight_compression.target_sparsity = target_sparsity

    @property
    def epoch_range(self):
        return [self.prune_config.usr_cfg.pruning.approach.weight_compression.start_epoch, self.prune_config.usr_cfg.pruning.approach.weight_compression.end_epoch]

    @epoch_range.setter
    def epoch_range(self, epoch_range):
        assert isinstance(epoch_range,list) and len(epoch_range) == 2, "you should set epoch_range like [a,b] format to match the pruning start and end epoch."
        self.prune_config.usr_cfg.pruning.approach.weight_compression.start_epoch = epoch_range[0]
        self.prune_config.usr_cfg.pruning.approach.weight_compression.end_epoch = epoch_range[1]

    @property
    def framework(self):
        return self.prune_config.usr_cfg.model.framework

    @framework.setter
    def framework(self, framework):
        assert framework in ["pytorch"], \
            "framework: {} is not support!".format(framework)
        self.prune_config.usr_cfg.model.framework = framework

    @property
    def metrics(self):
        return self.prune_config.usr_cfg.evaluation.accuracy

    @metrics.setter
    def metrics(self, metrics):
        self.prune_config.usr_cfg.evaluation.accuracy = metrics

class DistillationConfig(object):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.distill_config = Distillation_Conf()

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
    def criterion(self, criterion):
        self.distill_config.usr_cfg.distillation.train.criterion = criterion

    @property
    def metrics(self):
        return self.distill_config.usr_cfg.evaluation.accuracy

    @metrics.setter
    def metrics(self, metrics):
        self.distill_config.usr_cfg.evaluation.accuracy = metrics

class OptimizeConfig(object):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.quantization = QuantizationConfig()
        self.pruning = PruningConfig()
        self.distillation = DistillationConfig()
        self._provider = Provider.INC.value
        self._provider_arguments = None
        self._opt_cfg = {
            "quantization": self.quantization,
            "pruning": self.pruning,
            "distillation": self.distillation
        }

    @property
    def provider(self):
        return self._provider

    @provider.setter
    def provider(self, provider: str):
        self._provider = provider

    @property
    def provider_arguments(self):
        return self._provider_arguments

    @provider_arguments.setter
    def provider_arguments(self, provider_arguments):
        self._provider_arguments = provider_arguments

    def parse_nncf_arguments(self):
        assert self._provider_arguments is not None, "Please pass arguments to trainer.provider_arguments"
        assert isinstance(self._provider_arguments, Dict), "provider_arguments must be a dictionary type"
        assert "nncf_config" in self._provider_arguments.keys(), "provider_arguments must be included nncf_config"

    def parse_inc_arguments(self):
        if self._provider_arguments is not None:
            assert isinstance(self._provider_arguments, Dict), "provider_arguments must be a dictionary type"
            framework = "pytorch"
            for opt_cfg in self._provider_arguments:
                if opt_cfg == "framework":
                    framework = self._provider_arguments[opt_cfg]
                    continue
                if opt_cfg == "quantization":
                    for cfg in self._provider_arguments[opt_cfg]:
                        if cfg == "approach":
                            self.quantization.approach = \
                                nlp_toolkit.QuantizationMode[self._provider_arguments[opt_cfg][cfg]].value
                        if cfg == "strategy":
                            self.quantization.strategy = self._provider_arguments[opt_cfg][cfg]
                        if cfg == "timeout":
                            self.quantization.timeout = self._provider_arguments[opt_cfg][cfg]
                        if cfg == "max_trials":
                            self.quantization.max_trials = self._provider_arguments[opt_cfg][cfg]
                        if cfg == "save_path":
                            self.quantization.save_path = self._provider_arguments[opt_cfg][cfg]
                        if cfg == "criterion":
                            self.quantization.criterion = self._provider_arguments[opt_cfg][cfg]
                        if cfg == "objects":
                            self.quantization.objects = self._provider_arguments[opt_cfg][cfg]
                        if cfg == "metrics":
                            self.metrics = self._provider_arguments[opt_cfg][cfg]

                if opt_cfg == "pruning":
                    for cfg in self._provider_arguments[opt_cfg]:
                        if cfg == "approach":
                            self.pruning.approach = self._provider_arguments[opt_cfg][cfg]
                        if cfg == "custom_pruner":
                            self.pruning.custom_pruner = self._provider_arguments[opt_cfg][cfg]
                        if cfg == "target_sparsity":
                            self.pruning.target_sparsity = self._provider_arguments[opt_cfg][cfg]
                        if cfg == "epoch_range":
                            self.pruning.epoch_range = self._provider_arguments[opt_cfg][cfg]
                        if cfg == "metrics":
                            self.pruning.metrics = self._provider_arguments[opt_cfg][cfg]
                        if cfg == "timeout":
                            self.pruning.timeout = self._provider_arguments[opt_cfg][cfg]
                        if cfg == "max_trials":
                            self.pruning.max_trials = self._provider_arguments[opt_cfg][cfg]
                        if cfg == "save_path":
                            self.pruning.save_path = self._provider_arguments[opt_cfg][cfg]
                        if cfg == "criterion":
                            self.pruning.criterion = self._provider_arguments[opt_cfg][cfg]
                        if cfg == "objects":
                            self.pruning.objects = self._provider_arguments[opt_cfg][cfg]

                if opt_cfg == "distillation":
                    for cfg in self._provider_arguments[opt_cfg]:
                        if cfg == "metrics":
                            self.pruning.metrics = self._provider_arguments[opt_cfg][cfg]
                        if cfg == "timeout":
                            self.pruning.timeout = self._provider_arguments[opt_cfg][cfg]
                        if cfg == "max_trials":
                            self.pruning.max_trials = self._provider_arguments[opt_cfg][cfg]
                        if cfg == "save_path":
                            self.pruning.save_path = self._provider_arguments[opt_cfg][cfg]
                        if cfg == "criterion":
                            self.pruning.criterion = self._provider_arguments[opt_cfg][cfg]
                        if cfg == "objects":
                            self.pruning.objects = self._provider_arguments[opt_cfg][cfg]

            if framework == "pytorch" and \
              self.quantization.approach != nlp_toolkit.QuantizationMode.PostTrainingDynamic.value:
                self.quantization.quant_config.usr_cfg.model.framework = "pytorch_fx"
            else:
                self.quantization.quant_config.usr_cfg.model.framework = framework
            self.pruning.prune_config.usr_cfg.model.framework = framework
            self.distillation.distill_config.usr_cfg.model.framework = framework
