import os
import yaml
from enum import Enum
from functools import reduce
from neural_compressor.conf.config import (
    Distillation_Conf, Pruner, Pruning_Conf, Quantization_Conf
)
from neural_compressor.conf.dotdict import DotDict
from neural_compressor.utils import logger
from nlp_toolkit.optimization.metrics import Metric
from nlp_toolkit.optimization.objectives import Objective, performance
from nlp_toolkit.optimization.quantization import QuantizationMode, SUPPORTED_QUANT_MODE
from nlp_toolkit.optimization.distillation import (
    Criterion, DistillationCriterionMode, SUPPORTED_DISTILLATION_CRITERION_MODE
)
from nlp_toolkit.optimization.utils.utility import LazyImport
from transformers.file_utils import cached_path, hf_bucket_url
from typing import Any, List, Optional, Union
from xmlrpc.client import boolean

nncf = LazyImport("nncf")


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
        approach: str = "PostTrainingStatic",
        timeout: int = 0,
        max_trials: int = 100,
        metrics: Union[Metric, List] = None,
        objectives: Union[Objective, List] = performance,
    ):
        super().__init__()
        self.inc_config = Quantization_Conf()
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
        return self.inc_config.usr_cfg.quantization.approach

    @approach.setter
    def approach(self, approach):
        approach = approach.upper()
        assert approach in SUPPORTED_QUANT_MODE, \
            f"quantization approach: {approach} is not support!" + \
            "PostTrainingStatic, PostTrainingDynamic and QuantizationAwareTraining are supported!"
        self.inc_config.usr_cfg.quantization.approach = QuantizationMode[approach].value

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
            else:
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
        return self.inc_config.usr_cfg.model.framework

    @framework.setter
    def framework(self, framework):
        assert framework in ["pytorch", "pytorch_fx"], \
            "framework: {} is not support!".format(framework)
        self.inc_config.usr_cfg.model.framework = framework

    @property
    def objectives(self):
        return self._objectives

    @objectives.setter
    def objectives(self, objectives: Union[List, Objective]):
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
        return self.inc_config.usr_cfg.tuning.strategy.name

    @strategy.setter
    def strategy(self, strategy):
        assert strategy in ["basic", "bayesian", "mse"], \
            "strategy: {} is not support!".format(strategy)
        self.inc_config.usr_cfg.tuning.strategy.name = strategy

    @property
    def timeout(self):
        return self.inc_config.usr_cfg.tuning.exit_policy.timeout

    @timeout.setter
    def timeout(self, timeout):
        assert isinstance(timeout, int), "timeout should be integer!"
        self.inc_config.usr_cfg.tuning.exit_policy.timeout = timeout

    @property
    def max_trials(self):
        return self.inc_config.usr_cfg.tuning.exit_policy.max_trials

    @max_trials.setter
    def max_trials(self, max_trials):
        assert isinstance(max_trials, int), "max_trials should be integer!"
        self.inc_config.usr_cfg.tuning.exit_policy.max_trials = max_trials

    @property
    def performance_only(self):
        return self.inc_config.usr_cfg.tuning.exit_policy.performance_only

    @performance_only.setter
    def performance_only(self, performance_only):
        assert isinstance(performance_only, boolean), "performance_only should be boolean!"
        self.inc_config.usr_cfg.tuning.exit_policy.performance_only = performance_only

    @property
    def random_seed(self):
        return self.inc_config.usr_cfg.tuning.random_seed

    @random_seed.setter
    def random_seed(self, random_seed):
        assert isinstance(random_seed, int), "random_seed should be integer!"
        self.inc_config.usr_cfg.tuning.random_seed = random_seed

    @property
    def tensorboard(self):
        return self.inc_config.usr_cfg.tuning.tensorboard

    @tensorboard.setter
    def tensorboard(self, tensorboard):
        assert isinstance(tensorboard, boolean), "tensorboard should be boolean!"
        self.inc_config.usr_cfg.tuning.tensorboard = tensorboard

    @property
    def output_dir(self):
        return self.inc_config.usr_cfg.tuning.workspace.path

    @output_dir.setter
    def output_dir(self, path):
        assert isinstance(path, str), "save_path should be a string of directory!"
        self.inc_config.usr_cfg.tuning.workspace.path = path

    @property
    def resume_path(self):
        return self.inc_config.usr_cfg.tuning.workspace.resume

    @resume_path.setter
    def resume_path(self, path):
        assert isinstance(path, str), "resume_path should be a string of directory!"
        self.inc_config.usr_cfg.tuning.workspace.resume = path


class PruningConfig(object):
    def __init__(
        self,
        framework: str = "pytorch",
        epoch_range: List = [0, 4],
        initial_sparsity_ratio: float=0.0,
        target_sparsity_ratio: float = 0.97,
        metrics: Metric = None,
        pruner: Union[List, Pruner] = None,
    ):
        super().__init__()
        self.inc_config = Pruning_Conf()
        self.framework = framework
        if initial_sparsity_ratio is not None:
            self.initial_sparsity_ratio = initial_sparsity_ratio
        if target_sparsity_ratio is not None:
            self.target_sparsity_ratio = target_sparsity_ratio
        if epoch_range is not None:
            self.epoch_range = epoch_range
        if metrics is not None:
            self.metrics = metrics
        if pruner is not None:
            self.pruner = pruner
        else:
            self.init_prune_config()


    def init_prune_config(self):
        pruner = Pruner()
        self.inc_config.usr_cfg.pruning.approach.weight_compression['pruners'] = [pruner]

    @property
    def pruner(self):
        return self.inc_config.usr_cfg.pruning.approach.weight_compression.pruners

    @pruner.setter
    def pruner(self, pruner):
        if isinstance(pruner, list):
            self.inc_config.usr_cfg.pruning.approach.weight_compression.pruners = pruner
        else:
            self.inc_config.usr_cfg.pruning.approach.weight_compression.pruners = [pruner]

    @property
    def target_sparsity_ratio(self):
        return self.inc_config.usr_cfg.pruning.approach.weight_compression.target_sparsity

    @target_sparsity_ratio.setter
    def target_sparsity_ratio(self, target_sparsity_ratio):
        self.inc_config.usr_cfg.pruning.approach.weight_compression.target_sparsity = \
            target_sparsity_ratio

    @property
    def initial_sparsity_ratio(self):
        return self.inc_config.usr_cfg.pruning.approach.weight_compression.initial_sparsity

    @initial_sparsity_ratio.setter
    def initial_sparsity_ratio(self, initial_sparsity_ratio):
        self.inc_config.usr_cfg.pruning.approach.weight_compression.initial_sparsity = \
            initial_sparsity_ratio

    @property
    def epoch_range(self):
        return [self.inc_config.usr_cfg.pruning.approach.weight_compression.start_epoch,
                self.inc_config.usr_cfg.pruning.approach.weight_compression.end_epoch]

    @epoch_range.setter
    def epoch_range(self, epoch_range):
        assert isinstance(epoch_range, list) and len(epoch_range) == 2, \
          "You should set epoch_range like [a,b] format to match the pruning start and end epoch."
        self.inc_config.usr_cfg.pruning.approach.weight_compression.start_epoch = epoch_range[0]
        self.inc_config.usr_cfg.pruning.approach.weight_compression.end_epoch = epoch_range[1]

    @property
    def framework(self):
        return self.inc_config.usr_cfg.model.framework

    @framework.setter
    def framework(self, framework):
        assert framework.lower() in ["pytorch"], \
            "framework: {} is not support!".format(framework)
        self.inc_config.usr_cfg.model.framework = framework.lower()

    @property
    def metrics(self):
        return self._metrics

    @metrics.setter
    def metrics(self, metrics: Metric):
        self._metrics = metrics
        assert isinstance(metrics, Metric), \
            "metric should be a Metric calss!"


class DistillationConfig(object):
    def __init__(
        self,
        framework: str = "pytorch",
        criterion: Criterion = None,
        metrics: Metric = None,
    ):
        super().__init__()
        self.inc_config = Distillation_Conf()
        self.framework = framework
        if criterion is not None:
            self.criterion = criterion
        if metrics is not None:
            self.metrics = metrics

    @property
    def framework(self):
        return self.inc_config.usr_cfg.model.framework

    @framework.setter
    def framework(self, framework):
        assert framework in ["pytorch"], \
            "framework: {} is not support!".format(framework)
        self.inc_config.usr_cfg.model.framework = framework

    @property
    def criterion(self):
        return self.inc_config.usr_cfg.distillation.train.criterion

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
        return self._metrics

    @metrics.setter
    def metrics(self, metrics):
        assert isinstance(metrics, Metric), \
            "metric should be a Metric calss!"
        self._metrics = metrics


class FlashDistillationConfig(object):
    def __init__(
        self,
        block_names: list = [],
        layer_mappings_for_knowledge_transfer: list = [],
        loss_types: list = [],
        loss_weights: list = [],
        add_origin_loss: list = [],
        train_steps: list = [],
    ):
        super().__init__()
        self.block_names = block_names
        self.layer_mappings_for_knowledge_transfer = layer_mappings_for_knowledge_transfer
        self.loss_types = loss_types
        self.loss_weights = loss_weights
        self.add_origin_loss = add_origin_loss
        self.train_steps = train_steps


class AutoDistillationConfig(object):
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
        return self.config.auto_distillation.flash_distillation.knowledge_transfer

    @knowledge_transfer.setter
    def knowledge_transfer(self, knowledge_transfer):
        if knowledge_transfer is None:
            knowledge_transfer = FlashDistillationConfig()
        for k, v in knowledge_transfer.__dict__.items():
            if v:
                self.config.auto_distillation.flash_distillation.knowledge_transfer[k] = v

    @property
    def regular_distillation(self):
        return self.config.auto_distillation.flash_distillation.regular_distillation

    @regular_distillation.setter
    def regular_distillation(self, regular_distillation):
        if regular_distillation is None:
            regular_distillation = FlashDistillationConfig()
        for k, v in regular_distillation.__dict__.items():
            if v:
                self.config.auto_distillation.flash_distillation.regular_distillation[k] = v

    @property
    def framework(self):
        return self.config.model.framework

    @framework.setter
    def framework(self, framework):
        assert framework in ["pytorch"], \
            "framework: {} is not support!".format(framework)
        self.config.model.framework = framework

    @property
    def search_space(self):
        return self.config.auto_distillation.search.search_space

    @search_space.setter
    def search_space(self, search_space: dict):
        self.config.auto_distillation.search.search_space = search_space

    @property
    def search_algorithm(self):
        return self.config.auto_distillation.search.search_algorithm

    @search_algorithm.setter
    def search_algorithm(self, search_algorithm: str):
        self.config.auto_distillation.search.search_algorithm = search_algorithm

    @property
    def max_trials(self):
        return self.config.auto_distillation.search.max_trials

    @max_trials.setter
    def max_trials(self, max_trials: int):
        self.config.auto_distillation.search.max_trials = max_trials

    @property
    def seed(self):
        return self.config.auto_distillation.search.seed

    @seed.setter
    def seed(self, seed: int):
        self.config.auto_distillation.search.seed = seed

    @property
    def metrics(self):
        return self._metrics

    @metrics.setter
    def metrics(self, metrics: Union[List, Metric]):
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


# pylint: disable=E0401
class NncfConfig(object):
    def __init__(
        self,
        nncf_config,
        distributed: bool = False,
        to_onnx: bool = False,
        metrics: Union[List, Metric] = None,
    ):
        super().__init__()
        from nncf import NNCFConfig
        assert isinstance(nncf_config, NNCFConfig)
        self.nncf_config = nncf_config
        if metrics is not None:
            self._metrics = metrics
        self._distributed = distributed
        self._to_onnx = to_onnx


    @property
    def distributed(self):
        return self._distributed

    @distributed.setter
    def distributed(self, distributed):
        self._distributed = distributed

    @property
    def to_onnx(self):
        return self._to_onnx

    @to_onnx.setter
    def to_onnx(self, to_onnx):
        self._to_onnx = to_onnx

    @property
    def metrics(self):
        return self._metrics

    @metrics.setter
    def metrics(self, metrics):
        self._metrics = metrics

    @property
    def nncf_config(self):
        return self._nncf_config

    @nncf_config.setter
    def nncf_config(self, nncf_config):
        self._nncf_config = nncf_config
