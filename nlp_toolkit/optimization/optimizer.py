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

import logging
import torch

from neural_compressor.experimental import common, Component, Distillation
from neural_compressor.experimental.scheduler import Scheduler
from nlp_toolkit import OptimizeConfig, Provider
from transformers import PreTrainedModel
from typing import Callable, Optional, Union, List


logger = logging.getLogger(__name__)


class OptimizerPipeline:
    def __init__(
        self,
        model: Union[PreTrainedModel, torch.nn.Module],
        components: Optional[List[Component]] = [],
        one_shot_optimization: Optional[bool] = False,
        eval_func: Optional[Callable] = None,
        train_func: Optional[Callable] = None,
    ):
        """
        Args:
            model (:obj:`Union[PreTrainedModel, torch.nn.Module]`):
                Model to quantize and/or prune.
            components (List[:obj:`Component`], `optional`):
                List of Component objects which contains Quantization, 
                Pruning, Distillation objects.
            one_shot_optimization (bool, `optional`):
                Whether to do multiple compression processes together.
            eval_func (:obj:`Callable`, `optional`):
                Evaluation function to evaluate the tuning objective.
            train_func (:obj:`Callable`, `optional`):
                Training function which will be combined with pruning.
        """

        if len(components) == 0:
            raise RuntimeError("`NLPOptimizer` requires at least one `Quantization`, "
                               "`Pruning` or `Distillation` object")

        self.scheduler = Scheduler()
        self.scheduler.model = common.Model(model)

        if one_shot_optimization and len(components) > 1:
            agent = self.scheduler.combine(*components)
            agent.train_func = train_func
            agent.eval_func = eval_func
            for component in components:
                if isinstance(component, Distillation) and hasattr(component, 'criterion'):
                    agent.criterion = component.criterion
            print(agent)
            self.scheduler.append(agent)
        else:
            self.scheduler.append(*components)

    def fit(self):
        opt_model = self.scheduler()
        return opt_model


class NoTrainerOptimizer(OptimizeConfig):
    def __init__(
        self,
        model: Union[PreTrainedModel, torch.nn.Module],
        output_dir: Optional[str] = "saved_results",
    ):
        """
        Args:
            model (:obj:`Union[PreTrainedModel, torch.nn.Module]`):
                FP32 model specified for low precision tuning.
            eval_func (:obj:`Callable`, `optional`):
                Evaluation function to evaluate the tuning objective.
            train_func (:obj:`Callable`, `optional`):
                Training function for quantization aware training approach.
            calib_dataloader (:obj:`DataLoader`, `optional`):
                DataLoader for post-training quantization calibration.

        Returns:
            quantizer: NLPQuantizer object.
        """

        self.model = model
        self.teacher_model = None
        self._eval_func = None
        self._train_func = None
        self._calib_dataloader = None
        self.output_dir = output_dir

    @property
    def eval_func(self):
        return self._eval_func

    @property
    def train_func(self):
        return self._train_func

    @property
    def calib_dataloader(self):
        return self._calib_dataloader

    @eval_func.setter
    def eval_func(self, func: Callable):
        self._eval_func = func

    @train_func.setter
    def train_func(self, func: Callable):
        self._train_func = func

    @calib_dataloader.setter
    def calib_dataloader(self, dataloader):
        self._calib_dataloader = dataloader

    def _init_quantize(self):
        from .quantization import QuantizationMode
        from neural_compressor.experimental import Quantization, common
        if self.quant_config.usr_cfg.quantization.approach == \
          QuantizationMode.PostTrainingDynamic.value:
            self.quant_config.usr_cfg.model.framework = "pytorch"
        else:
            self.quant_config.usr_cfg.model.framework = "pytorch_fx"

        quantizer = Quantization(self.quant_config)
        quantizer.model = common.Model(self.model)

        assert self._eval_func is not None, "eval_func must be provided for quantization!"

        if self._eval_func is not None:
            quantizer.eval_func = self._eval_func

        if self.quant_config.usr_cfg.quantization.approach == \
          QuantizationMode.PostTrainingStatic.value:
            assert self._calib_dataloader is not None, \
                "calib_dataloader must be provided for post-training quantization."
            quantizer.calib_dataloader = self._calib_dataloader
        elif self.quant_config.usr_cfg.quantization.approach == \
          QuantizationMode.QuantizationAwareTraining.value:
            assert self._train_func is not None, \
                "train_func must be provided for quantization aware training."
            quantizer.q_func = self._train_func

        self.quantizer = quantizer
        return quantizer

    def _nncf_quantize(self):
        from nncf import create_compressed_model
        assert self._provider_arguments is not None, "Please pass arguments to trainer.provider_arguments!"
        compression_state = None

        if os.path.isfile(self._nncf_compression_state_file):
            compression_state = torch.load(self._nncf_compression_state_file)
        else:
            compression_state = None
        compression_algo_controller, model = create_compressed_model(
            self.model, self._provider_arguments["nncf_config"], compression_state=compression_state
        )

        self.compression_ctrl = \
            compression_algo_controller.distributed() if self._provider_arguments["distributed"] else compression_algo_controller

    def quantize(
        self,
        metric_name: Optional[str] = None,
        eval_func: Optional[Callable] = None,
        train_func: Optional[Callable] = None,
        calib_dataloader=None,
    ):
        if self._provider == "nncf":
            return self._nncf_quantize()
        if metric_name is not None:
            self.metric_name = metric_name
        if eval_func is not None:
            self._eval_func = eval_func
        if train_func is not None:
            self._train_func = train_func
        if calib_dataloader is not None:
            self._calib_dataloader = calib_dataloader
        quantizer = self._init_quantizer()
        opt_model = quantizer.fit()
        opt_model.save(self.args.output_dir)
        logger.info(
            "quantized model and configure file have saved to {}".format(self.args.output_dir)
        )
        return opt_model.model

    def _init_pruner(self):
        from .pruning import PruningMode
        from neural_compressor.experimental import Pruning, common

        self.prune_config.usr_cfg.model.framework = "pytorch"
        pruning_start_epoch = self.prune_config.usr_cfg.pruning.approach.weight_compression.start_epoch
        pruning_end_epoch = self.prune_config.usr_cfg.pruning.approach.weight_compression.end_epoch

        if pruning_start_epoch > self.args.num_train_epochs - 1:
            logger.warning(
                f"Pruning end epoch {pruning_start_epoch} is higher than the total number of training epoch "
                f"{self.args.num_train_epochs}. No pruning will be applied."
            )

        if pruning_end_epoch > self.args.num_train_epochs - 1:
            logger.warning(
                f"Pruning end epoch {pruning_end_epoch} is higher than the total number of training epoch "
                f"{self.args.num_train_epochs}. The target sparsity will not be reached."
            )

        pruner = Pruning(self.prune_config)
        pruner.model = common.Model(self.model)

        assert self._eval_func is not None, "eval_func must be provided for pruning!"

        if self._eval_func is not None:
            pruner.eval_func = self._eval_func

        assert self._train_func is not None, \
                "train_func must be provided for pruning."
        pruner.pruning_func = self._train_func
        self.pruner = pruner
        return pruner

    def prune(
        self,
        metric_name: Optional[str] = None,
        eval_func: Optional[Callable] = None,
        train_func: Optional[Callable] = None,
        calib_dataloader=None,
    ):
        if metric_name is not None:
            self.metric_name = metric_name
        if eval_func is not None:
            self._eval_func = eval_func
        if train_func is not None:
            self._train_func = train_func

        pruner = self._init_pruner()
        opt_model = pruner.fit()

        return opt_model.model

    def _init_distiller(self):
        from .distillation import DistillationMode
        from neural_compressor.experimental import Distillation, common
        assert self.teacher_model is not None, \
                    "teacher model must be provided for distillation."
        self.distill_config.usr_cfg.model.framework = "pytorch"
        distiller = Distillation(self.distill_config)
        distiller.model = common.Model(self.model)
        distiller.teacher_model = common.Model(self.teacher_model)

        assert self._eval_func is not None, "eval_func must be provided for distillation!"

        if self._eval_func is not None:
            distiller.eval_func = self._eval_func

        assert self._train_func is not None, \
                "train_func must be provided for distillation."
        distiller.train_func = self._train_func
        distiller.create_criterion()
        self.distiller = distiller
        return distiller

    def distillation(
        self,
        teacher_model: Union[PreTrainedModel, torch.nn.Module],
        metric_name: Optional[str] = None,
        eval_func: Optional[Callable] = None,
        train_func: Optional[Callable] = None,
    ):
        if metric_name is not None:
            self.metric_name = metric_name
        if eval_func is not None:
            self._eval_func = eval_func
        if train_func is not None:
            self._train_func = train_func

        self.teacher_model = teacher_model
        distiller = self._init_distiller()
        opt_model = distiller.fit()

        return opt_model.model