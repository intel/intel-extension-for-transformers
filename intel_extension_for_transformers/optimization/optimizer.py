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
"""Optimization: provides the orchestrate optimizer for Pytorch."""
import logging
import os
import shlex

from neural_compressor.experimental import(
    common,
    Component,
    Distillation,
    Quantization,
    Pruning,
)
from neural_compressor.experimental.scheduler import Scheduler
from intel_extension_for_transformers.optimization import(
    DistillationConfig,
    Provider,
    QuantizationConfig,
    PruningConfig
)
from intel_extension_for_transformers.optimization.utils.utility import LazyImport
from intel_extension_for_transformers.optimization.quantization import QuantizationMode
from transformers import PreTrainedModel, PretrainedConfig
from transformers.file_utils import WEIGHTS_NAME
from typing import Callable, Optional, Union, List

torch = LazyImport("torch")

logger = logging.getLogger(__name__)


class Orchestrate_optimizer:
    """Orchestrate_optimizer aggregates and orchestrates components such as Quantization, Pruning and Distillation."""
    def __init__(
        self,
        model,
        components: Optional[List[Component]] = [],
        eval_func: Optional[Callable] = None,
        train_func: Optional[Callable] = None,
        output_dir: Optional[str] = "saved_results",
    ):
        """Init an orchestrate optimizer.

        Args:
            model: Model to quantize and/or prune.
            components: List of Component objects which contains Quantization, Pruning, Distillation objects.
            eval_func: Evaluation function to evaluate the tuning objective.
            train_func: Training function which will be combined with pruning.
        """
        if len(components) == 0:
            raise RuntimeError("`NLPOptimizer` requires at least one `Quantization`, "
                               "`Pruning` or `Distillation` object")
        self.output_dir = output_dir
        if hasattr(model, 'config') and isinstance(model.config, PretrainedConfig):
            self.model_config = model.config
        self.enable_inc_quant = False
        self.enable_inc_pruning = False
        self.scheduler = Scheduler()
        self.scheduler.model = common.Model(model)

        if len(components) > 1:
            agent = self.scheduler.combine(*components)
            agent.train_func = train_func
            agent.eval_func = eval_func
            for component in components:
                if isinstance(component, Distillation) and hasattr(component, 'criterion'):
                    agent.criterion = component.criterion
                if isinstance(component, Quantization):
                    self.enable_inc_quant = True
                if isinstance(component, Pruning):
                    self.enable_inc_pruning = True
            self.scheduler.append(agent)
        else:
            self.scheduler.append(*components)

    def fit(self):
        """Run the scheduler."""
        self.opt_model = self.scheduler()
        self.save_model(self.output_dir)
        if self.enable_inc_pruning == True:
            stats, sparsity = self.opt_model.report_sparsity()
            logger.info(stats)
            logger.info(sparsity)
        return self.opt_model.model

    def save_model(self, output_dir, tokenizer=None):
        """Save the model and tokenizer in the output directory.

        Args:
            output_dir: the path to save config.json and pytorch_model.bin.
            tokenizer (object, optional): the tokenizer object, use it if you want to 
                                          save tokenizer.json in output_dir. Defaults to None.
        """
        os.makedirs(shlex.quote(output_dir), exist_ok=True)
        torch.save(self.opt_model.quantized_state_dict(), os.path.join(shlex.quote(output_dir), WEIGHTS_NAME))
        if hasattr(self, 'model_config') and isinstance(self.model_config, PretrainedConfig):
            if self.enable_inc_quant == True:
                self.model_config.torch_dtype = "int8"
            self.model_config.save_pretrained(output_dir)
        if tokenizer:   # pragma: no cover
            tokenizer.save_pretrained(output_dir)
        logger.info("orchestrate_optimizations model and configure file have saved to {}".format(
                    output_dir))


class NoTrainerOptimizer:   # pragma: no cover
    """Optimizer without using Trainer."""
    def __init__(
        self,
        model,
        output_dir: Optional[str] = "saved_results",
    ):
        """Init a NoTrainerOptimizer object.

        Args:
            model: FP32 model specified for low precision tuning.
            output_dir: The folder for saving the results.
        """
        self.model = model
        self.teacher_model = None
        self._eval_func = None
        self._train_func = None
        self._calib_func = None
        self._calib_dataloader = None
        self.output_dir = output_dir
        self.quant_config = None
        self.pruning_config = None
        self.distillation_config = None
        self._provider = Provider.INC.value
        self.pruner = None
        self.quantizer = None
        self.distiller = None
        self.in_training = False
        self.enable_inc_quant = False

    @property
    def eval_func(self):
        """Get the evaluation function."""
        return self._eval_func

    @property
    def train_func(self):
        """Get the train function."""
        return self._train_func

    @property
    def calib_func(self):
        """Get the calib function."""
        return self._calib_func

    @property
    def provider(self):
        """Get the provider."""
        return self._provider

    @property
    def calib_dataloader(self):
        """Get the calibration dataloader."""
        return self._calib_dataloader

    @eval_func.setter
    def eval_func(self, func: Callable):
        """Set the evaluation function.
            
        Args:
            func: evaluation function.
        """
        self._eval_func = func

    @train_func.setter
    def train_func(self, func: Callable):
        """Set the train function.
        
        Args:
            func: train function.
        """
        self._train_func = func

    @provider.setter
    def provider(self, provider):
        """Set the provider.
        
        Args:
            provider: optimization provider.
        """
        self._provider = provider

    @calib_dataloader.setter
    def calib_dataloader(self, dataloader):
        """Set the calibration dataloader.
        
        Args:
            dataloader: calibration dataloader.
        """
        self._calib_dataloader = dataloader

    def init_quantizer(
        self,
        quant_config,
        provider: str = Provider.INC.value,
    ):
        """Init a Quantization object with config.
        
        Args:
            quant_config: quantization config.
            provider: define the quantization provider.
        """
        from neural_compressor.experimental import Quantization

        assert isinstance(quant_config, QuantizationConfig), \
            "Please pass QuantizationConfig instance to trainer.quantize!"
        self.quant_config = quant_config
        self.metrics = self.quant_config.metrics
        self._provider = Provider[provider.upper()].value

        if self.quant_config.framework == "pytorch":
            if self.quant_config.approach == \
              QuantizationMode.POSTTRAININGDYNAMIC.value:
                self.quant_config.framework = "pytorch"
            else:
                self.quant_config.framework = "pytorch_fx"

        quantizer = Quantization(self.quant_config.inc_config)
        quantizer.model = common.Model(self.model)

        self.quantizer = quantizer
        return quantizer

    def _inc_quantize(
        self,
        quant_config,
        provider: str = Provider.INC.value,
    ):
        """Do the quantization."""
        if self.quantizer is None:
            self.init_quantizer(quant_config=quant_config, provider=provider)
        if self._eval_func is not None:
            self.quantizer.eval_func = self._eval_func
        if self._calib_func is not None:
            self.quantizer.calib_func = self._calib_func
        if self.quant_config.approach == QuantizationMode.POSTTRAININGSTATIC.value:
            assert self._calib_dataloader is not None, \
                "Please pass calib_dataloader to NoTrainerOptimizer.calib_dataloader"
            self.quantizer.calib_dataloader = self._calib_dataloader
        elif self.quant_config.approach == QuantizationMode.QUANTIZATIONAWARETRAINING.value:
            assert self._train_func is not None, \
                "Please pass train_func to NoTrainerOptimizer.train_func"
            self.quantizer.q_func = self._train_func
        self.opt_model = self.quantizer.fit()
        self.enable_inc_quant = True
        self.save_model(self.output_dir)
        return self.opt_model.model

    def quantize(
        self,
        quant_config: QuantizationConfig = None,
        provider: str = Provider.INC.value,
        eval_func: Optional[Callable] = None,
        train_func: Optional[Callable] = None,
        calib_func: Optional[Callable] = None,
        calib_dataloader=None,
    ):
        """Prepare for invoking the _inc_quantize function.
        
        Args:
            quant_config: quantization config.
            provider: define the quantization provider.
            eval_func: evaluation function.
            train_func: train function.
            calib_func: calibration function.
            calib_dataloader: calibration dataloader.
        """
        if eval_func is not None:
            self._eval_func = eval_func
        if train_func is not None:
            self._train_func = train_func
        if calib_func is not None:
            self._calib_func = calib_func
        if calib_dataloader is not None:
            self._calib_dataloader = calib_dataloader

        if self.quantizer is None:
            self._provider = Provider[provider.upper()].value

        if self._provider == Provider.INC.value:
            return self._inc_quantize(quant_config=quant_config, provider=provider)
        else:
            assert False, "Unsupport provider:{}".format(self._provider)

    def init_pruner(
        self,
        pruning_config = None,
        provider: str = Provider.INC.value,
    ):
        """Init a Pruning object with config.
        
        Args:
            pruning_config: pruning config.
            provider: define the pruning provider.
        """
        from neural_compressor.experimental import Pruning
        self.pruning_config = pruning_config
        self.metrics = self.pruning_config.metrics
        self._provider = Provider[provider.upper()].value

        assert isinstance(self.pruning_config, PruningConfig), \
            "please pass a instance of PruningConfig to trainer.prune!"

        pruner = Pruning(self.pruning_config.inc_config)
        pruner.model = common.Model(self.model)

        self.pruner = pruner
        return pruner

    def prune(
        self,
        pruning_config = None,
        provider: str = Provider.INC.value,
        eval_func: Optional[Callable] = None,
        train_func: Optional[Callable] = None,
    ):
        """Do the pruning.
        
        Args:
            pruning_config: pruning config.
            provider: define the pruning provider.
            eval_func: evaluation function.
            train_func: train function.
        """
        if self.pruner is None:
            self.init_pruner(pruning_config=pruning_config, provider=provider)
        if eval_func is not None:
            self._eval_func = eval_func
        if train_func is not None:
            self._train_func = train_func

        self.pruner.eval_func = self._eval_func

        self.pruner.pruning_func = self._train_func

        self.opt_model = self.pruner.fit()
        self.save_model(self.output_dir)
        stats, sparsity = self.opt_model.report_sparsity()
        logger.info(stats)
        logger.info(sparsity)

        return self.opt_model.model

    def init_distiller(
        self,
        distillation_config,
        teacher_model,
        provider: str = Provider.INC.value,
    ):
        """Init a Distillation object with config and the teacher model.
        
        Args:
            distillation_config: distillation config.
            teacher_model: set the teacher model.
            provider: define the distillation provider.
        """
        from neural_compressor.experimental import Distillation, common
        assert isinstance(distillation_config, DistillationConfig), \
            "please pass a instance of PruningConfig to trainer.prune!"
        self.distillation_config = distillation_config
        self._provider = Provider[provider.upper()].value
        self.metrics = self.distillation_config.metrics
        self.teacher_model = teacher_model

        distiller = Distillation(self.distillation_config.inc_config)
        distiller.model = common.Model(self.model)
        distiller.teacher_model = common.Model(self.teacher_model)

        self.distiller = distiller
        return distiller

    def distill(
        self,
        distillation_config,
        teacher_model,
        provider: str = Provider.INC.value,
        eval_func: Optional[Callable] = None,
        train_func: Optional[Callable] = None,
    ):
        """Do the distillation.
        
        Args:
            distillation_config: distillation config.
            teacher_model: set the teacher model.
            provider: define the distillation provider.
            eval_func: evaluation function.
            train_func: train function.
        """
        if self.distiller is None:
            self.init_distiller(
                distillation_config=distillation_config,
                teacher_model=teacher_model,
                provider=provider
            )
        if eval_func is not None:
            self._eval_func = eval_func
        if train_func is not None:
            self._train_func = train_func

        self.distiller.eval_func = self._eval_func
        self.distiller.train_func = self._train_func
        self.distiller.create_criterion()

        self.opt_model = self.distiller.fit()
        self.save_model(self.output_dir)
        return self.opt_model.model

    def _save_inc_int8(self, opt_model, output_dir):
        """Save the optimized model in the output directory.
        
        Args:
            opt_model: optimized model.
            output_dir: output path.
        """
        self.model.config.architectures = [self.model.__class__.__name__]
        self.model.config.torch_dtype = "int8"
        if isinstance(self.model.config, PretrainedConfig):
            self.model.config.save_pretrained(output_dir)
        weights_file = os.path.join(os.path.abspath(
          os.path.expanduser(output_dir)), WEIGHTS_NAME)
        torch.save(opt_model.quantized_state_dict(), weights_file)

    def save_model(self, output_dir, tokenizer=None):
        """Save the model and tokenizer in the output directory.

        Args:
            output_dir: the path to save config.json and pytorch_model.bin.
            tokenizer (object, optional): the tokenizer object, use it if you want to 
                                          save tokenizer.json in output_dir. Defaults to None.
        """
        os.makedirs(shlex.quote(output_dir), exist_ok=True)
        torch.save(self.opt_model.quantized_state_dict(), os.path.join(shlex.quote(output_dir), WEIGHTS_NAME))
        if self.enable_inc_quant and self.opt_model:
            self._save_inc_int8(self.opt_model, output_dir)
        else:
            self.model.save_pretrained(output_dir)
            self.model.config.save_pretrained(output_dir)
        if tokenizer:   # pragma: no cover
            tokenizer.save_pretrained(output_dir)
        logger.info("Optimized model and configure file have saved to {}".format(
                    output_dir))
