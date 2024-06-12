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
"""The trainer class for pytorch framework, to easily train or finetune a model."""
import collections
import inspect
import math
import numpy as np
import os
import copy
import sys
import time
import json
import warnings
from functools import partial
from neural_compressor import __version__ as nc_version
from neural_compressor.utils import logger
from intel_extension_for_transformers.transformers import (
    DistillationConfig,
    Provider,
    PruningMode,
    # QuantizationConfig,
    # QuantizationMode,
    PruningConfig,
    DynamicLengthConfig,
    BenchmarkConfig,
)
from neural_compressor.training import prepare_compression
from neural_compressor.quantization import fit
from neural_compressor.config import (
    PostTrainingQuantConfig,
    QuantizationAwareTrainingConfig,
    TuningCriterion
)
from intel_extension_for_transformers.transformers.benchmark import benchmark
from intel_extension_for_transformers.transformers.utils.metrics import Metric
from intel_extension_for_transformers.transformers.utils.utility import LazyImport
from packaging import version
from tqdm.auto import tqdm
from transformers import __version__, Seq2SeqTrainer, Trainer, PreTrainedModel
from transformers.configuration_utils import PretrainedConfig
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
from transformers.utils import (
    CONFIG_NAME,
    WEIGHTS_NAME,
    is_torch_tpu_available,
    is_sagemaker_mp_enabled,
)
# Integrations must be imported before ML frameworks:
from transformers.integrations import hp_params
from transformers.modeling_utils import unwrap_model
from transformers.trainer import TRAINER_STATE_NAME, TRAINING_ARGS_NAME
from transformers.trainer_callback import TrainerState
from transformers.trainer_pt_utils import (
    IterableDatasetShard,
    find_batch_size,
    nested_numpify,
)
from transformers.trainer_utils import (
    HPSearchBackend,
    TrainOutput,
    EvalLoopOutput,
    EvalPrediction,
    get_last_checkpoint,
    set_seed,
    speed_metrics,
    denumpify_detensorize,
)
from typing import Any, Callable, Dict, List, Optional, Union
from .dynamic.drop_and_restore_utils import (
    sample_length_configuration,
    sample_layer_configuration,
)
from .dynamic.evolution import (
    Evolution, approx_ratio, inverse, store2str
)

from torch.nn import KLDivLoss
import torch.nn.functional as F

# pylint: disable=E0611
if version.parse(nc_version).release < version.parse("2.0").release:
    from neural_compressor.model.torch_model import PyTorchIpexModel as IPEXModel
else:
    from neural_compressor.model.torch_model import IPEXModel

amp = LazyImport('apex.amp')
datasets = LazyImport('datasets')
optuna = LazyImport('optuna')
onnx = LazyImport('onnx')
ort = LazyImport('onnxruntime')
ortq = LazyImport('onnxruntime.quantization')
# pylint: disable=E1102
smp_forward_backward = LazyImport('transformers.trainer_pt_utils.smp_forward_backward')
torch = LazyImport("torch")
torchprofile = LazyImport("torchprofile")
xm = LazyImport('torch_xla.core.xla_model')
timeit = LazyImport('timeit')

if version.parse(__version__) < version.parse("4.30"):
    NEW_DEEPSPEED_FLAG = False
else:
    NEW_DEEPSPEED_FLAG = True


class BaseTrainer():
    """The base class of trainer."""
    def __init__(self, *args, **kwargs):
        """Initialization function.

        Args:
            args: defined parameters.
            kwargs: additional keyword arguments used to hide deprecated arguments.
        """
        super().__init__(*args, **kwargs)
        self.in_training = False
        self._provider = "inc"
        self._eval_func = None
        self._train_func = None
        self.teacher_model = None
        self._calib_dataloader = None
        self._resuming_checkpoint = None
        self.compression_ctrl = None
        self.component = None
        self.enable_inc_quant = False
        self.pruner = None
        self.quantizer = None
        self.distiller = None
        self.fp32_model = None
        self.opt_model = None
        # This flag is set for the engine in the export_to_int8_onnx API.
        self.enable_executor = False
        self.enable_bf16 = False
        self.orchestrate_opt = False
        self.orchestrate_opt_pruning = False
        self.dynamic_config = None

    @property
    def resuming_checkpoint(self):
        """Getter of the resuming checkpoint."""
        return self._resuming_checkpoint

    @resuming_checkpoint.setter
    def resuming_checkpoint(self, path: str):
        self._resuming_checkpoint = path

    @property
    def provider(self):
        """Getter of the provider."""
        return self._provider

    @property
    def eval_func(self):
        """Getter of the evaluation function."""
        return self._eval_func

    @property
    def train_func(self):
        """Getter of the training function."""
        return self._train_func

    @property
    def calib_dataloader(self):
        """Getter of the calibration dataloader."""
        return self._calib_dataloader

    @provider.setter
    def provider(self, prov: str):
        self._provider = prov

    @eval_func.setter
    def eval_func(self, func: Callable):
        self._eval_func = func

    @train_func.setter
    def train_func(self, func: Callable):
        self._train_func = func

    @calib_dataloader.setter
    def calib_dataloader(self, dataloader):
        self._calib_dataloader = dataloader

    def builtin_eval_func(self, model):
        """Custom Evaluate function to inference the model for specified metric on validation dataset.

        Args:
            model: The model to evaluate.

        Returns:
            [float]: evaluation result, the larger is better.
        """
        self.model = model
        # pylint: disable=E1101
        if self.args.seed:
            torch.manual_seed(self.args.seed)
        results = self.evaluate()
        logger.info(results)
        task_name = self.eval_dataset.config_name
        if "wikitext" in task_name:
            metric_name = "eval_loss"
        elif task_name == "stsb":
            metric_name = "eval_person"
        elif task_name == "cola":
            metric_name = "eval_matthews_correlation"
        else:
            metric_name = "eval_accuracy"
        assert metric_name in results.keys(), \
                "Please set metric from {}".format(results.keys())
        result = results.get(metric_name)
        logger.info("Throughput: {} samples/sec".format(results.get("eval_samples_per_second")))
        return result

    # pylint: disable=E1101
    def builtin_train_func(self, model):
        """Custom training function to train the model on training dataset.

        Args:
            model: The model to train.

        Returns:
            [float]: evaluation result, the larger is better.
        """
        self.model_wrapped = model
        self.model = model
        train_result = self.train(component=self.component,
                                  resume_from_checkpoint=self._resuming_checkpoint)
        metrics = train_result.metrics
        if not self.orchestrate_opt:
            self.save_model()  # Saves the tokenizer too for easy upload
        self.log_metrics("train", metrics)
        self.save_metrics("train", metrics)
        self.save_state()
        return self.model

    def _inc_quantize(
        self,
        quant_config,
        provider: str = Provider.INC.value,
    ):
        try:
            # we do deepcopy to keep the fp32 model for the export_to_int8_onnx API.
            self.fp32_model = copy.deepcopy(self.model)
        except Exception as e:  # pragma: no cover
            logger.warning("Model deepcopy failed: {}!".format(repr(e)))
        if isinstance(quant_config, PostTrainingQuantConfig):
            self.opt_model = fit(self.model,
                                 conf=quant_config,
                                 calib_dataloader=self.get_train_dataloader(),
                                 eval_func=self._eval_func)
        else:
            compression_manager = prepare_compression(self.model, quant_config)
            compression_manager.callbacks.on_train_begin()
            self.train()
            compression_manager.callbacks.on_train_end()
            self.opt_model = compression_manager.model
        self.enable_inc_quant = True
        self.save_model(self.args.output_dir)
        return self.opt_model.model

    def quantize(
        self,
        quant_config: Union[PostTrainingQuantConfig, QuantizationAwareTrainingConfig] = None,
        provider: str = Provider.INC.value,
        eval_func: Optional[Callable] = None,
        train_func: Optional[Callable] = None,
        calib_dataloader=None,
    ):
        """The main entry point of automatic quantization tuning.

        Args:
            quant_config: The path to the YAML configuration file or QuantizationConfig class containing
                accuracy goal, quantization objective and related dataloaders etc.
            provider: The provider used to quantize.
            eval_func (:obj:`Callable`, optional): The function used to evaluate the model.
            train_func (:obj:`Callable`, optional): The function used to train the model.
            calib_dataloader: The dataloader for calibration dataset.

        Returns:
            An objective of neural_compressor Quantization class, which can automativally searches for
            optimal quantization recipes for low precision model inference and achieving best tuning
            objectives.
        """
        self._eval_func = self.builtin_eval_func if eval_func is None else eval_func
        self._train_func = self.builtin_train_func if train_func is None else train_func
        if calib_dataloader is not None:
            self._calib_dataloader = calib_dataloader

        if self.quantizer is None:
            self._provider = Provider[provider.upper()].value

        if self._provider == Provider.INC.value:
            return self._inc_quantize(quant_config=quant_config, provider=provider)
        else:
            assert False, "Unsupported provider:{}".format(self._provider)

    def _save_inc_int8(self, opt_model, output_dir):
        weights_file = os.path.join(os.path.abspath(os.path.expanduser(output_dir)), WEIGHTS_NAME)
        if isinstance(opt_model, IPEXModel):
            try:
                with open(os.path.join(output_dir, "best_configure.json"), 'w') as f:
                    json.dump(opt_model.tune_cfg, f, indent = 4)
            except IOError as e:
                logger.error("Fail to save ipex configure file due to {}.".format(e))
            opt_model.model.save(weights_file)
            self.model_config.backend = "ipex"
            self.model_config.save_pretrained(output_dir)
        else:
            if hasattr(self.model, "config") and hasattr(self.model.config, "save_pretrained"):
                self.model.config.architectures = [self.model.__class__.__name__]
                self.model.config.torch_dtype = "int8"
                self.model.config.save_pretrained(output_dir)
            torch.save(opt_model.quantized_state_dict(), weights_file)
        logger.info("quantized model and configure file have saved to {}".format(output_dir))

    def init_pruner(
        self,
        pruning_config=None,
        provider: str = Provider.INC.value,
    ):
        """Initialize the pruner.

        Args:
            pruning_config: The path to the YAML configuration file or PruningConf class containing
            accuracy goal, pruning objective and related dataloaders etc.
            provider: The provider used to quantize.

        Returns:
            An objective of neural_compressor Pruning class.
        """

        from neural_compressor.experimental import Pruning
        self.pruning_config = pruning_config
        self.metrics = self.pruning_config.metrics
        self._provider = Provider[provider.upper()].value

        assert isinstance(self.pruning_config, PruningConfig), \
            "please pass a instance of PruningConfig to trainer.prune!"

        pruning_start_epoch, pruning_end_epoch = self.pruning_config.epoch_range

        # pylint: disable=E1101
        if pruning_start_epoch > self.args.num_train_epochs - 1:
            logger.warning(f"Pruning end epoch {pruning_start_epoch} is higher than "
                           f"the total number of training epoch "
                           f"{self.args.num_train_epochs}. No pruning will be applied.")

        # pylint: disable=E1101
        if pruning_end_epoch > self.args.num_train_epochs - 1:
            logger.warning(
                f"Pruning end epoch {pruning_end_epoch} is higher than "
                f"the total number of training epoch "
                f"{self.args.num_train_epochs}. The target sparsity will not be reached.")

        pruner = Pruning(self.pruning_config.inc_config)
        pruner.model = self.model

        self.pruner = pruner
        return pruner

    def prune(
        self,
        pruning_config=None,
        provider: str = Provider.INC.value,
        eval_func: Optional[Callable] = None,
        train_func: Optional[Callable] = None,
    ):
        """The main entry point of automatic quantization tuning.

        Args:
            pruning_config: The path to the YAML configuration file or PruningConf class containing
            accuracy goal, pruning objective and related dataloaders etc.
            provider (str): The provider used to quantize.
            eval_func (:obj:`Callable`, optional): The function used to evaluate the model.
            train_func (:obj:`Callable`, optional): The function used to train the model.

        Returns:
            An objective of neural_compressor Pruning class.
        """
        if self.pruner is None:
            self.init_pruner(pruning_config=pruning_config, provider=provider)
        if eval_func is not None:
            self._eval_func = eval_func
        if train_func is not None:
            self._train_func = train_func

        if self._eval_func is not None:
            self.pruner.eval_func = self._eval_func
        else:
            assert self.metrics is not None, "Please pass metrics to trainer.pruning.metrics!"
            assert self.pruning_config.pruner_config[0].prune_type == PruningMode.BASICMAGNITUDE.value, \
                "Please pass eval_func to trainer.eval_func"
            self.pruner.eval_func = self.builtin_eval_func

        if self._train_func is not None:
            self.pruner.pruning_func = self._train_func
        else:
            assert self.pruning_config.pruner_config[0].prune_type == PruningMode.BASICMAGNITUDE.value, \
                "Please pass train_func to trainer.train_func"
            self.pruner.pruning_func = self.builtin_train_func

        self.component = self.pruner
        self.opt_model = self.pruner.fit()
        stats, sparsity = self.opt_model.report_sparsity()
        logger.info(stats)
        logger.info(sparsity)

        return self.opt_model.model

    def init_distiller(
        self,
        distillation_config,
        teacher_model: Union[PreTrainedModel, torch.nn.Module],
        provider: str = Provider.INC.value,
    ):
        """The main entry point of automatic distillation tuning.

        Args:
            quant_config: The path to the YAML configuration file or DistillationConfig class containing.
            accuracy goal, distillation objective and related dataloaders etc.
            teacher_model: The model(torch.nn.Module) transfers knowledge to a smaller model.
            provider (str): The provider used to quantize.

        Returns:
            An objective of neural_compressor Distillation class.
        """
        from neural_compressor.experimental import Distillation
        assert isinstance(distillation_config, DistillationConfig), \
            "please pass a instance of PruningConfig to trainer.prune!"
        self.distillation_config = distillation_config
        self._provider = Provider[provider.upper()].value
        self.metrics = self.distillation_config.metrics
        self.teacher_model = teacher_model

        distiller = Distillation(self.distillation_config.inc_config)
        distiller.model = self.model
        distiller.teacher_model = self.teacher_model

        self.distiller = distiller
        return distiller

    def distill(
        self,
        distillation_config,
        teacher_model: Union[PreTrainedModel, torch.nn.Module],
        provider: str = Provider.INC.value,
        eval_func: Optional[Callable] = None,
        train_func: Optional[Callable] = None,
    ):
        """The main entry point of automatic distillation tuning.

        Args:
            quant_config: The path to the YAML configuration file or DistillationConfig class containing
            accuracy goal, distillation objective and related dataloaders etc.
            teacher_model: The model(torch.nn.Module) transfers knowledge to a smaller model.
            provider (str): The provider used to quantize.
            eval_func (:obj:`Callable`, optional: The function to evaluate the model.
            train_func (:obj:`Callable`, optional: The function to train the model.

        Returns:
            An objective of neural_compressor Distillation class.
        """
        if self.distiller is None:
            self.init_distiller(distillation_config=distillation_config,
                                teacher_model=teacher_model,
                                provider=provider)
        if eval_func is not None:
            self._eval_func = eval_func
        if train_func is not None:
            self._train_func = train_func

        if self._eval_func is not None:
            self.distiller.eval_func = self._eval_func
        else:
            assert self.metrics is not None, \
                "Please pass metrics to trainer.distillation.metrics!"
            self.distiller.eval_func = self.builtin_eval_func

        self.distiller.train_func = \
            self.builtin_train_func if self._train_func is None else self._train_func
        self.distiller.create_criterion()
        self.component = self.distiller
        self.opt_model = self.distiller.fit()

        return self.opt_model.model

    def orchestrate_optimizations(
        self,
        config_list,
        teacher_model: Optional[Callable] = None,
        eval_func: Optional[Callable] = None,
        train_func: Optional[Callable] = None,
    ):
        """Main entry point for orchestrate optimizations.

        Args:
            config_list: The list of configs.
            teacher_model (:obj:`Callable`, optional): The model(torch.nn.Module) transfers knowledge
                to a smaller model.
            eval_func (:obj:`Callable`, optional): Evaluation function to evaluate the tuning objective.
            train_func (:obj:`Callable`, optional): Training function which will be combined with pruning.
        """
        from intel_extension_for_transformers.transformers.optimizer import Orchestrate_optimizer
        self.orchestrate_opt = True
        self._eval_func = self.builtin_eval_func if eval_func is None else eval_func
        self._train_func = self.builtin_train_func if train_func is None else train_func
        components = self.create_optimizer_builtin(config_list, teacher_model)
        self.orchestrate_optimizer = Orchestrate_optimizer(self.model, components, \
                                     eval_func=self.eval_func, train_func=self.train_func, \
                                     output_dir=self.args.output_dir)
        self.component = self.orchestrate_optimizer.scheduler.components[0]
        torch_model = self.orchestrate_optimizer.fit()
        return torch_model

    def create_optimizer_builtin(self, config_list, teacher_model=None):
        """The function to create optimizer.

        Args:
            config_list: The list of configs.
            teacher_model (:obj:`Callable`, optional): The model(torch.nn.Module) transfers knowledge
                to a smaller model.
        """
        components = []
        for config in config_list:
            if isinstance(config, PostTrainingQuantConfig):
                component = self.init_quantizer(config)
                component.eval_func = self._eval_func
                component.q_func = self._train_func
                self.enable_inc_quant = True
            elif isinstance(config, PruningConfig):
                self.orchestrate_opt_pruning = True
                component = self.init_pruner(config)
                component.eval_func = self._eval_func
                component.pruning_func = self._train_func
            elif isinstance(config, DistillationConfig):
                assert isinstance(teacher_model, torch.nn.Module), \
                        "The teacher_model is needed for distiller"
                component = self.init_distiller(config, teacher_model)
                component.eval_func = self._eval_func
                component.train_func = self._train_func
                component.create_criterion()
            else:  # pragma: no cover
                assert False, "Orchestrate_optimizations config_list requires at least one" \
                    "       `QuantizationConfig`, `PruningConfig` or `DistillationConfig` object"
            components.append(component)
        return components

    # pylint: disable=E1101
    def training_step(
            self, model: torch.nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:  # pragma: no cover
        """Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (:obj:`nn.Module`): The model to train.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`): The inputs and targets of the model.
                The dictionary will be unpacked before being fed to the model. Most models expect
                the targets under the argument :obj:`labels`. Check your model's documentation for
                all accepted arguments.

        Return:
            :obj:`torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        if is_sagemaker_mp_enabled():
            # pylint: disable=E0401
            if version.parse(__version__) < version.parse("4.20"):
                scaler = self.scaler if self.use_amp else None
                loss_mb = smp_forward_backward(model,
                                               inputs,
                                               self.args.gradient_accumulation_steps,
                                               scaler=scaler)
            else:
                loss_mb = smp_forward_backward(model, inputs,
                                               self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        # pylint: disable=E0401
        if version.parse(__version__) < version.parse("4.20"):
            if self.use_amp:
                from torch.cuda.amp import autocast
                with autocast():
                    loss = self.compute_loss(model, inputs)
            else:
                loss = self.compute_loss(model, inputs)
        else:
            # pylint: disable=E0401
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if not NEW_DEEPSPEED_FLAG:
            if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
                # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
                loss = loss / self.args.gradient_accumulation_steps

        if self.compression_ctrl is not None:
            compression_loss = self.compression_ctrl.loss()
            loss += compression_loss

        # pylint: disable=E0401
        if version.parse(__version__) < version.parse("4.20"):
            if self.use_amp:
                self.scaler.scale(loss).backward()
            elif self.use_apex:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            elif self.deepspeed:
                # loss gets scaled under gradient_accumulation_steps in deepspeed
                loss = self.deepspeed.backward(loss)
            else:
                loss.backward()
        else:
            if self.use_apex:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            elif NEW_DEEPSPEED_FLAG:
                self.accelerator.backward(loss)
                loss / self.args.gradient_accumulation_steps
            elif self.deepspeed:
                # loss gets scaled under gradient_accumulation_steps in deepspeed
                loss = self.deepspeed.backward(loss)
            else:
                loss.backward()

        return loss.detach()


    def training_step_length_adaptive(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]]

    ) -> torch.Tensor:  # pragma: no cover
        """Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (:obj:`nn.Module`): The model to train.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`): The inputs and targets of the model.
                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.

        Return:
            :obj:`torch.Tensor`: The tensor with training loss on this batch.
        """

        model.train()
        inputs = self._prepare_inputs(inputs)
        tr_loss_sum = 0.0

        # compute loss of full model

        # pylint: disable=E0401
        if version.parse(__version__) < version.parse("4.20"):
            if self.use_amp:
                from torch.cuda.amp import autocast
                with autocast():
                    loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
            else:
                loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
        else:
            # pylint: disable=E0401
            with self.compute_loss_context_manager():
                loss, outputs = self.compute_loss(model, inputs, return_outputs=True)

        start_logits = outputs[0].detach()
        end_logits = outputs[1].detach()

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if not NEW_DEEPSPEED_FLAG:
            if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
                # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
                loss = loss / self.args.gradient_accumulation_steps

        if self.compression_ctrl is not None:  # TODO- should be added here?
            compression_loss = self.compression_ctrl.loss()
            loss += compression_loss

        loss = loss / (self.dynamic_config.num_sandwich + 2)
        tr_loss_sum += loss

        ## backward

        # pylint: disable=E0401
        if version.parse(__version__) < version.parse("4.20"):
            if self.use_amp:
                self.scaler.scale(loss).backward()
            elif self.use_apex:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            elif self.deepspeed:
                # loss gets scaled under gradient_accumulation_steps in deepspeed
                loss = self.deepspeed.backward(loss)
            else:
                loss.backward()
        else:
            if self.use_apex:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            elif NEW_DEEPSPEED_FLAG:
                self.accelerator.backward(loss)
                loss / self.args.gradient_accumulation_steps
            elif self.deepspeed:
                # loss gets scaled under gradient_accumulation_steps in deepspeed
                loss = self.deepspeed.backward(loss)
            else:
                loss.backward()

        # inplace distillation
        for i in range(self.dynamic_config.num_sandwich + 1):
            ## prepare inputs for sub-models
            num_h_layers = model.config.num_hidden_layers if hasattr(model, "config") else \
                                model.module.config.num_hidden_layers

            layer_config = sample_layer_configuration(
                num_h_layers,
                layer_dropout_prob=self.dynamic_config.layer_dropout_prob,
                layer_dropout=(self.dynamic_config.layer_dropout_bound if i == 0 else None),
                layer_dropout_bound=self.dynamic_config.layer_dropout_bound,
            )
            inputs["layer_config"] = layer_config

            length_config = sample_length_configuration(
                self.dynamic_config.max_length,
                num_h_layers,
                layer_config,
                length_drop_ratio=(self.dynamic_config.length_drop_ratio_bound if i == 0 else None),
                length_drop_ratio_bound=self.dynamic_config.length_drop_ratio_bound,
            )
            inputs["layer_config"] = layer_config
            inputs["length_config"] = length_config
            inputs["output_attentions"] = True


            # Compute inplace distillation loss

            # pylint: disable=E0401
            if version.parse(__version__) < version.parse("4.20"):
                if self.use_amp:
                    from torch.cuda.amp import autocast
                    with autocast():
                        _ , outputs_sub = self.compute_loss(model, inputs, return_outputs=True)
                else:
                    _ , outputs_sub = self.compute_loss(model, inputs, return_outputs=True)
            else:
                # pylint: disable=E0401
                with self.compute_loss_context_manager():
                    _ , outputs_sub = self.compute_loss(model, inputs, return_outputs=True)

            start_logits_sub = outputs_sub[0]
            end_logits_sub = outputs_sub[1]

            loss_fct = KLDivLoss(reduction="batchmean")
            start_kl_loss = loss_fct(F.log_softmax(start_logits, -1), F.softmax(start_logits_sub, -1))
            end_kl_loss = loss_fct(F.log_softmax(end_logits, -1), F.softmax(end_logits_sub, -1))
            loss = (start_kl_loss + end_kl_loss) / 2

            if self.args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training

            if not NEW_DEEPSPEED_FLAG:
                if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
                    # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
                    loss = loss / self.args.gradient_accumulation_steps

            if self.compression_ctrl is not None: # TODO- should be added here?
                compression_loss = self.compression_ctrl.loss()
                loss += compression_loss

            loss = loss / (self.dynamic_config.num_sandwich + 2)
            tr_loss_sum += loss


            ## backward

            # pylint: disable=E0401

            if version.parse(__version__) < version.parse("4.20"):
                if self.use_amp:
                    self.scaler.scale(loss).backward()
                elif self.use_apex:
                    with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                elif self.deepspeed:
                    # loss gets scaled under gradient_accumulation_steps in deepspeed
                    loss = self.deepspeed.backward(loss)
                else:
                    loss.backward()
            else:
                if self.use_apex:
                    with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                elif NEW_DEEPSPEED_FLAG:
                    self.accelerator.backward(loss)
                    loss / self.args.gradient_accumulation_steps
                elif self.deepspeed:
                    # loss gets scaled under gradient_accumulation_steps in deepspeed
                    loss = self.deepspeed.backward(loss)
                else:
                    loss.backward()


        return tr_loss_sum.detach()

    # pylint: disable=E1101
    def compute_loss(self, model, inputs, return_outputs=False):  # pragma: no cover
        """How the loss is computed by Trainer.

        By default, all models return the loss in the first element.

        Subclass and override for custom behavior.

        Args:
            model (:obj:`nn.Module`): The target model to compute the loss.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`): The inputs and targets of the model.
        """
        labels = inputs.pop("labels") \
            if self.label_smoother is not None and "labels" in inputs else None

        teacher_logits = inputs.pop("teacher_logits") if "teacher_logits" in inputs else None

        outputs = model(**inputs)

        if self.in_training and hasattr(self, "component") and \
           hasattr(self.component, "criterion"):
            qa_output_merger = lambda outputs: torch.vstack([
                torch.vstack([sl, el])
                for sl, el in zip(outputs["start_logits"], outputs["end_logits"])
            ])
            qa_output_spliter = lambda outputs: (outputs[0::2], outputs[1::2])

            def get_logits(outputs):
                if isinstance(outputs, dict):
                    if "logits" in outputs:
                        logits = outputs["logits"]
                    elif "start_logits" in outputs and "end_logits" in outputs:
                        logits = qa_output_merger(outputs)
                    elif "prediction_logits" in outputs:
                        logits = outputs["prediction_logits"]
                    else:
                        raise AssertionError("Logits of outputs not included, can't compute loss")
                elif isinstance(outputs, torch.Tensor):
                    logits = outputs
                else:
                    logits = outputs[1]
                return logits

            if teacher_logits is not None:
                if "start_positions" in inputs and "end_positions" in inputs:  # for SQuAD
                    teacher_logits = torch.vstack(list(teacher_logits))
            else:
                teacher_outputs = self.component.criterion.teacher_model_forward(inputs)
                teacher_logits = get_logits(self.component.criterion.teacher_outputs
                                            if teacher_outputs is None else teacher_outputs)

            logits = get_logits(outputs)
            if version.parse(nc_version) <= version.parse("1.12"):
                if labels is None:
                    if "labels" in inputs:  # for GLUE
                        labels = inputs["labels"]
                    elif "start_positions" in inputs and "end_positions" in inputs:  # for SQuAD
                        labels = torch.hstack([torch.tensor([sp, ep]) for sp, ep in \
                                zip(inputs["start_positions"], inputs["end_positions"])])
                    else:
                        raise AssertionError(
                            "Labels of input data not provided, can't compute loss")
                if hasattr(self.component, "on_post_forward"):
                    self.component.on_post_forward(inputs, teacher_output=teacher_logits)
                    if hasattr(self.component.criterion, "teacher_outputs"):
                        self.component.criterion.teacher_outputs = \
                            get_logits(self.component.criterion.teacher_outputs)
                loss = self.component.criterion(logits, labels)
                if hasattr(self.component.criterion, 'add_origin_loss') and \
                    self.component.criterion.add_origin_loss:
                    loss = loss + outputs['loss']
            else:
                if self.args.past_index >= 0:
                    self._past = outputs[self.args.past_index]

                if labels is not None:
                    loss = self.label_smoother(outputs, labels)
                else:
                    # We don't use .loss here since the model may return tuples instead of ModelOutput.
                    loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
                loss = self.component.on_after_compute_loss(inputs, logits, loss, teacher_logits)
            if "start_positions" in inputs and "end_positions" in inputs:
                start_logits, end_logits = qa_output_spliter(logits)
                outputs = {"start_logits": start_logits, "end_logits": end_logits, "loss": loss}
            else:
                outputs = {"logits": logits, "loss": loss}
        else:
            # Save past state if it exists
            # TODO: this needs to be fixed and made cleaner later.
            if self.args.past_index >= 0:
                self._past = outputs[self.args.past_index]

            if labels is not None:
                loss = self.label_smoother(outputs, labels)
            else:
                # We don't use .loss here since the model may return tuples instead of ModelOutput.
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss

    def _remove_unused_columns(self,
                               dataset: "datasets.Dataset",
                               description: Optional[str] = None):  # pragma: no cover
        # pylint: disable=E1101
        if not self.args.remove_unused_columns:
            return dataset
        if self._signature_columns is None:  # pylint: disable=E0203
            # Inspect model forward signature to keep only the arguments it accepts.
            signature = inspect.signature(self.model.forward)
            self._signature_columns = list(signature.parameters.keys())
            # Labels may be named label or label_ids, the default data collator handles that.
            self._signature_columns += ["label", "label_ids", "teacher_logits"]
        columns = [k for k in self._signature_columns if k in dataset.column_names]
        ignored_columns = list(set(dataset.column_names) - set(self._signature_columns))
        if len(ignored_columns) > 0:
            dset_description = "" if description is None else f"in the {description} set "
            logger.info(
                f"The following columns {dset_description} don't have a corresponding argument in "
                f"`{self.model.__class__.__name__}.forward` and have been ignored: {', '.join(ignored_columns)}."
            )

        if version.parse(datasets.__version__) < version.parse("1.4.0"):
            dataset.set_format(type=dataset.format["type"],
                               columns=columns,
                               format_kwargs=dataset.format["format_kwargs"])
            return dataset
        else:
            return dataset.remove_columns(ignored_columns)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        # pylint: disable=E1101
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(self.model, PreTrainedModel) \
          and not isinstance(self.model, IPEXModel):  # pragma: no cover
            unwrapped_model = unwrap_model(self.model)
            is_pretrained = isinstance(unwrapped_model, PreTrainedModel)

            if is_pretrained:
                if state_dict is None:
                    state_dict = unwrapped_model.state_dict()
                unwrapped_model.save_pretrained(output_dir, state_dict=state_dict,
                                                    safe_serialization=self.args.save_safetensors)
            else:
                logger.info(
                    "Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
                if state_dict is None:
                    state_dict = self.model.state_dict()
                torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
        else:
            # overwrite `pytorch_model.bin` with inc int8 format.
            if self.enable_inc_quant and self.opt_model:
                self._save_inc_int8(self.opt_model, output_dir)
            else:
                self.model.save_pretrained(output_dir, state_dict=state_dict,
                                                safe_serialization=self.args.save_safetensors)
        if self.tokenizer is not None:  # pragma: no cover
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        # pylint: disable=E1101
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

    def export_to_onnx(self, *args, **kwargs):
        """The function to transfer model into onnx model.

        Args:
            args: defined parameters.
            kwargs: additional keyword arguments used to hide deprecated arguments.
        """
        if self.enable_bf16:
            self.export_to_bf16_onnx(*args, **kwargs)
        elif not self.enable_inc_quant:
            self.export_to_fp32_onnx(*args, **kwargs)
        else:
            self.export_to_int8_onnx(*args, **kwargs)

    def export_to_fp32_onnx(
        self,
        save_path=None,
        opset_version=14,
        do_constant_folding=True,
        verbose=True,
    ):
        """The function to transfer model into fp32 onnx model.

        Args:
            save_path: the save path of the exported model.
            opset_version: the onnx op version of the exported model.
            do_constant_folding: select to do constant folding or not.
            verbose: save onnx model.
        """
        if self.fp32_model is None:
            model = self.model.eval()
        else:
            # Quantized model cannot be converted into onnx
            model = self.fp32_model.eval()
        # pylint: disable=E1101
        onnx_save_path = save_path if save_path \
          else os.path.join(self.args.output_dir, 'fp32-model.onnx')

        # get export args
        input, input_names, output_names, axes_dict = self.get_export_args(model)

        torch.onnx.export(
            model,
            (input, ),
            onnx_save_path,
            opset_version=opset_version,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=axes_dict,
            do_constant_folding=do_constant_folding,
        )
        if verbose:
            info = "The ONNX Model is exported to path: {0}".format(onnx_save_path)
            logger.info("*" * len(info))
            logger.info(info)
            logger.info("*" * len(info))

    def export_to_bf16_onnx(
        self,
        save_path=None,
        opset_version=14,
        do_constant_folding=True,
        verbose=True,
    ):
        """The function to transfer model into bf16 onnx model.

        Args:
            save_path: the save path of the exported model.
            opset_version: the onnx op version of the exported model.
            do_constant_folding: select to do constant folding or not.
            verbose: save onnx model.
        """
        fp32_path = save_path + '.tmp' if save_path \
          else os.path.join(self.args.output_dir, 'bf16-model.onnx.tmp')
        onnx_save_path = save_path if save_path \
          else os.path.join(self.args.output_dir, 'bf16-model.onnx')
        self.export_to_fp32_onnx(
            save_path=fp32_path,
            opset_version=opset_version,
            do_constant_folding=do_constant_folding,
            verbose=False,
        )

        model = onnx.load(fp32_path)
        bf16_type_list = ['MatMul', 'Gemm']
        bf16_tensor_name_list = []

        for node in model.graph.node:
            if node.op_type in bf16_type_list:
                for inp in node.input:
                    bf16_tensor_name_list.append(inp)

        from onnx import TensorProto, helper, numpy_helper
        for tensor in model.graph.initializer:
            if tensor.name in bf16_tensor_name_list:

                def fp32_to_bf16(fp32_np):
                    assert (fp32_np.dtype == np.float32)
                    int32_np = fp32_np.view(dtype=np.int32)
                    int32_np = int32_np >> 16
                    bf16_np = int32_np.astype(np.int16)
                    return bf16_np

                fp16_data = fp32_to_bf16(numpy_helper.to_array(tensor))
                tensor.raw_data = fp16_data.tobytes()
                tensor.data_type = TensorProto.BFLOAT16
        onnx.save(model, onnx_save_path)
        if os.path.isfile(fp32_path):
            os.remove(fp32_path)

        if verbose:
            info = "The ONNX Model is exported to path: {0}".format(onnx_save_path)
            logger.info("*" * len(info))
            logger.info(info)
            logger.info("*" * len(info))

    def export_to_int8_onnx(
        self,
        save_path=None,
        quant_format='QDQ',
        dtype='S8S8',
        opset_version=14,
        sample_size=100,
        calibrate_method='minmax',
        scale_mapping=False,
    ):
        """The function to transfer model into int8 onnx model.

        Args:
            save_path: the save path of the exported model.
            quant_format: quantization format.
            dtype: the quantized op type.
            opset_version: the onnx op version of the exported model.
            sample_size: the sampling size to calibrate the min-max range of ops.
            calibrate_method: the calibration method for onnx export.
            scale_mapping: make scale mapping of pytorch model and onnx model.
        """
        if self.provider != 'inc':  # pragma: no cover
            logger.error("export_to_onnx API only supports INC model right now.")
            sys.exit(0)

        if self.enable_executor: # pragma: no cover
            # Will deprecate after engine supports QDQ format and other op_types.
            op_types_to_quantize = ['MatMul']
            pytorch_op_types_to_quantize = ['Linear']
            addition_op_to_quantize = []
            opset_version = 13
            quant_format = 'QDQ'
            dtype = 'U8S8'
            logger.info("Engine only support opset_version=11 " + "and int8 MatMul.")
        else:
            if 'dynamic' in self.opt_model.q_config['approach']:
                op_types_to_quantize = ['MatMul', 'Gather', "LSTM", 'Conv']
                pytorch_op_types_to_quantize = ['Linear', 'Embedding', "LSTM", 'Conv1d', 'Conv2d']
                addition_op_to_quantize = list(ortq.registry.IntegerOpsRegistry.keys())
            else:
                op_types_to_quantize = ['MatMul', 'Gather', 'Conv']
                pytorch_op_types_to_quantize = ['Linear', 'Embedding', 'Conv1d', 'Conv2d']
                if quant_format == 'QDQ':
                    addition_op_to_quantize = list(ortq.registry.QDQRegistry.keys())
                    addition_op_to_quantize.remove('Relu')  # ValueError: x not in list
                else:
                    addition_op_to_quantize = list(ortq.registry.QLinearOpsRegistry.keys())

        if quant_format == 'QDQ' and opset_version < 13:  # pragma: no cover
            opset_version = 13
            logger.warning("QDQ format requires opset_version >= 13, " +
                           "we reset opset_version={} here".format(opset_version))
        all_op_types_to_quantize = op_types_to_quantize + addition_op_to_quantize

        # pylint: disable=E1101
        fp32_path = save_path + '.tmp' if save_path \
          else os.path.join(self.args.output_dir, 'int8-model.onnx.tmp')
        # pylint: disable=E1101
        onnx_save_path = save_path if save_path \
          else os.path.join(self.args.output_dir, 'int8-model.onnx')
        self.export_to_fp32_onnx(fp32_path,
                                 opset_version=opset_version,
                                 do_constant_folding=False,
                                 verbose=False)
        # Fix onnx accuracy drop when transformers > 4.21.0
        if version.parse(__version__) > version.parse("4.21.0"):
            from onnx import TensorProto
            model = onnx.load(fp32_path)
            for node in model.graph.node:
                if node.op_type == 'Constant' and len(node.attribute) != 0:
                    constant_value = onnx.numpy_helper.to_array(node.attribute[0].t)
                    if constant_value.shape == () and \
                        constant_value == torch.finfo(torch.float32).min:
                        new_tensor = onnx.helper.make_tensor(
                            name=node.output[0],
                            data_type=TensorProto.FLOAT,
                            dims=[],
                            vals=[-10000],
                        )
                        model.graph.initializer.append(new_tensor)
                        model.graph.node.remove(node)
            onnx.save(model, fp32_path)
        model = onnx.load(fp32_path)

        int8_model_dict = {}
        if self.opt_model.q_config['approach'] == 'quant_aware_training':
            # collect weights, bias from int8 QAT PT model
            model_dict = self.opt_model.model.state_dict()
            for name, param in model_dict.items():
                # '_packed_params._packed_weight' is specific for quantized Embedding
                if '_packed_params._packed_weight' in name:
                    name = name.replace('._packed_params._packed_weight', '').split('.module')[0]
                    int8_model_dict[name + '.weight'] = param.dequantize()
                # '_packed_params._packed_params' is specific for quantized Linear
                elif '_packed_params._packed_params' in name and isinstance(param, tuple):
                    name = name.replace('._packed_params._packed_params', '').split('.module')[0]
                    int8_model_dict[name + '.bias'] = param[1]
                    int8_model_dict[name + '.weight'] = param[0].dequantize()
                # '.weight' and '.bias' is specific for quantized Conv
                elif '.weight' in name:
                    int8_model_dict[name] = param.dequantize()
                elif '.bias' in name:
                    int8_model_dict[name] = param
                else:
                    int8_model_dict[name] = param

            # replace weight and bias in onnx fp32 model for QAT
            from onnx import helper
            tensor_list = [tensor for tensor in model.graph.initializer]
            for tensor in tensor_list:
                if tensor.name in int8_model_dict:
                    np_tensor = int8_model_dict[tensor.name].detach().cpu().numpy()
                    new_tensor = helper.make_tensor(
                        name=tensor.name,
                        data_type=tensor.data_type,
                        dims=tensor.dims,
                        vals=np_tensor,
                    )
                    model.graph.initializer.remove(tensor)
                    model.graph.initializer.append(new_tensor)
            onnx.save(model, fp32_path)

        if scale_mapping and \
          self.opt_model.q_config['approach'] != 'post_training_dynamic_quant':  # pragma: no cover
            # get output scale and zp from module
            import torch.nn.quantized.modules as q_modules
            for name, module in self.opt_model.model.named_modules():
                if isinstance(module, q_modules.Conv1d) or \
                  isinstance(module, q_modules.Conv2d) or \
                  isinstance(module, q_modules.Linear):
                    int8_model_dict[name] = {
                        'output_scale': module.scale,
                        'output_zeropoint': module.zero_point,
                    }

            # a name mapping to avoid '_' and '.' mismatch, we only use '.'.
            new_name_mapping = {}
            for name in int8_model_dict.keys():
                new_name = name.replace("_", '.')
                new_name_mapping.update({new_name: name})

            # get input scale and zp from q_config
            for name, value in self.opt_model.q_config['get_attr'].items():
                node_name, node_target = name.split('--')
                if 'scale' in name:
                    value_dict = {'input_scale': value}
                if 'zero_point' in name:
                    value_dict = {'input_zeropoint': value}
                tmp_name = node_name + '.' + node_target.split('_input_')[0]
                tmp_name = tmp_name.replace("_", '.')
                node_name = new_name_mapping[tmp_name]
                int8_model_dict[node_name].update(value_dict)

        from neural_compressor.adaptor.onnxrt import ONNXRUNTIMEAdaptor
        # pylint: disable=E1120
        inc_model = ONNXRUNTIMEAdaptor._replace_gemm_with_matmul(model)
        model = inc_model.model
        onnx.save(model, fp32_path)

        # Get weight name from onnx initializer
        weight_name_list = []
        for tensor in model.graph.initializer:
            weight_name_list.append(tensor.name)

        # Match weight name with onnx node name with fp32 model
        quantize_nodes = []
        tmp_node_mapping = {}
        module_node_mapping = {}
        for node in model.graph.node:
            if node.op_type not in op_types_to_quantize:
                for inp in node.input:
                    if inp in weight_name_list and 'weight' in inp:
                        tmp_node_mapping.update({node.output[0]: inp.split('.weight')[0]})
                    elif inp in tmp_node_mapping:
                        tmp_node_mapping.update({node.output[0]: tmp_node_mapping[inp]})
            else:
                for inp in node.input:
                    if inp in weight_name_list and 'weight' in inp:
                        module_node_mapping.update({inp.split('.weight')[0]: node.name})
                    elif inp in tmp_node_mapping:
                        module_node_mapping.update({tmp_node_mapping[inp]: node.name})

            # Save all quantizable node name
            if node.op_type in all_op_types_to_quantize:
                quantize_nodes.append(node.name)

        # Match pytorch module name with onnx node name for fallbacked fp32 module
        for k, v in self.opt_model.q_config['op'].items():  # pragma: no cover
            if k[1] not in pytorch_op_types_to_quantize or 'int8' in v['weight']['dtype']:
                continue
            k_0 = k[0].split('.module')[0] if k[0] not in module_node_mapping else k[0]
            if k_0 in module_node_mapping:
                fallback_op = module_node_mapping[k_0]
                quantize_nodes.remove(fallback_op)

        # Quantization
        quant_format = ortq.QuantFormat.QOperator if quant_format != 'QDQ' else ortq.QuantFormat.QDQ

        if 'U8U8' in dtype:  # pragma: no cover
            activation_type = ortq.QuantType.QUInt8
            weight_type = ortq.QuantType.QUInt8
        elif 'S8S8' in dtype:
            activation_type = ortq.QuantType.QInt8
            weight_type = ortq.QuantType.QInt8
        elif 'U8S8' in dtype: # pragma: no cover
            if not self.enable_executor:
                logger.error("Right now, we don't support dtype: {}, please use \
                              U8U8/S8S8 or set trainer.enable_executor=True \
                              for U8S8.".format(dtype))
                sys.exit(0)
            activation_type = ortq.QuantType.QUInt8
            weight_type = ortq.QuantType.QInt8
        else:  # pragma: no cover
            # Gather requires weight type be the same as activation.
            # So U8S8(activation|weight) option is not workable for best performance.
            logger.error("Right now, we don't support dtype: {}, \
                          please use U8U8/U8S8/S8S8.".format(dtype))
            sys.exit(0)
        logger.info("Weight type: {}.".format(weight_type))
        logger.info("Activation type: {}.".format(activation_type))

        # Calibrate_method, min/max method as default.
        if 'minmax' in calibrate_method:
            calibrate_method = ortq.CalibrationMethod.MinMax
        elif 'percentile' in calibrate_method: # pragma: no cover
            calibrate_method = ortq.CalibrationMethod.Percentile
        elif 'entropy' in calibrate_method: # pragma: no cover
            calibrate_method = ortq.CalibrationMethod.Entropy

        if 'dynamic' in self.opt_model.q_config['approach']:
            ortq.quantize_dynamic(
                fp32_path,
                onnx_save_path,
                per_channel=True,
                weight_type=weight_type,
                nodes_to_quantize=quantize_nodes,
                nodes_to_exclude=[],
                #op_types_to_quantize=op_types_to_quantize,
                extra_options={})
        else:

            class NLPDataReader(ortq.CalibrationDataReader):
                def __init__(self, dataloader, sample_size=sample_size):
                    import math
                    self.dataloader = dataloader
                    self.batch_size = dataloader.batch_size
                    self.batch_num = math.ceil(sample_size / self.batch_size)
                    self.datasize = self.batch_num * self.batch_size

                    self.data = []
                    for i, batch in enumerate(self.dataloader):
                        if i * self.batch_size >= self.datasize:
                            break
                        NLPTrainer._remove_label(batch)
                        batch = {k: v.detach().cpu().numpy() for k, v in batch.items()}
                        self.data.append(batch)
                    self.data = iter(self.data)

                def get_next(self):
                    return next(self.data, None)

            # pylint: disable=E1101
            calib_datareader = NLPDataReader(self.get_eval_dataloader())
            ortq.quantize_static(
                fp32_path,
                onnx_save_path,
                calib_datareader,
                quant_format=quant_format,
                per_channel=True,
                weight_type=weight_type,
                activation_type=activation_type,
                nodes_to_quantize=quantize_nodes,
                nodes_to_exclude=[],
                #op_types_to_quantize=op_types_to_quantize,
                calibrate_method=calibrate_method,
                extra_options={})

            if scale_mapping:  # pragma: no cover
                node_module_mapping = {}
                for module_name, node_name in module_node_mapping.items():
                    node_module_mapping[node_name] = module_name
                # match scale and zeropoint from PyTorch to ONNX node
                scale_zp_dict = {}
                for node in model.graph.node:
                    if node.name in node_module_mapping:
                        module_name = node_module_mapping[node.name]
                        if module_name not in int8_model_dict:
                            module_name = module_name + '.module'
                        if module_name in int8_model_dict:
                            recoder = int8_model_dict[module_name]
                            input_scale_args = node.input[0] + '_scale'
                            input_zp_args = node.input[0] + '_zero_point'
                            scale_zp_dict[input_scale_args] = recoder['input_scale']
                            scale_zp_dict[input_zp_args] = recoder['input_zeropoint']
                            # We need Matmul+Add to match Linear for output scale and zero-point
                            # output_scale_args = node.output[0] + '_scale'
                            # output_zp_args = node.output[0] + '_zero_point'
                            # scale_zp_dict[output_scale_args] = recoder['output_scale']
                            # scale_zp_dict[output_zp_args] = recoder['output_zeropoint']
                # set scale and zeropoint from PyTorch int8 model to ONNX int8 model
                from onnx import helper
                int8_model = onnx.load(onnx_save_path)
                tensor_list = [tensor for tensor in int8_model.graph.initializer]
                for tensor in tensor_list:
                    if tensor.name in scale_zp_dict:
                        value = scale_zp_dict[tensor.name]
                        if 'zero_point' in tensor.name and activation_type == ortq.QuantType.QInt8:
                            value -= 128
                        new_tensor = helper.make_tensor(
                            name=tensor.name,
                            data_type=tensor.data_type,
                            dims=tensor.dims,
                            vals=[value],
                        )
                        int8_model.graph.initializer.remove(tensor)
                        int8_model.graph.initializer.append(new_tensor)
                onnx.save(int8_model, onnx_save_path)

        if os.path.isfile(fp32_path):
            os.remove(fp32_path)

        info = "The ONNX Model is exported to path: {0}".format(onnx_save_path)
        logger.info("*" * len(info))
        logger.info(info)
        logger.info("*" * len(info))

    # pylint: disable=E1101
    def export_to_jit(self):
        """The function to transfer model into jit model."""
        self.model.eval()
        eval_dataloader = self.get_eval_dataloader()
        it = iter(eval_dataloader)
        input = next(it)
        self._remove_label(input)
        jit_model = torch.jit.trace(self.model, tuple(input.values()), strict=False)
        info = "JIT Model exported"
        logger.info("*" * len(info))
        logger.info(info)
        logger.info("*" * len(info))
        return jit_model

    def get_export_args(self, model):
        """Get input name, output names and axes for export."""
        from itertools import chain
        from optimum.exporters.tasks import TasksManager

        # prepare input data
        # pylint: disable=E1101
        eval_dataloader = self.get_eval_dataloader()
        it = iter(eval_dataloader)
        input = next(it)
        self._remove_label(input)

        # convert to a dict
        input = dict(input.items())

        if model.__class__.__name__ == 'XLNetForSequenceClassification': # pragma: no cover
            input.pop('token_type_ids')

        # set variable length axes
        symbolic_names = {0: 'batch_size', 1: 'max_seq_len'}
        axes_dict = {k: symbolic_names for k in input.keys()}

        # set input and output names
        input_names=list(input.keys())
        output_names=None

        task = self.infer_task(model)
        try:
            # try to get export config
            onnx_config_constructor = TasksManager.get_exporter_config_constructor(
                model=model, exporter="onnx", task=task
            )
            onnx_config = onnx_config_constructor(model.config)
            inputs = onnx_config.ordered_inputs(model)
            input_names = list(inputs.keys())
            output_names = list(onnx_config.outputs.keys())
            axes_dict = dict(chain(inputs.items(), onnx_config.outputs.items()))
        except:
            # skip and use settings collected from dataloader
            pass

        return input, input_names, output_names, axes_dict

    def infer_task(self, model):
        """Infer task."""
        from optimum.exporters.tasks import TasksManager

        if not hasattr(model, "config"):
            raise ValueError("model doesn't have 'config' attribute.")

        try:
            # infer task from model id
            model_name_or_path = model.config._name_or_path
            task = TasksManager.infer_task_from_model(model_name_or_path)
        except:
            try:
                # infer task from model itself
                task = TasksManager.infer_task_from_model(model)
            except:  # pragma: no cover
                try:
                    # infer task from model type
                    model_type = model.config.model_type.replace("_", "-")
                    tasks = TasksManager.get_supported_tasks_for_model_type(model_type, "onnx")
                    if len(tasks) != 0:
                        task = tasks[0]
                except:
                    raise ValueError("Could not infer the task.")

        return task

    @staticmethod
    def _remove_label(input):
        if "labels" in input:  # for GLUE
            input.pop('labels')
        elif "start_positions" in input and "end_positions" in input:  # for SQuAD
            # pragma: no cover
            input.pop('start_positions')
            input.pop('end_positions')
        return input

    def get_train_dataloader(self, *args, **kwargs):
        obj = super().get_train_dataloader(*args, **kwargs)
        if obj.batch_size is None:
            from .utils.utility import _build_inc_dataloader
            return _build_inc_dataloader(obj)
        else:
            return obj

    def get_eval_dataloader(self, *args, **kwargs):
        obj = super().get_eval_dataloader(*args, **kwargs)
        if obj.batch_size is None:
            from .utils.utility import _build_inc_dataloader
            return _build_inc_dataloader(obj)
        else:
            return obj

    def benchmark(
        self,
        model_name_or_path=None,
        backend: str = "torch",  # select from ["torch", "ipex", "neural_engine"]
        batch_size: int = 8,
        cores_per_instance: int = 4,
        num_of_instance: int = -1,
        torchscript: bool = False,
        generate: bool = False,
        **kwargs
    ):
        """Get performance of model.

        Args:
            backend (str, optional): Defaults to "torch".
            cores_per_instance (int, optional): Defaults to 4.
            num_of_instance (int, optional): Defaults to -1.
            torchscript (bool, optional):Defaults to False.
            generate (bool, optional): Defaults to False.
        """
        eval_dataloader = self.get_eval_dataloader()
        config = BenchmarkConfig(
            backend=backend,
            batch_size=batch_size,
            cores_per_instance=cores_per_instance,
            num_of_instance=num_of_instance,
            torchscript=torchscript,
            generate=generate,
        )
        config.kwargs = kwargs
        if model_name_or_path is None:
            model_name_or_path = self.model
        benchmark(model_name_or_path, config, example_inputs=None, dataloader=eval_dataloader)


    def set_dynamic_config(
        self,
        dynamic_config: DynamicLengthConfig,
    ):
        """The function to set dynamic config.

        Args:
            dynamic_config: the settings of the dynamic config.
        """
        self.dynamic_config = dynamic_config
        lc = None

        if self.dynamic_config.length_config is not None:
            import ast
            lc = ast.literal_eval(self.dynamic_config.length_config)
        else:
            assert self.dynamic_config.max_length is not None, \
            """
            Please set max_length DynamicLengthConfig
            """
            if self.dynamic_config.const_rate is not None:
                lc = sample_length_configuration(
                        self.dynamic_config.max_length,
                        self.model.config.num_hidden_layers,
                        length_drop_ratio=self.dynamic_config.const_rate,
                    )

        if lc is not None:
            # set the model with length config
            if self.model.config.model_type == "distilbert":
                bert = self.model.distilbert
            elif self.model.config.model_type == "roberta":
                bert = self.model.roberta
            else:
                assert hasattr(self.model, "bert")
                bert = self.model.bert

            print("setting length config to - " + str(lc))
            bert.set_length_config(lc)
            bert.set_output_attentions(True)


    def run_evolutionary_search(self):
        """Do evolutionary search."""
        assert self.dynamic_config is not None, \
            """
            Please set a DynamicLengthConfig to run evo-search
            """
        evolution = Evolution(self.model, self.dynamic_config.max_length, self.args.device, self.evaluate, \
                                eval_metric=self.dynamic_config.evo_eval_metric)
        # evolution.load_store(os.path.join(self.dynamic_config.model_name_or_path, 'store.tsv'))

        lower_gene = sample_length_configuration(
            self.dynamic_config.max_length,
            self.model.config.num_hidden_layers,
            length_drop_ratio=self.dynamic_config.length_drop_ratio_bound,
        )
        upper_gene = (self.dynamic_config.max_length,) * self.model.config.num_hidden_layers
        evolution.add_gene(lower_gene, method=0)
        evolution.add_gene(upper_gene, method=0)
        evolution.lower_constraint = evolution.store[lower_gene][0]
        evolution.upper_constraint = evolution.store[upper_gene][0]

        length_drop_ratios = [inverse(r) for r in \
                              np.linspace(approx_ratio(self.dynamic_config.length_drop_ratio_bound), \
                              1, self.dynamic_config.population_size + 2)[1:-1]]
        for p in length_drop_ratios:
            gene = sample_length_configuration(
                self.dynamic_config.max_length,
                self.model.config.num_hidden_layers,
                length_drop_ratio=p,
            )
            evolution.add_gene(gene, method=0)

        for i in range(self.dynamic_config.evo_iter + 1):
            logger.info(f"| Start Iteration {i}:")
            population, area = evolution.pareto_frontier()
            parents = evolution.convex_hull()
            results = {"area": area, "population_size": len(population), "num_parents": len(parents)}

            logger.info(f"| >>>>>>>> {' | '.join([f'{k} {v}' for k, v in results.items()])}")
            for gene in parents:  # population
                logger.info("| " + store2str(gene, *evolution.store[gene][:3]))

            evolution.save_store(os.path.join(self.args.output_dir, f'store-iter{i}.tsv'))
            evolution.save_population(os.path.join(self.args.output_dir, f'population-iter{i}.tsv'), population)
            evolution.save_population(os.path.join(self.args.output_dir, f'parents-iter{i}.tsv'), parents)

            if i == self.dynamic_config.evo_iter:
                break

            k = 0
            while k < self.dynamic_config.mutation_size:
                if evolution.mutate(self.dynamic_config.mutation_prob):
                    k += 1

            k = 0
            while k < self.dynamic_config.crossover_size:
                if evolution.crossover():
                    k += 1


class NLPTrainer(BaseTrainer, Trainer):
    """Trainer for nlp base on class BaseTrainer and Trainer form Transformers."""
    def __init__(self, *args, **kwargs):
        """Initialization function."""
        super().__init__(*args, **kwargs)


class NLPSeq2SeqTrainer(BaseTrainer, Seq2SeqTrainer):
    """Trainer for seq2seq model."""
    def __init__(self, *args, **kwargs):
        """Initialization function."""
        super().__init__(*args, **kwargs)
        self._max_length = None
        self._num_beams = None

    @property
    def max_length(self):
        """Getter of the max length."""
        return self._max_length

    @max_length.setter
    def max_length(self, max_length):
        self._max_length = max_length

    @property
    def num_beams(self):
        """Getter of the number of beams."""
        return self._num_beams

    @num_beams.setter
    def num_beams(self, num_beams):
        self._num_beams = num_beams

    def builtin_eval_func(self, model):
        """Custom Evaluate function to inference the model for specified metric on validation dataset.

        Args:
            model: The model to evaluate.

        Returns:
            evaluation result, the larger is better.
        """
        assert self.max_length is not None, \
            """
            Please set max_length in trainer, like as:
            trainer.max_length = xxx
            """
        logger.info("max_length = {}, num_beams = {}".format(self.max_length, self.num_beams))
        self.model = model
        # pylint: disable=E1101
        if self.args.seed:
            torch.manual_seed(self.args.seed)
        results = self.evaluate(metric_key_prefix="eval")
        logger.info(results)
        metric_name = "eval_bleu"
        assert metric_name in results.keys(), \
            "Please set metric from {}".format(results.keys())
        result = results.get(metric_name)
        logger.info("Throughput: {} samples/sec".format(results.get("eval_samples_per_second")))
        return result
