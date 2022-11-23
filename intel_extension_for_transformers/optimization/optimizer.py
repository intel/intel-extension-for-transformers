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

import logging
import os

from neural_compressor.experimental import(
    common,
    Component,
    Distillation,
)
from neural_compressor.experimental.scheduler import Scheduler
from intel_extension_for_transformers import(
    DistillationConfig,
    Provider,
    QuantizationConfig,
    PruningConfig
)
from intel_extension_for_transformers.optimization.utils.utility import LazyImport
from intel_extension_for_transformers.optimization.quantization import QuantizationMode
from transformers import PreTrainedModel
from transformers.file_utils import WEIGHTS_NAME
from typing import Callable, Optional, Union, List

torch = LazyImport("torch")

logger = logging.getLogger(__name__)


class Orchestrate_optimizer:
    def __init__(
        self,
        model,
        components: Optional[List[Component]] = [],
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

        if len(components) > 1:
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



class NoTrainerOptimizer:   # pragma: no cover
    def __init__(
        self,
        model,
        output_dir: Optional[str] = "saved_results",
    ):
        """
        Args:
            model (:obj:`Union[PreTrainedModel, torch.nn.Module]`):
                FP32 model specified for low precision tuning.
            output_dir (:obj:`string`, `optional`):
                The folder for saving the results.
        """

        self.model = model
        self.teacher_model = None
        self._eval_func = None
        self._train_func = None
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

    @property
    def eval_func(self):
        return self._eval_func

    @property
    def train_func(self):
        return self._train_func

    @property
    def provider(self):
        return self._provider

    @property
    def calib_dataloader(self):
        return self._calib_dataloader

    @eval_func.setter
    def eval_func(self, func: Callable):
        self._eval_func = func

    @train_func.setter
    def train_func(self, func: Callable):
        self._train_func = func

    @provider.setter
    def provider(self, provider):
        self._provider = provider

    @calib_dataloader.setter
    def calib_dataloader(self, dataloader):
        self._calib_dataloader = dataloader

    def init_quantizer(
        self,
        quant_config,
        provider: str = Provider.INC.value,
    ):
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

    # pylint: disable=E0401
    def _nncf_quantize(self):   # pragma: no cover
        from intel_extension_for_transformers import NncfConfig
        from nncf import create_compressed_model
        compression_state = None
        assert isinstance(self.quant_config, NncfConfig), \
            "Please pass a NNCFConfig instance to trainer.quantize!"
        
        self.metrics = self.quant_config.metrics
        nncf_compression_state_file = self.quant_config.compression_state

        if os.path.isfile(nncf_compression_state_file):
            compression_state = torch.load(nncf_compression_state_file)
        else:
            compression_state = None

        compression_algo_controller, model = create_compressed_model(
            self.model, self.quant_config.nncf_config,
            compression_state=compression_state
        )

        self.compression_ctrl = \
            compression_algo_controller.distributed() \
            if self.quant_config.distributed else compression_algo_controller

        self.model = self._train_func(model)

    def _inc_quantize(
        self,
        quant_config,
        provider: str = Provider.INC.value,
    ):
        if self.quantizer is None:
            self.init_quantizer(quant_config=quant_config, provider=provider)
        if self._eval_func is not None:
            self.quantizer.eval_func = self._eval_func

        if self.quant_config.approach == QuantizationMode.POSTTRAININGSTATIC.value:
            assert self._calib_dataloader is not None, \
                "Please pass calib_dataloader to NoTrainerOptimizer.calib_dataloader"
            self.quantizer.calib_dataloader = self._calib_dataloader
        elif self.quant_config.approach == QuantizationMode.QUANTIZATIONAWARETRAINING.value:
            assert self._train_func is not None, \
                "Please pass train_func to NoTrainerOptimizer.train_func"
            self.quantizer.q_func = self._train_func
        self.opt_model = self.quantizer.fit()
        self.inc_int8_flag = True
        self._save_inc_int8(self.opt_model, self.output_dir)
        logger.info(
            "quantized model and configure file have saved to {}".format(self.output_dir)
        )
        return self.opt_model

    def quantize(
        self,
        quant_config: QuantizationConfig = None,
        provider: str = Provider.INC.value,
        eval_func: Optional[Callable] = None,
        train_func: Optional[Callable] = None,
        calib_dataloader=None,
    ):
        if eval_func is not None:
            self._eval_func = eval_func
        if train_func is not None:
            self._train_func = train_func
        if calib_dataloader is not None:
            self._calib_dataloader = calib_dataloader

        if self.quantizer is None:
            self._provider = Provider[provider.upper()].value

        if self._provider == Provider.NNCF.value:
            return self._nncf_quantize()
        elif self._provider == Provider.INC.value:
            return self._inc_quantize(quant_config=quant_config, provider=provider)
        else:
            assert False, "Unsupport provider:{}".format(self._provider)

    def init_pruner(
        self,
        pruning_config = None,
        provider: str = Provider.INC.value,
    ):
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
        if self.pruner is None:
            self.init_pruner(pruning_config=pruning_config, provider=provider)
        if eval_func is not None:
            self._eval_func = eval_func
        if train_func is not None:
            self._train_func = train_func

        self.pruner.eval_func = self._eval_func

        self.pruner.pruning_func = self._train_func

        self.opt_model = self.pruner.fit()

        return self.opt_model

    def init_distiller(
        self,
        distillation_config,
        teacher_model,
        provider: str = Provider.INC.value,
    ):
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

        return self.opt_model

    def _save_inc_int8(self, opt_model, output_dir):
        self.model.config.architectures = [self.model.__class__.__name__]
        self.model.config.torch_dtype = "int8"
        self.model.config.save_pretrained(output_dir)
        weights_file = os.path.join(os.path.abspath(
          os.path.expanduser(output_dir)), WEIGHTS_NAME)
        torch.save(opt_model.quantized_state_dict(), weights_file)
        logger.info(
            "quantized model and configure file have saved to {}".format(weights_file)
        )
