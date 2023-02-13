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

"""AutoDistillation: handling the whole pipeline of AutoDistillation for both pytorch and tensorflow framework."""

from pickletools import optimize
import numpy as np
import os
import random
import shutil
import tempfile

from functools import partial
from neural_compressor.conf.config import Conf, schema
from neural_compressor.conf.dotdict import DotDict
from neural_compressor.experimental import Distillation
from neural_compressor.experimental.nas.nas import NASBase
from neural_compressor.experimental.nas.nas_utils import find_pareto_front, NASMethods
from neural_compressor.experimental.nas.search_algorithms import \
    BayesianOptimizationSearcher, GridSearcher, RandomSearcher
from neural_compressor.strategy.bayesian import BayesianOptimization
from neural_compressor.utils import logger
from intel_extension_for_transformers.optimization.config import AutoDistillationConfig
from intel_extension_for_transformers.optimization.utils.utility import LazyImport

torch = LazyImport("torch")

class AutoDistillation(NASBase):
    """The framework class is designed for handling the whole pipeline of AutoDistillation.

    AutoDistillation is composed of three major stages, i.e. Model Exploration, Flash Distillation, 
    and Evaluation.
    In Model Exploration, a search engine will search for a better compressed model from the architecture 
    design space in each iteration.
    Flash Distillation is the stage for training the searched model to discover its potential.
    In Evaluation stage, the trained model will be evaluated to measure its performances (e.g. 
    the prediction accuracy, the hardware performance etc.) in order to select the best model architecture.
    """
    def __init__(self, model_builder, conf_fname_or_obj, framework='pytorch'):
        """Init an AutoDistillation instance base on config.

        Args:
        model_builder (function obj): A function to build model instance with the specified 
            model architecture parameters.
        conf_fname_or_obj (string or obj): The path to the YAML configuration file or
            a configuration object containing search setting, flash distillation settings, etc.
        framework: Specify the framework used as backend.
        """
        NASBase.__init__(self, search_space={}, model_builder=model_builder)
        self._train_func = None
        self._eval_func = None
        self.framework = framework.lower()
        self.init_by_cfg(conf_fname_or_obj)

    def search(self, res_save_path=None, model_cls=None):
        """Auto distillation search process.
        
        Returns:
            Best model architecture found in search process.
        """
        def reload_tf_model(model):
            with tempfile.TemporaryDirectory() as tmp_dir:
                model.save_pretrained(tmp_dir)
                assert model_cls, 'model_cls should not be None'
                model = model_cls.from_pretrained(tmp_dir)
            return model

        if res_save_path is None or not os.path.isdir(res_save_path):
            res_save_path = os.getcwd()
        save_path = os.path.join(res_save_path, 'AutoDistillationResults')
        self.model_paras_num = {}
        logger.info(f'search save_path = {save_path}')
        self.load_search_results(save_path)
        os.makedirs(save_path, exist_ok=True)

        for i in range(self.max_trials):
            logger.info(
                "{fix} Trial {n} starts, {r} trials to go {fix}".format(
                    n=i+1, r=self.max_trials-i-1, fix="="*30
                )
            )
            if self.framework == 'pytorch':
                if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                    model_arch_paras = self.select_model_arch()
                    logger.info("Model architecture {} proposed.".format(model_arch_paras))
                if torch.distributed.is_initialized():
                    model_arch_paras_sync = [model_arch_paras] \
                        if torch.distributed.get_rank() == 0 else [None]
                    torch.distributed.broadcast_object_list(model_arch_paras_sync, src=0)
                    model_arch_paras = model_arch_paras_sync[0]
            else:
                model_arch_paras = self.select_model_arch()
            model = self._model_builder(model_arch_paras)
            if self.framework == 'tensorflow':
                model = reload_tf_model(model)

            model_paras = self.count_model_parameters(model)
            logger.info(
                "***** Number of model parameters: {:.2f}M *****".format(model_paras / 10**6)
            )
            self.model_paras_num[tuple(model_arch_paras.values())] = model_paras
            if tuple(model_arch_paras.values()) in self.search_results:
                logger.info("Skip evaluated model architecture {}.".format(model_arch_paras))
                continue
            if tuple(model_arch_paras.values()) in self.resumed_search_results:
                logger.info(
                    "Find previous results of model architecture: {}.".format(model_arch_paras)
                )
                metrics = self.resumed_search_results[tuple(model_arch_paras.values())]
            else:
                logger.info("Assessing model architecture: {}.".format(model_arch_paras))
                metrics = self.estimate(model)
            logger.info(
                "Metrics of model architecture {} is {}.".format(model_arch_paras, metrics)
            )
            self.search_results[tuple(model_arch_paras.values())] = metrics

            if (self.framework != "pytorch" or not torch.distributed.is_initialized() 
              or torch.distributed.get_rank() == 0):
                self._search_algorithm.get_feedback(sum(self.metrics_conversion(metrics)))
                print(f'res_save_path: {res_save_path}, save_path = {save_path}')
                os.makedirs(save_path, exist_ok=True)
                self.dump_search_results(
                    os.path.join(save_path, 'Trial_{}_results.txt'.format(i + 1)))

        if self.framework != "pytorch" or not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            for model_arch_vec in self.resumed_search_results:
                if model_arch_vec not in self.search_results:
                    self.search_results[model_arch_vec] = \
                        self.resumed_search_results[model_arch_vec]
                    model = self._model_builder(self.params_vec2params_dict(model_arch_vec))
                    if self.framework == 'tensorflow':
                        model = reload_tf_model(model)
                    self.model_paras_num[model_arch_vec] = self.count_model_parameters(model)
            self.dump_search_results(os.path.join(save_path, 'Final_results.txt'.format(i+1)))
        self.find_best_model_archs()
        logger.info(
            "{fix} Found {n} best model architectures {fix}".format(
                n=len(self.best_model_archs), fix="="*30
            )
        )
        for i, model_arch in enumerate(self.best_model_archs):
            logger.info("Best model architecture {}: {}".format(i+1, model_arch))
        return self.best_model_archs

    def estimate(self, model):
        """Train and evaluate the model.

        Returns:
            Evaluated metrics of the model.
        """
        assert self._train_func is not None and self._eval_func is not None, \
            "train_func and eval_func must be set."
        model = self._train_func(model)
        return self._eval_func(model)

    def count_model_parameters(self, model):
        if self.framework == 'pytorch':
            if isinstance(model, torch.nn.Module):
                return sum(p.numel() for p in model.parameters())
            else: # pragma: no cover
                raise NotImplementedError("Only support torch model now.")
        elif self.framework == 'tensorflow':
            return model.num_parameters()

    def load_search_results(self, path):
        """Load previous search results.
        
        Args:
            path: The file path which stores the previous results.
        """
        self.resumed_search_results = {}
        lastest_results_record = os.path.join(path, 'lastest_results.npy')
        if not os.path.exists(path) or not os.path.exists(lastest_results_record):
            return
        self.resumed_search_results = np.load(lastest_results_record, allow_pickle=True).item()
        if self.framework == "pytorch" and torch.distributed.is_initialized():
            torch.distributed.barrier()
        if self.framework != "pytorch" or not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            os.makedirs(os.path.join(path, 'previous_results'), exist_ok=True)
            for f in os.listdir(path):
                if os.path.isfile(os.path.join(path, f)):
                    shutil.move(os.path.join(path, f), os.path.join(path, 'previous_results', f))
        logger.info("Loaded previous results.")

    def init_by_cfg(self, conf_fname_or_obj):
        """Use auto distillation config to init the instance of autodistillation."""
        if isinstance(conf_fname_or_obj, str): # pragma: no cover
            if os.path.isfile(conf_fname_or_obj):
                self.config = Conf(conf_fname_or_obj).usr_cfg
            else:
                raise FileNotFoundError(
                    "{} is not a file, please provide a file path.".format(conf_fname_or_obj)
                )
        elif isinstance(conf_fname_or_obj, AutoDistillationConfig):
            self.config = conf_fname_or_obj.config
            if self.framework == 'pytorch':
                schema.validate(self.config)
        else: # pragma: no cover
            raise NotImplementedError(
                "Please provide a str path to config file or a config dict."
            )
        assert self.config.auto_distillation is not None, "auto_distillation section must be set"
        auto_distillation_cfg = self.config.auto_distillation
        # search related
        self.init_search_cfg(auto_distillation_cfg)
        # flash distillation related
        self.flash_distillation_config = auto_distillation_cfg.flash_distillation

    def create_distillers(self):
        """Create flash and regular distillers."""
        def create_tf_distiller(distillation_cfg):
            block_names = distillation_cfg.block_names \
                if distillation_cfg.block_names else ['']
            train_steps = distillation_cfg.train_steps \
                if distillation_cfg.train_steps else [500]
            conf = DotDict({
                'model':{
                    'name': 'flash_distillation_{}'.format(0), 'framework': self.framework
                },
                'distillation':{
                    'train':{'optimizer': {'SGD':{'learning_rate': 1e-4}},
                            'criterion': {'KnowledgeDistillationLoss':
                                        {'temperature': distillation_cfg.temperature,
                                        'loss_types': distillation_cfg.loss_types,
                                        'loss_weights': distillation_cfg.loss_weights,
                                        }}}
                }
            })
            return [Distillation(conf)], block_names, train_steps

        def create_distiller(distillation_cfg):
            distillers = []
            assert distillation_cfg.layer_mappings_for_knowledge_transfer, \
                'layer_mappings_for_knowledge_transfer should be defined in flash_distillation'
            layer_mappings = \
                distillation_cfg.layer_mappings_for_knowledge_transfer
            block_names = distillation_cfg.block_names \
                if distillation_cfg.block_names else [''] * len(layer_mappings)
            loss_types = distillation_cfg.loss_types \
                if distillation_cfg.loss_types else [['MSE']*len(_) for _ in layer_mappings]
            loss_weights = distillation_cfg.loss_weights \
                if distillation_cfg.loss_weights else [[1.0 / len(_)] * len(_) \
                    for _ in layer_mappings]
            add_origin_loss = distillation_cfg.add_origin_loss \
                if distillation_cfg.add_origin_loss else [False] * len(layer_mappings)
            train_steps = distillation_cfg.train_steps \
                if distillation_cfg.train_steps else [500] * len(layer_mappings)
            temperatures = distillation_cfg.temperatures \
                if distillation_cfg.temperatures else [1.0] * len(layer_mappings)
            assert len(layer_mappings) == len(block_names) == len(loss_types) == \
                len(loss_weights) == len(add_origin_loss) == len(train_steps), \
                "lengths of layer_mappings_for_knowledge_transfer, block_names, " + \
                "loss_types, loss_weights, add_origin_loss and train_steps should be the same."
            distillers = []
            for i, lm in enumerate(layer_mappings):
                conf = DotDict({
                    'model':{
                        'name': 'flash_distillation_{}'.format(i), 'framework': self.framework
                    },
                    'distillation':{
                        'train':{'optimizer': {'SGD':{'learning_rate': 1e-4}},
                                'criterion': {'IntermediateLayersKnowledgeDistillationLoss':
                                            {'layer_mappings': lm,
                                                'loss_types': loss_types[i],
                                                'loss_weights': loss_weights[i],
                                                'add_origin_loss': add_origin_loss[i]}}}
                    }
                })
                distillers.append(Distillation(conf))
            return distillers, block_names, train_steps

        flash_distillation_config = self.flash_distillation_config
        # knowledge transfer related
        self.flash_distillers = []
        if flash_distillation_config.knowledge_transfer:
            knowledge_transfer_cfg = flash_distillation_config.knowledge_transfer
            if self.framework == 'pytorch':
                self.flash_distillers, self.flash_block_names, self.flash_train_steps = \
                    create_distiller(knowledge_transfer_cfg)
            else:
                self.flash_distillers, self.flash_block_names, self.flash_train_steps = \
                    create_tf_distiller(knowledge_transfer_cfg)

        # regular distillation related
        self.regular_distillers = []
        if flash_distillation_config.regular_distillation:
            regular_distillation_cfg = flash_distillation_config.regular_distillation
            if self.framework == 'pytorch':
                self.regular_distillers, self.regular_block_names, self.regular_train_steps = \
                    create_distiller(regular_distillation_cfg)
            else:
                self.regular_distillers, self.regular_block_names, self.regular_train_steps = \
                    create_tf_distiller(regular_distillation_cfg)

    @property
    def teacher_model(self):
        """Getter of teacher model."""
        return self._teacher_model

    @teacher_model.setter
    def teacher_model(self, user_model):
        """Setter of teacher model."""
        self._teacher_model = user_model

    @property
    def student_model(self):
        """Getter of student model."""
        return self._model

    @student_model.setter
    def student_model(self, user_model):
        """Setter of student model."""
        self._model = user_model

    @property
    def train_func(self):
        """Getter of train function."""
        return self._train_func

    @train_func.setter
    def train_func(self, train_func):
        """Setter of train function."""
        self._train_func = train_func

    @property
    def eval_func(self):
        """Getter of evaluation function."""
        return self._eval_func

    @eval_func.setter
    def eval_func(self, eval_func):
        """Setter of evaluation function."""
        self._eval_func = eval_func

    def __repr__(self): # pragma: no cover
        """Return class name."""
        return 'AutoDistillation'