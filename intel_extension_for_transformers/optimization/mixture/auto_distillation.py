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

import numpy as np
import os
import random
import shutil

from functools import partial
from neural_compressor.conf.config import Conf, schema
from neural_compressor.conf.dotdict import DotDict
from neural_compressor.experimental import Distillation
from neural_compressor.strategy.bayesian import BayesianOptimization
from neural_compressor.utils import logger
from intel_extension_for_transformers.optimization.config import AutoDistillationConfig
from intel_extension_for_transformers.optimization.utils.utility import LazyImport

torch = LazyImport("torch")

def distributed_log_wrapper(func, msg):
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        func(msg)


class AutoDistillation(object):
    """

    Args:
        model_builder (function obj): A function to build model instance with the specified 
            model architecture parameters.
        conf_fname_or_obj (string or obj): The path to the YAML configuration file or
            a configuration object containing search setting, flash distillation settings, etc.

    """

    def __init__(self, model_builder, conf_fname_or_obj):
        self.search_space = {}
        self.model_builder = model_builder
        self._advisor = None
        self._train_func = None
        self._eval_func = None
        self.search_results = {}
        self.best_model_archs = None
        self.seed = None
        self.init_by_cfg(conf_fname_or_obj)

    def model_arch_proposition(self):
        """Propose architecture of the model based on search algorithm for next search iteration.

        Returns:
            Model architecture description.
        """
        model_arch_paras = self.advisor.suggestion()
        assert self.search_space_keys and isinstance(model_arch_paras, dict) and \
            self.search_space_keys == list(model_arch_paras.keys()), \
            "Keys of model_arch_paras should be the same with search_space_keys."
        return model_arch_paras

    def search(self, res_save_path=None):
        """AutoDistillation search process.
        
        Returns:
            Best model architecture found in search process.
        """
        if res_save_path is None or not os.path.isdir(res_save_path):
            res_save_path = os.getcwd()
        save_path = os.path.join(res_save_path, 'AutoDistillationResults')
        self.model_paras_num = {}
        self.load_search_results(save_path)
        os.makedirs(save_path, exist_ok=True)

        for i in range(self.max_trials):
            logger.info(
                "{fix} Trial {n} starts, {r} trials to go {fix}".format(
                    n=i+1, r=self.max_trials-i-1, fix="="*30
                )
            )
            if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                model_arch_paras = self.model_arch_proposition()
                logger.info("Model architecture {} proposed.".format(model_arch_paras))
            if torch.distributed.is_initialized():
                model_arch_paras_sync = [model_arch_paras] \
                    if torch.distributed.get_rank() == 0 else [None]
                torch.distributed.broadcast_object_list(model_arch_paras_sync, src=0)
                model_arch_paras = model_arch_paras_sync[0]
            model = self.model_builder(model_arch_paras)
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
            if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                self.advisor.feedback(sum(self.metrics_conversion(metrics)))
                self.dump_search_results(
                    os.path.join(save_path, 'Trial_{}_results.txt'.format(i+1))
                )

        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            for model_arch_vec in self.resumed_search_results:
                if model_arch_vec not in self.search_results:
                    self.search_results[model_arch_vec] = \
                        self.resumed_search_results[model_arch_vec]
                    model = self.model_builder(self.params_vec2params_dict(model_arch_vec))
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
        if isinstance(model, torch.nn.Module):
            return sum(p.numel() for p in model.parameters())
        else: # pragma: no cover
            raise NotImplementedError("Only support torch model now.")

    def load_search_results(self, path):
        self.resumed_search_results = {}
        lastest_results_record = os.path.join(path, 'lastest_results.npy')
        if not os.path.exists(path) or not os.path.exists(lastest_results_record):
            return
        self.resumed_search_results = np.load(lastest_results_record, allow_pickle=True).item()
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            os.makedirs(os.path.join(path, 'previous_results'), exist_ok=True)
            for f in os.listdir(path):
                if os.path.isfile(os.path.join(path, f)):
                    shutil.move(os.path.join(path, f), os.path.join(path, 'previous_results', f))
        logger.info("Loaded previous results.")

    def dump_search_results(self, path):
        lastest_results_record = os.path.join(os.path.dirname(path), 'lastest_results.npy')
        np.save(lastest_results_record, self.search_results, allow_pickle=True)
        write_contents = '=' * 30 + ' All Search Results ' + '=' * 30 + '\n\n'
        for model_arch_vec in self.search_results:
            tmp = ','.join(['{}_{}'.format(k, v) \
                for k, v in zip(self.search_space_keys, model_arch_vec)])
            write_contents += '{}: {} Paras: {}M\n'.format(
                tmp, self.search_results[model_arch_vec],
                self.model_paras_num[model_arch_vec] / 10**6
            )
        write_contents += '\n\n\n' + '=' * 30 + ' Best Search Results ' + '=' * 30 + '\n\n'
        self.find_best_model_archs()
        for i, model_arch in enumerate(self.best_model_archs):
            model_arch_vec = tuple(model_arch.values())
            tmp = ','.join(['{}_{}'.format(k, v) \
                for k, v in zip(self.search_space_keys, model_arch_vec)])
            write_contents += \
                '{}. {}: {} Paras: {}M\n'.format(
                    i+1, tmp, self.search_results[model_arch_vec],
                    self.model_paras_num[model_arch_vec] / 10**6
            )
        with open(path, mode='w') as f:
            f.write(write_contents)

    def params_vec2params_dict(self, paras_vec):
        assert len(paras_vec) == len(self.search_space_keys), \
            "Length of paras_vec and search_space_keys should be the same."
        return {k:v for k, v in zip(self.search_space_keys, paras_vec)}

    def find_best_model_archs(self):
        assert len(self.search_results) > 0, "Zero result in search_results."
        model_arches = list(self.search_results.keys())
        metrics = [self.metrics_conversion(self.search_results[ma]) for ma in model_arches]
        pareto_front_indices = find_pareto_front(metrics)
        self.best_model_archs = [self.params_vec2params_dict(model_arches[i]) \
            for i in pareto_front_indices]

    def metrics_conversion(self, metrics):
        if isinstance(metrics, dict):
            if self.metrics is None:
                self.metrics = list(metrics.keys())
            assert list(metrics.keys()) == list(self.metrics), \
                "Keys of metrics not match with metrics in the configuration."
            metrics = list(metrics.values()) 
        if self.higher_is_better is None:
            self.higher_is_better = [True,] * len(metrics)
            logger.warning("higher_is_better not set in the configuration, " + \
                "set it to all True for every metric entry by default.")
        converted_metrics = [metric if higher_is_better else -metric \
            for metric, higher_is_better in zip(metrics, self.higher_is_better)]
        return converted_metrics

    def init_by_cfg(self, conf_fname_or_obj):
        if isinstance(conf_fname_or_obj, str): # pragma: no cover
            if os.path.isfile(conf_fname_or_obj):
                self.config = Conf(conf_fname_or_obj).usr_cfg
            else:
                raise FileNotFoundError(
                    "{} is not a file, please provide a file path.".format(conf_fname_or_obj)
                )
        elif isinstance(conf_fname_or_obj, AutoDistillationConfig):
            self.config = conf_fname_or_obj.config
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

    def init_search_cfg(self, config):
        self.search_cfg = config.search
        self.search_space = self.search_cfg.search_space
        self.search_space_keys = sorted(self.search_space.keys())
        for k in self.search_space_keys:
            assert isinstance(self.search_space[k], (list, tuple)), \
                "Value of key \'{}\' must be a list or tuple".format(k)

        self.metrics = self.search_cfg.metrics \
            if self.search_cfg.metrics is not None else None
        self.higher_is_better = self.search_cfg.higher_is_better \
            if self.search_cfg.higher_is_better is not None else None
        self.seed = self.search_cfg.seed
        self.max_trials = self.search_cfg.max_trials \
            if self.search_cfg.max_trials is not None else 3 # set default 3 for max_trials
        self.search_algorithm = self.search_cfg.search_algorithm
        if self.search_algorithm is None:
            self.advisor = BayesianOptimizationSearcher(self.search_space, self.seed)
        elif self.search_algorithm.lower() == 'grid':
            self.advisor = GridSearcher(self.search_space)
        elif self.search_algorithm.lower() == 'random':
            self.advisor = RandomSearcher(self.search_space, self.seed)
        elif self.search_algorithm.lower() == 'bo':
            self.advisor = BayesianOptimizationSearcher(self.search_space, self.seed)
        else: # pragma: no cover
            raise NotImplementedError(
                'Unsupported \'{}\' search algorithm'.format(self.search_algorithm)
            )

    def create_distillers(self):
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
            assert len(layer_mappings) == len(block_names) == len(loss_types) == \
                len(loss_weights) == len(add_origin_loss) == len(train_steps), \
                "lengths of layer_mappings_for_knowledge_transfer, block_names, " + \
                "loss_types, loss_weights, add_origin_loss and train_steps should be the same."
            distillers = []
            for i, lm in enumerate(layer_mappings):
                conf = DotDict({
                    'model':{
                        'name': 'flash_distillation_{}'.format(i), 'framework': 'pytorch'
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
            self.flash_distillers, self.flash_block_names, self.flash_train_steps = \
                create_distiller(knowledge_transfer_cfg)

        # regular distillation related
        self.regular_distillers = []
        if flash_distillation_config.regular_distillation:
            regular_distillation_cfg = flash_distillation_config.regular_distillation
            self.regular_distillers, self.regular_block_names, self.regular_train_steps = \
                create_distiller(regular_distillation_cfg)

    @property
    def teacher_model(self):
        return self._teacher_model

    @teacher_model.setter
    def teacher_model(self, user_model):
        self._teacher_model = user_model

    @property
    def student_model(self):
        return self._model

    @student_model.setter
    def student_model(self, user_model):
        self._model = user_model

    @property
    def advisor(self):
        return self._advisor
    
    @advisor.setter
    def advisor(self, advisor):
        self._advisor = advisor

    @property
    def train_func(self):
        return self._train_func

    @train_func.setter
    def train_func(self, train_func):
        self._train_func = train_func

    @property
    def eval_func(self):
        return self._eval_func

    @eval_func.setter
    def eval_func(self, eval_func):
        self._eval_func = eval_func
    
    def __repr__(self): # pragma: no cover
        return 'AutoDistillation'

def create_search_space_pool(search_space, idx=0):
    search_space_keys = sorted(search_space.keys())
    if idx == len(search_space_keys):
        return [[]]
    key = search_space_keys[idx]
    search_space_pool = []
    for v in search_space[key]:
        sub_search_space_pool = create_search_space_pool(search_space, idx+1)
        search_space_pool += [[v] + item for item in sub_search_space_pool]
    return search_space_pool

class Searcher(object):
    def __init__(self, search_space) -> None:
        assert isinstance(search_space, dict) and search_space, \
            "Expect search_space to be a dict."
        self.search_space = search_space
        self.search_space_keys = sorted(search_space.keys())
        for k in self.search_space_keys:
            assert isinstance(self.search_space[k], (list, tuple)), \
                "Value of key \'{}\' must be a list or tuple to specify choices".format(k)

    def suggestion(self): # pragma: no cover
        raise NotImplementedError('Depends on specific search algorithm.')

    def feedback(self, metric):
        pass

    def params_vec2params_dict(self, para_vec):
        assert len(para_vec) == len(self.search_space_keys), \
            "Length of para_vec and search_space_keys should be the same."
        return {k: para_vec[i] for i, k in enumerate(self.search_space_keys)}

class GridSearcher(Searcher):
    def __init__(self, search_space) -> None:
        super(GridSearcher, self).__init__(search_space)
        self.search_space_pool = create_search_space_pool(search_space)
        self.idx = 0

    def suggestion(self):
        res = self.search_space_pool[self.idx]
        self.idx = (self.idx + 1) % len(self.search_space_pool)
        return self.params_vec2params_dict(res)

class RandomSearcher(Searcher):
    def __init__(self, search_space, seed=42) -> None:
        super(RandomSearcher, self).__init__(search_space)
        self.search_space_pool = create_search_space_pool(search_space)
        self.indices_pool = list(range(len(self.search_space_pool)))
        random.seed(seed)
        random.shuffle(self.indices_pool)

    def suggestion(self):
        if not self.indices_pool:
            self.indices_pool = list(range(len(self.search_space_pool)))
            random.shuffle(self.indices_pool)
        idx = self.indices_pool.pop(-1)
        return self.params_vec2params_dict(self.search_space_pool[idx])

class BayesianOptimizationSearcher(Searcher):
    def __init__(self, search_space, seed=42) -> None:
        super(BayesianOptimizationSearcher, self).__init__(search_space)
        idx_search_space = {k: (0, len(search_space[k])-1) for k in self.search_space_keys}
        self.bo_agent = BayesianOptimization(idx_search_space, random_seed=seed)
        self.last_param_indices = None

    def suggestion(self):
        param_indices = self.bo_agent.gen_next_params()
        self.last_param_indices = param_indices
        return self.params_vec2params_dict(self.indices2params_vec(param_indices))

    def feedback(self, metric):
        assert self.last_param_indices is not None, "Need run suggestion first " + \
            "to get parameters and the input metric is corresponding to this parameters."
        try:
            self.bo_agent._space.register(self.last_param_indices, metric)
        except KeyError: # pragma: no cover
            logger.debug("Find registered params, skip it.")
            pass
        self.last_param_indices = None

    def indices2params_vec(self, indices):
        res = []
        for key, ind in indices.items():
            # keep ind within the index range of self.search_space[key]
            ind = int(min(max(round(ind), 0), len(self.search_space[key])-1))
            res.append(self.search_space[key][ind])
        return res

def find_pareto_front(metrics):
    """
    Find the pareto front points, assuming all metrics are "higher is better".

    Args:
        metrics: An (n_points, n_metrics) array
    Return:
        An array of indices of pareto front points. 
        It is a (n_pareto_points, ) integer array of indices.
    """
    metrics = np.array(metrics)
    pareto_front_point_indices = np.arange(metrics.shape[0])
    next_point_idx = 0
    while next_point_idx < len(metrics):
        nondominated_points = np.any(metrics > metrics[next_point_idx], axis=1)
        nondominated_points[next_point_idx] = True
        # Remove points being dominated by current point
        pareto_front_point_indices = pareto_front_point_indices[nondominated_points]
        metrics = metrics[nondominated_points]
        next_point_idx = np.sum(nondominated_points[:next_point_idx+1])
    return pareto_front_point_indices
