"""Basic NAS approach class."""

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

import os

from intel_extension_for_transformers.optimization.config import NASConfig
from neural_compressor.conf.config import Conf
from neural_compressor.experimental.nas.nas import NASBase
from neural_compressor.experimental.nas.nas_utils import nas_registry

@nas_registry("Basic")
class BasicNAS(NASBase):
    """Basic NAS approach.

    Defining the pipeline for basic NAS approach.

    Args:
        conf_fname (string or obj): The path to the YAML configuration file or the object of NASConfig.
        search_space (dict): A dictionary for defining the search space.
        model_builder (function obj): A function to build model instance with the specified
            model architecture parameters.
    """

    def __init__(self, conf_fname_or_obj, search_space=None, model_builder=None):
        """Initialize the attributes."""
        NASBase.__init__(self, search_space=search_space, model_builder=model_builder)
        self._train_func = None
        self._eval_func = None
        self.init_by_cfg(conf_fname_or_obj)

    def execute(self, res_save_path=None):
        """Execute the search process.

        Returns:
            Best model architectures found in the search process.
        """
        return self.search(res_save_path)

    def estimate(self, model):
        """Estimate performance of the model.

        Depends on specific NAS algorithm. Here we use train and evaluate.

        Returns:
            Evaluated metrics of the model.
        """
        assert self._train_func is not None and self._eval_func is not None, \
            "train_func and eval_func must be set."
        self._train_func(model)
        return self._eval_func(model)

    def init_by_cfg(self, conf_fname_or_obj):
        """Initialize the configuration."""
        if isinstance(conf_fname_or_obj, str):
            if os.path.isfile(conf_fname_or_obj):
                self.conf = Conf(conf_fname_or_obj).usr_cfg
            else: # pragma: no cover
                raise FileNotFoundError(
                    "{} is not a file, please provide a NAS config file path.".format(
                        conf_fname_or_obj
                    )
                )
        elif isinstance(conf_fname_or_obj, NASConfig):
            self.config = conf_fname_or_obj.config
        else: # pragma: no cover
            raise NotImplementedError(
                "Please provide a str path to the config file or an object of NASConfig."
            )
        assert self.config.nas is not None, "nas section must be set"
        # search related config
        self.init_search_cfg(self.config.nas)

    @property
    def train_func(self): # pragma: no cover
        """Not support get train_func."""
        assert False, 'Should not try to get the value of `train_func` attribute.'

    @train_func.setter
    def train_func(self, user_train_func):
        """Training function.

        Args:
            user_train_func: This function takes "model" as input parameter
                         and executes entire training process with self
                         contained training hyper-parameters. If training_func set,
                         an evaluation process must be triggered and user should
                         set eval_dataloader with metric configured or directly eval_func
                         to make evaluation of the model executed. training_func will return
                         a trained model.
        """
        self._train_func = user_train_func

    @property
    def eval_func(self): # pragma: no cover
        """Not support get eval_func."""
        assert False, 'Should not try to get the value of `eval_func` attribute.'

    @eval_func.setter
    def eval_func(self, user_eval_func):
        """Eval function for component.

        Args:
            user_eval_func: This function takes "model" as input parameter
                         and executes entire evaluation process with self
                         contained metrics. If eval_func set,
                         an evaluation process must be triggered
                         to make evaluation of the model executed.
        """
        self._eval_func = user_eval_func

    def __repr__(self):
        """Class representation."""
        return 'BasicNAS' # pragma: no cover