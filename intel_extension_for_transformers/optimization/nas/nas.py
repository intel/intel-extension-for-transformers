"""Common classes for different NAS approaches.

NAS class for creating object of different NAS approaches.
NASBase class defines the common methods of different NAS approaches.
"""

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

import numpy as np
import os
import shutil

from collections.abc import Iterable
from intel_extension_for_transformers.optimization.config import NASConfig
from neural_compressor.conf.config import Conf
from neural_compressor.experimental.nas.nas_utils import find_pareto_front, NASMethods
from neural_compressor.experimental.nas.search_algorithms import \
    BayesianOptimizationSearcher, GridSearcher, RandomSearcher
from neural_compressor.utils.utility import logger, LazyImport

torch = LazyImport('torch')


class NAS(object):
    """Create object of different NAS approaches.

    Args:
        conf_fname_or_obj (string or obj):
            The path to the YAML configuration file or the object of NASConfig.

    Returns:
        An object of specified NAS approach.
    """

    def __new__(self, conf_fname_or_obj, *args, **kwargs):
        """Create an object of specified NAS approach."""
        if isinstance(conf_fname_or_obj, str) and os.path.isfile(conf_fname_or_obj):
            self.config = Conf(conf_fname_or_obj).usr_cfg
        elif isinstance(conf_fname_or_obj, NASConfig):
            self.config = conf_fname_or_obj.config
        else:  # pragma: no cover
            raise NotImplementedError(
                "Please provide a str path to the config file."
            )
        assert self.config.nas is not None, "nas section must be set"
        if isinstance(self.config.nas.approach, str) and \
                self.config.nas.approach.lower() in NASMethods:
            method = self.config.nas.approach.lower()
        else:  # pragma: no cover
            logger.warning(
                "NAS approach not set in config, use default NAS approach, i.e. Basic."
            )
            method = 'basic'
        return NASMethods[method](conf_fname_or_obj, *args, **kwargs)