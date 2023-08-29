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

"""Pruning: specify the supported pruning mode."""

from packaging import version
from enum import Enum
from neural_compressor.conf.config import Pruner as INCPruner
from typing import Dict, List
from neural_compressor import __version__ as nc_version


class PruningMode(Enum):
    """Currently support three pruning modes."""
    BASICMAGNITUDE = "basic_magnitude"
    PATTERNLOCK = "pattern_lock"
    GROUPLASSO = "group_lasso"


SUPPORTED_PRUNING_MODE = set([approach.name for approach in PruningMode])


class PrunerConfig(INCPruner):
    """Pruner configuration."""
    def __init__(self, epoch_range: List=[0, 4], initial_sparsity_ratio: float=0.0,
                 target_sparsity_ratio: float=0.97, update_frequency: int=1,
                 prune_type: str='BasicMagnitude', method: str='per_tensor',
                 names: List=[], parameters: Dict=None):
        """Init the pruner config.

        Args:
            epoch_range: A list with length of 2. The first element is the start epoch and the second element
                is the end epoch. Pruning will be done from the start epoch to the end epoch.
            initial_sparsity_ratio: Initial sparsity goal
            target_sparsity_ratio: Target sparsity goal
            update_frequency: How many epochs to update once
            prune_type: "BasicMagnitude", "PatternLock", or "GroupLasso"
            method: TODO (Remove this parameter)
            names: A list of layer names that need to be pruned
            parameters: A dictionary of extra parameters
        """
        if epoch_range is not None:
            assert len(epoch_range) == 2, "Please set the epoch_range as [start_epoch, end_epoch]"
            self.start_epoch = epoch_range[0]
            self.end_epoch = epoch_range[1]
        else:  # pragma: no cover
            self.start_epoch = None
            self.end_epoch = None
        self.update_frequency = update_frequency
        self.target_sparsity = target_sparsity_ratio
        self.initial_sparsity = initial_sparsity_ratio
        self.update_frequency = update_frequency
        assert prune_type.upper() in SUPPORTED_PRUNING_MODE, \
            "prune_type only support {}!".format(
                [mode.lower() for mode in SUPPORTED_PRUNING_MODE]
            )
        self.prune_type = PruningMode[prune_type.upper()].value
        self.method = method
        self.names = names
        self.parameters = parameters



