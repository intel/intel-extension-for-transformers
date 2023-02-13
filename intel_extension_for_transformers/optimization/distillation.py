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
"""Distillation: set criterion mode to distillation."""
from enum import Enum
from typing import List


class Criterion(object):
    """Criterion class for distillation."""
    def __init__(
        self,
        name: str = "KNOWLEDGELOSS",
        temperature: float = 1.0,
        loss_types: List = ['CE', 'CE'],
        loss_weight_ratio: List = [0.5, 0.5],
        layer_mappings: List = None,
        add_origin_loss: bool = False
    ):
        """Init a Criterion object."""
        self.name = name
        self.temperature = temperature
        self.loss_types = loss_types
        self.loss_weight_ratio = loss_weight_ratio
        self.layer_mappings = layer_mappings
        self.add_origin_loss = add_origin_loss


class DistillationCriterionMode(Enum):
    """Criterion mode class for distillation."""
    KNOWLEDGELOSS = "KnowledgeDistillationLoss"
    INTERMEDIATELAYERSLOSS = "IntermediateLayersKnowledgeDistillationLoss"
    


SUPPORTED_DISTILLATION_CRITERION_MODE = \
    set([approach.name for approach in DistillationCriterionMode])
