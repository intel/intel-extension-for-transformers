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


from .config import (
    AutoDistillationConfig,
    DistillationConfig,
    FlashDistillationConfig,
    TFDistillationConfig,
    NASConfig,
    Provider,
    PruningConfig,
    QuantizationConfig,
    WEIGHTS_NAME,
    DynamicLengthConfig,
    BenchmarkConfig,
    PrunerV2,
    
)
from .distillation import (
    DistillationCriterionMode,
    SUPPORTED_DISTILLATION_CRITERION_MODE,
)
from .mixture.auto_distillation import AutoDistillation
from .model import OptimizedModel
from .nas import NAS
from .optimizer import NoTrainerOptimizer, Orchestrate_optimizer
from .optimizer_tf import TFOptimization
from .pruning import PrunerConfig, PruningMode, SUPPORTED_PRUNING_MODE
from .quantization import QuantizationMode, SUPPORTED_QUANT_MODE
from .utils import metrics
from .utils import objectives
from .utils.utility import LazyImport
